from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.hooks import RemovableHandle

from .adapters import ModelAdapter, ProjectionSite
from .schema import (
    Bounds,
    BoundsSource,
    Correction,
    FaultSpec,
    InjectionTrace,
    ProtectionSpec,
    fp16_bits_at,
    flip_fp16_bits,
)


def _logical_sequence_length(
    output: torch.Tensor,
    site: ProjectionSite,
) -> int:
    if (
        output.ndim == 3
        and output.shape[0] == 1
        and output.shape[2] == site.out_features
    ):
        return int(output.shape[1])
    if output.ndim == 2 and output.shape[1] == site.out_features:
        return int(output.shape[0])
    raise RuntimeError(
        f"{site.key} returned unexpected shape {tuple(output.shape)}"
    )
@dataclass(frozen=True)
class EngineRecord:
    mode_id: str
    final_step: int
    injection_count: int
    injection_trace: Optional[InjectionTrace]
    hook_calls: Mapping[str, int]
    online_bounds: Mapping[str, Bounds]
    correction_elements: int
    nan_elements: int
    low_elements: int
    high_elements: int
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "final_step": self.final_step,
            "injection_count": self.injection_count,
            "injection_trace": (
                self.injection_trace.to_dict() if self.injection_trace else None
            ),
            "hook_calls": dict(self.hook_calls),
            "online_bounds": {
                key: bounds.to_dict()
                for key, bounds in sorted(self.online_bounds.items())
            },
            "online_bounds_key_count": len(self.online_bounds),
            "correction_elements": self.correction_elements,
            "nan_elements": self.nan_elements,
            "low_elements": self.low_elements,
            "high_elements": self.high_elements,
            "complete": self.complete,
        }


class FT2HookEngine:
    """One reusable hook installation with fresh state for every inference."""

    def __init__(
        self,
        adapter: ModelAdapter,
        protected_site_keys: Optional[Sequence[str]] = None,
    ):
        self.adapter = adapter
        all_keys = frozenset(site.key for site in adapter.sites)
        if protected_site_keys is None:
            requested = tuple(site.key for site in adapter.critical_sites)
        else:
            requested = tuple(protected_site_keys)
        if len(requested) != len(set(requested)):
            raise ValueError("protected_site_keys must not contain duplicates")
        unknown = set(requested).difference(all_keys)
        if unknown:
            raise ValueError(f"Unknown protected sites: {sorted(unknown)}")
        self._protected_keys = frozenset(requested)
        self._handles: list[RemovableHandle] = []
        self._installed = False
        self._active = False
        self._step = -1
        self._fault: Optional[FaultSpec] = None
        self._protection = ProtectionSpec(BoundsSource.NONE, Correction.NONE)
        self._offline_bounds: dict[str, Bounds] = {}
        self._online_bounds: dict[str, Bounds] = {}
        self._hook_calls: dict[str, int] = {}
        self._event_calls: dict[Tuple[int, str], int] = {}
        self._injection_count = 0
        self._trace: Optional[InjectionTrace] = None
        self._correction_counts: Optional[torch.Tensor] = None

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("FT2 hooks are already installed")
        for site in self.adapter.sites:
            handle = site.module.register_forward_hook(self._make_hook(site))
            self._handles.append(handle)
        self._installed = True

    def remove(self) -> None:
        if self._active:
            raise RuntimeError("Cannot remove hooks during an active inference")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._installed = False

    def __enter__(self) -> "FT2HookEngine":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._active:
            self.abort()
        self.remove()

    def start_inference(
        self,
        *,
        fault: Optional[FaultSpec],
        protection: ProtectionSpec,
        offline_bounds: Optional[Mapping[str, Bounds]] = None,
    ) -> None:
        if not self._installed:
            raise RuntimeError("Install hooks before starting an inference")
        if self._active:
            raise RuntimeError("A previous inference is still active")
        protection = ProtectionSpec(
            protection.bounds_source,
            protection.correction,
            protection.scaling_factor,
        )
        if fault is not None:
            if fault.model_key != self.adapter.model_key:
                raise ValueError("Fault model_key does not match the loaded model")
            try:
                site = self.adapter.sites_by_key[fault.site_key]
            except KeyError as exc:
                raise ValueError(f"Unknown fault site: {fault.site_key}") from exc
            if (
                site.layer_index != fault.layer_index
                or site.projection != fault.projection
                or site.out_features != fault.expected_out_features
            ):
                raise ValueError("Fault coordinates do not match the resolved site")

        supplied = dict(offline_bounds or {})
        if protection.bounds_source is BoundsSource.OFFLINE:
            missing = self._protected_keys.difference(supplied)
            extra = set(supplied).difference(self._protected_keys)
            if missing or extra:
                raise ValueError(
                    f"Offline bounds key mismatch: missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            supplied = {
                key: Bounds(value.raw_min, value.raw_max, protection.scaling_factor)
                for key, value in supplied.items()
            }
        elif supplied:
            raise ValueError("Offline bounds supplied to a non-offline mode")

        self._active = True
        self._step = -1
        self._fault = fault
        self._protection = protection
        self._offline_bounds = supplied
        self._online_bounds = {}
        self._hook_calls = {site.key: 0 for site in self.adapter.sites}
        self._event_calls = {}
        self._injection_count = 0
        self._trace = None
        self._correction_counts = None

    def begin_step(self, step: int) -> None:
        if not self._active:
            raise RuntimeError("No active inference")
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an int")
        if step != self._step + 1:
            raise RuntimeError(
                f"Generation steps must be sequential: previous={self._step}, next={step}"
            )
        self._step = step

    @property
    def current_step(self) -> int:
        return self._step

    def _make_hook(self, site: ProjectionSite):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any):
            if not self._active:
                raise RuntimeError("A target Linear ran outside an active FT2 inference")
            if self._step < 0:
                raise RuntimeError("Call begin_step before every model forward")
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"{site.key} returned non-Tensor output")
            if output.dtype != torch.float16:
                raise TypeError(f"{site.key} returned {output.dtype}, expected float16")
            sequence_length = _logical_sequence_length(output, site)

            event = (self._step, site.key)
            event_count = self._event_calls.get(event, 0) + 1
            self._event_calls[event] = event_count
            if event_count != 1:
                raise RuntimeError(f"{site.key} ran more than once at step {self._step}")
            self._hook_calls[site.key] += 1

            value = output
            target = (
                self._fault is not None
                and self._fault.target_step == self._step
                and self._fault.site_key == site.key
            )
            target_flat_index: Optional[int] = None
            before_bits = after_bits = 0
            before_value = after_value = 0.0

            if target:
                fault = self._fault
                assert fault is not None
                actual_shape = tuple(int(size) for size in value.shape)
                if sequence_length != fault.expected_sequence_length:
                    raise RuntimeError(
                        f"Fault activation shape mismatch: actual={actual_shape}, "
                        f"expected logical sequence={fault.expected_sequence_length}"
                    )
                target_flat_index = fault.flat_index
                before_bits = fp16_bits_at(value, target_flat_index)
                before_value = float(value.contiguous().reshape(-1)[target_flat_index].item())
                value = flip_fp16_bits(
                    value, target_flat_index, fault.bit_positions
                )
                after_bits = fp16_bits_at(value, target_flat_index)
                after_value = float(value.reshape(-1)[target_flat_index].item())
                self._injection_count += 1
                if self._injection_count != 1:
                    raise RuntimeError("More than one physical XOR event occurred")
                self._trace = InjectionTrace(
                    spec_id=fault.spec_id,
                    observed_step=self._step,
                    site_key=site.key,
                    output_shape=actual_shape,
                    flat_index=target_flat_index,
                    bit_positions=fault.bit_positions,
                    before_bits=before_bits,
                    after_bits=after_bits,
                    protected_bits=after_bits,
                    before_value=before_value,
                    after_value=after_value,
                    protected_value=after_value,
                    bounds=None,
                    correction_action="none",
                )

            target_bounds: Optional[Bounds] = None
            target_action = "none"
            if (
                site.key in self._protected_keys
                and self._protection.bounds_source is not BoundsSource.NONE
            ):
                if (
                    self._protection.bounds_source is BoundsSource.FIRST_TOKEN
                    and self._step == 0
                ):
                    if self._protection.correction is Correction.PAPER_CLAMP:
                        nan_mask = torch.isnan(value)
                        value = torch.where(nan_mask, torch.zeros_like(value), value)
                        self._add_correction_counts(
                            nan_mask=nan_mask,
                            low_mask=torch.zeros_like(nan_mask),
                            high_mask=torch.zeros_like(nan_mask),
                            invalid_mask=nan_mask,
                        )
                        if target and target_flat_index is not None and bool(
                            nan_mask.reshape(-1)[target_flat_index].item()
                        ):
                            target_action = "nan_to_zero"
                    raw_min = float(value.detach().amin().item())
                    raw_max = float(value.detach().amax().item())
                    target_bounds = Bounds(
                        raw_min,
                        raw_max,
                        self._protection.scaling_factor,
                    )
                    if site.key in self._online_bounds:
                        raise RuntimeError(f"Duplicate first-token bounds for {site.key}")
                    self._online_bounds[site.key] = target_bounds
                else:
                    if self._protection.bounds_source is BoundsSource.FIRST_TOKEN:
                        try:
                            target_bounds = self._online_bounds[site.key]
                        except KeyError as exc:
                            raise RuntimeError(
                                f"Missing first-token bounds for {site.key}"
                            ) from exc
                    else:
                        target_bounds = self._offline_bounds[site.key]
                    value, invalid, nan, low, high = self._apply_correction(
                        value,
                        target_bounds,
                        self._protection.correction,
                    )
                    self._add_correction_counts(
                        nan_mask=nan,
                        low_mask=low,
                        high_mask=high,
                        invalid_mask=invalid,
                    )
                    if target and target_flat_index is not None:
                        target_action = self._scalar_action(
                            after_value,
                            target_bounds,
                            self._protection.correction,
                        )

            if target:
                assert self._fault is not None and target_flat_index is not None
                protected_bits = fp16_bits_at(value, target_flat_index)
                protected_value = float(value.reshape(-1)[target_flat_index].item())
                self._trace = InjectionTrace(
                    spec_id=self._fault.spec_id,
                    observed_step=self._step,
                    site_key=site.key,
                    output_shape=tuple(int(size) for size in value.shape),
                    flat_index=target_flat_index,
                    bit_positions=self._fault.bit_positions,
                    before_bits=before_bits,
                    after_bits=after_bits,
                    protected_bits=protected_bits,
                    before_value=before_value,
                    after_value=after_value,
                    protected_value=protected_value,
                    bounds=target_bounds,
                    correction_action=target_action,
                )
            return value

        return hook

    def _apply_correction(
        self,
        value: torch.Tensor,
        bounds: Bounds,
        correction: Correction,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nan = torch.isnan(value)
        low = value < bounds.lower
        high = value > bounds.upper
        if correction is Correction.PAPER_CLAMP:
            invalid = nan | low | high
            clipped = torch.clamp(value, min=bounds.lower, max=bounds.upper)
            result = torch.where(nan, torch.zeros_like(value), clipped)
            return result, invalid, nan, low, high
        if correction is Correction.REPOSITORY_ZERO:
            invalid = (~torch.isfinite(value)) | low | high
            result = torch.where(invalid, torch.zeros_like(value), value)
            return result, invalid, nan, low, high
        raise AssertionError(f"Unhandled correction operator: {correction}")

    @staticmethod
    def _scalar_action(
        value: float,
        bounds: Bounds,
        correction: Correction,
    ) -> str:
        if math.isnan(value):
            return "nan_to_zero"
        if correction is Correction.REPOSITORY_ZERO and math.isinf(value):
            return "nonfinite_to_zero"
        if value < bounds.lower:
            return (
                "clamp_low"
                if correction is Correction.PAPER_CLAMP
                else "out_of_bounds_to_zero"
            )
        if value > bounds.upper:
            return (
                "clamp_high"
                if correction is Correction.PAPER_CLAMP
                else "out_of_bounds_to_zero"
            )
        return "none"

    def _add_correction_counts(
        self,
        *,
        nan_mask: torch.Tensor,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> None:
        counts = torch.stack(
            (
                invalid_mask.sum(),
                nan_mask.sum(),
                low_mask.sum(),
                high_mask.sum(),
            )
        ).to(dtype=torch.int64)
        if self._correction_counts is None:
            self._correction_counts = counts
        else:
            self._correction_counts.add_(counts)

    def _snapshot(self, *, complete: bool) -> EngineRecord:
        if self._correction_counts is None:
            counts = (0, 0, 0, 0)
        else:
            counts = tuple(int(value) for value in self._correction_counts.tolist())
        return EngineRecord(
            mode_id=self._protection.mode_id,
            final_step=self._step,
            injection_count=self._injection_count,
            injection_trace=self._trace,
            hook_calls=dict(self._hook_calls),
            online_bounds=dict(self._online_bounds),
            correction_elements=counts[0],
            nan_elements=counts[1],
            low_elements=counts[2],
            high_elements=counts[3],
            complete=complete,
        )

    def finish_inference(self) -> EngineRecord:
        if not self._active or self._step < 0:
            raise RuntimeError("No executed inference is active")
        expected_injections = 1 if self._fault is not None else 0
        if self._injection_count != expected_injections:
            raise AssertionError(
                f"Expected {expected_injections} XOR event(s), "
                f"observed {self._injection_count}"
            )
        if self._fault is not None:
            if self._trace is None or not self._trace.bit_flip_verified:
                raise AssertionError("Fault telemetry is missing or bit XOR is invalid")
            expected_hamming = len(self._fault.bit_positions)
            if self._trace.hamming_distance != expected_hamming:
                raise AssertionError("Observed FP16 Hamming distance is incorrect")
        if self._protection.bounds_source is BoundsSource.FIRST_TOKEN:
            missing = self._protected_keys.difference(self._online_bounds)
            extra = set(self._online_bounds).difference(self._protected_keys)
            if missing or extra:
                raise AssertionError(
                    f"First-token bounds mismatch: missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
        expected_calls = self._step + 1
        wrong_calls = {
            key: count
            for key, count in self._hook_calls.items()
            if count != expected_calls
        }
        if wrong_calls:
            raise AssertionError(
                f"Every target Linear must run once per step: {wrong_calls}"
            )
        record = self._snapshot(complete=True)
        self._active = False
        return record

    def abort(self) -> EngineRecord:
        if not self._active:
            raise RuntimeError("No active inference to abort")
        record = self._snapshot(complete=False)
        self._active = False
        return record


class OfflineBoundsProfiler:
    """Accumulate global clean prefill extrema without storing activations."""

    def __init__(self, adapter: ModelAdapter, scaling_factor: float = 2.0):
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")
        self.adapter = adapter
        self.scaling_factor = float(scaling_factor)
        self._critical_keys = frozenset(site.key for site in adapter.critical_sites)
        self._handles: list[RemovableHandle] = []
        self._active_example = False
        self._seen: set[str] = set()
        self._mins: dict[str, torch.Tensor] = {}
        self._maxs: dict[str, torch.Tensor] = {}
        self.example_count = 0

    def install(self) -> None:
        if self._handles:
            raise RuntimeError("Profiler hooks are already installed")
        for site in self.adapter.critical_sites:
            self._handles.append(
                site.module.register_forward_hook(self._make_hook(site))
            )

    def remove(self) -> None:
        if self._active_example:
            raise RuntimeError("Cannot remove profiler during an example")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def start_example(self) -> None:
        if not self._handles:
            raise RuntimeError("Install profiler hooks first")
        if self._active_example:
            raise RuntimeError("Previous profiling example is still active")
        self._active_example = True
        self._seen = set()

    def _make_hook(self, site: ProjectionSite):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any):
            if not self._active_example:
                raise RuntimeError("Profiler hook ran outside an active example")
            if site.key in self._seen:
                raise RuntimeError(f"Profiler saw {site.key} more than once")
            if not isinstance(output, torch.Tensor) or output.dtype != torch.float16:
                raise RuntimeError(
                    f"Profiler received invalid output at {site.key}: "
                    f"{type(output)!r}, {getattr(output, 'shape', None)}"
                )
            _logical_sequence_length(output, site)
            self._seen.add(site.key)
            current_min = output.detach().amin()
            current_max = output.detach().amax()
            if site.key in self._mins:
                self._mins[site.key] = torch.minimum(
                    self._mins[site.key], current_min
                )
                self._maxs[site.key] = torch.maximum(
                    self._maxs[site.key], current_max
                )
            else:
                self._mins[site.key] = current_min
                self._maxs[site.key] = current_max
            return output

        return hook

    def finish_example(self) -> None:
        if not self._active_example:
            raise RuntimeError("No profiling example is active")
        missing = self._critical_keys.difference(self._seen)
        extra = self._seen.difference(self._critical_keys)
        if missing or extra:
            raise AssertionError(
                f"Profiler key mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        self.example_count += 1
        self._active_example = False

    def finalize(self, expected_examples: int) -> dict[str, Bounds]:
        if self._active_example:
            raise RuntimeError("Finish the active profiling example first")
        if self.example_count != expected_examples:
            raise AssertionError(
                f"Expected {expected_examples} profiling examples, "
                f"observed {self.example_count}"
            )
        if set(self._mins) != self._critical_keys or set(self._maxs) != self._critical_keys:
            raise AssertionError("Offline profiler did not cover all critical keys")
        result = {}
        for key in sorted(self._critical_keys):
            result[key] = Bounds(
                float(self._mins[key].item()),
                float(self._maxs[key].item()),
                self.scaling_factor,
            )
        return result
