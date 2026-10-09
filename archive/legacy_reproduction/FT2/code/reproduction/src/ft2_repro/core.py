from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


CANDIDATE_PROJECTIONS: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

CRITICAL_PROJECTIONS = frozenset(
    ("v_proj", "o_proj", "up_proj", "down_proj")
)

ProjectionKey = Tuple[int, str]
FaultEventKey = Tuple[int, int, str]


class ProtectionMode(str, Enum):
    UNPROTECTED = "unprotected"
    REPOSITORY_ZERO = "repository_zero"
    PAPER_CLAMP = "paper_clamp"


@dataclass(frozen=True)
class Bounds:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("Protection bounds must be finite.")
        if self.lower > self.upper:
            raise ValueError(
                f"Invalid bounds: lower={self.lower} > upper={self.upper}"
            )


@dataclass(frozen=True)
class FaultSpec:
    sample_index: int
    trial_index: int
    token_step: int
    layer_index: int
    projection: str
    flat_index: int
    bit_index: int

    def __post_init__(self) -> None:
        integer_fields = {
            "sample_index": self.sample_index,
            "trial_index": self.trial_index,
            "token_step": self.token_step,
            "layer_index": self.layer_index,
            "flat_index": self.flat_index,
            "bit_index": self.bit_index,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value)!r}")

        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.trial_index < 0:
            raise ValueError("trial_index must be non-negative")
        if self.token_step < 1:
            raise ValueError(
                "token_step must be >= 1 because step 0 is reserved "
                "for clean first-token calibration"
            )
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if self.projection not in CANDIDATE_PROJECTIONS:
            raise ValueError(f"Unknown projection: {self.projection!r}")
        if self.flat_index < 0:
            raise ValueError("flat_index must be non-negative")
        if not 0 <= self.bit_index <= 15:
            raise ValueError("bit_index must be in [0, 15]")

    @property
    def stable_hash(self) -> str:
        payload = [
            "ft2-fault-v1",
            self.sample_index,
            self.trial_index,
            self.token_step,
            self.layer_index,
            self.projection,
            self.flat_index,
            self.bit_index,
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def fp16_bits_at(value: torch.Tensor, flat_index: int) -> int:
    """Return one binary16 element as an unsigned 16-bit integer."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError("Only strided tensors are supported")
    if value.dtype != torch.float16:
        raise TypeError(f"Expected torch.float16, got {value.dtype}")
    if isinstance(flat_index, bool) or not isinstance(flat_index, int):
        raise TypeError("flat_index must be an int")
    if not 0 <= flat_index < value.numel():
        raise IndexError(
            f"flat_index={flat_index} is outside tensor with "
            f"{value.numel()} elements"
        )

    signed = int(
        value.detach()
        .contiguous()
        .view(torch.int16)
        .reshape(-1)[flat_index]
        .item()
    )
    return signed & 0xFFFF


def fp16_bit_class(bits: int) -> str:
    """Classify an IEEE-754 binary16 bit pattern without float conversion."""
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError("bits must be an int")
    if not 0 <= bits <= 0xFFFF:
        raise ValueError("bits must be in [0, 65535]")

    sign = "negative" if bits & 0x8000 else "positive"
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x03FF
    if exponent == 0x1F:
        return "nan" if fraction else f"{sign}_infinity"
    if exponent == 0:
        return f"{sign}_zero" if fraction == 0 else f"{sign}_subnormal"
    return f"{sign}_normal"


@dataclass(frozen=True)
class InjectionRecord:
    """Auditable evidence for one planned binary16 fault injection."""

    spec_id: str
    observed_step: int
    layer_index: int
    projection: str
    flat_index: int
    bit_index: int
    output_shape: Tuple[int, ...]
    before_bits: int
    after_bits: int
    protected_bits: int

    @property
    def expected_after_bits(self) -> int:
        return self.before_bits ^ (1 << self.bit_index)

    @property
    def bit_flip_verified(self) -> bool:
        return self.after_bits == self.expected_after_bits

    @property
    def protection_changed_value(self) -> bool:
        return self.protected_bits != self.after_bits

    def __post_init__(self) -> None:
        if len(self.spec_id) != 64:
            raise ValueError("spec_id must be a SHA-256 hex digest")
        if not 0 <= self.bit_index <= 15:
            raise ValueError("bit_index must be in [0, 15]")
        for name in ("before_bits", "after_bits", "protected_bits"):
            bits = getattr(self, name)
            if not 0 <= bits <= 0xFFFF:
                raise ValueError(f"{name} must be in [0, 65535]")
        if not self.output_shape or any(size <= 0 for size in self.output_shape):
            raise ValueError("output_shape must contain positive dimensions")

    def to_dict(self) -> dict:
        def as_hex(bits: int) -> str:
            return f"0x{bits:04x}"

        return {
            "spec_id": self.spec_id,
            "observed_step": self.observed_step,
            "layer_index": self.layer_index,
            "projection": self.projection,
            "flat_index": self.flat_index,
            "bit_index": self.bit_index,
            "output_shape": list(self.output_shape),
            "before_bits_hex": as_hex(self.before_bits),
            "after_bits_hex": as_hex(self.after_bits),
            "expected_after_bits_hex": as_hex(self.expected_after_bits),
            "protected_bits_hex": as_hex(self.protected_bits),
            "before_class": fp16_bit_class(self.before_bits),
            "after_class": fp16_bit_class(self.after_bits),
            "protected_class": fp16_bit_class(self.protected_bits),
            "bit_flip_verified": self.bit_flip_verified,
            "protection_changed_value": self.protection_changed_value,
        }


def flip_fp16_bit(
    value: torch.Tensor,
    flat_index: int,
    bit_index: int,
) -> torch.Tensor:
    """Return a copy with exactly one IEEE-754 binary16 bit flipped.

    Bit 0 is the least-significant mantissa bit; bit 15 is the sign bit.
    The flattening order is the logical row-major order of a contiguous copy.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError("Only strided tensors are supported")
    if value.dtype != torch.float16:
        raise TypeError(
            f"Fault injection requires torch.float16, got {value.dtype}"
        )
    if isinstance(flat_index, bool) or not isinstance(flat_index, int):
        raise TypeError("flat_index must be an int")
    if isinstance(bit_index, bool) or not isinstance(bit_index, int):
        raise TypeError("bit_index must be an int")
    if not 0 <= flat_index < value.numel():
        raise IndexError(
            f"flat_index={flat_index} is outside tensor with "
            f"{value.numel()} elements"
        )
    if not 0 <= bit_index <= 15:
        raise ValueError("bit_index must be in [0, 15]")
    if torch.is_grad_enabled() and value.requires_grad:
        raise RuntimeError(
            "FT2 fault injection is inference-only; use torch.inference_mode()"
        )

    result = value.detach().contiguous().clone()
    bits = result.view(torch.int16).reshape(-1)

    unsigned_mask = 1 << bit_index
    signed_mask = (
        unsigned_mask
        if unsigned_mask < (1 << 15)
        else unsigned_mask - (1 << 16)
    )
    mask = torch.tensor(
        signed_mask,
        dtype=torch.int16,
        device=result.device,
    )
    bits[flat_index] = torch.bitwise_xor(bits[flat_index], mask)
    return result


def _effective_bounds(
    value: torch.Tensor,
    bounds: Bounds,
) -> Tuple[float, float]:
    if not value.is_floating_point():
        raise TypeError("Protection requires a floating-point tensor")

    finfo = torch.finfo(value.dtype)
    lower = max(bounds.lower, float(finfo.min))
    upper = min(bounds.upper, float(finfo.max))
    if lower > upper:
        raise RuntimeError(
            f"Bounds are not representable in {value.dtype}: "
            f"{bounds.lower}, {bounds.upper}"
        )
    return lower, upper


def apply_protection(
    value: torch.Tensor,
    mode: ProtectionMode,
    bounds: Optional[Bounds],
) -> torch.Tensor:
    mode = ProtectionMode(mode)
    if mode is ProtectionMode.UNPROTECTED:
        return value
    if bounds is None:
        raise ValueError("Protected modes require calibrated bounds")

    lower, upper = _effective_bounds(value, bounds)

    if mode is ProtectionMode.REPOSITORY_ZERO:
        invalid = torch.isnan(value) | (value < lower) | (value > upper)
        return torch.where(invalid, torch.zeros_like(value), value)

    if mode is ProtectionMode.PAPER_CLAMP:
        clipped = torch.clamp(value, min=lower, max=upper)
        return torch.nan_to_num(clipped, nan=0.0)

    raise AssertionError(f"Unhandled protection mode: {mode}")


@dataclass
class RunState:
    """One-shot, per-run state. Never reuse it for another generation."""

    mode: ProtectionMode
    faults: Sequence[FaultSpec]

    _current_step: int = field(init=False, default=-1)
    _bound: bool = field(init=False, default=False)
    _calibration_frozen: bool = field(init=False, default=False)
    _expected_critical_keys: set[ProjectionKey] = field(
        init=False, default_factory=set
    )
    _bounds: Dict[ProjectionKey, Bounds] = field(
        init=False, default_factory=dict
    )
    _fault_by_event: Dict[FaultEventKey, FaultSpec] = field(
        init=False, default_factory=dict
    )
    _injection_counts: Dict[str, int] = field(
        init=False, default_factory=dict
    )
    _injection_records: Dict[str, InjectionRecord] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = ProtectionMode(self.mode)
        self.faults = tuple(self.faults)

        stable_hashes: set[str] = set()
        for fault in self.faults:
            if not isinstance(fault, FaultSpec):
                raise TypeError("Every fault must be a FaultSpec")

            stable_hash = fault.stable_hash
            if stable_hash in stable_hashes:
                raise ValueError(f"Duplicate FaultSpec: {stable_hash}")
            stable_hashes.add(stable_hash)

            event = (
                fault.token_step,
                fault.layer_index,
                fault.projection,
            )
            if event in self._fault_by_event:
                raise ValueError(
                    "At most one fault may target a projection during "
                    f"one forward step; duplicate event={event}"
                )
            self._fault_by_event[event] = fault
            self._injection_counts[stable_hash] = 0

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def calibration_frozen(self) -> bool:
        return self._calibration_frozen

    @property
    def injection_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._injection_counts))

    @property
    def injection_records(self) -> Mapping[str, InjectionRecord]:
        return MappingProxyType(dict(self._injection_records))

    @property
    def bounds(self) -> Mapping[ProjectionKey, Bounds]:
        return MappingProxyType(dict(self._bounds))

    def bind(
        self,
        expected_critical_keys: Iterable[ProjectionKey],
    ) -> None:
        if self._bound:
            raise RuntimeError("RunState has already been bound to a model")
        if self._current_step != -1:
            raise RuntimeError("A started RunState cannot be bound")
        self._expected_critical_keys = set(expected_critical_keys)
        if not self._expected_critical_keys:
            raise ValueError("No critical projections were discovered")
        self._bound = True

    def begin_forward(self) -> None:
        if not self._bound:
            raise RuntimeError("RunState must be bound before model execution")

        if self._current_step == -1:
            self._current_step = 0
            return

        if self._current_step == 0:
            self.freeze_calibration()
        self._current_step += 1

    def calibrate(
        self,
        key: ProjectionKey,
        value: torch.Tensor,
    ) -> None:
        if self._current_step != 0:
            raise RuntimeError("Calibration is allowed only during step 0")
        if self._calibration_frozen:
            raise RuntimeError("Calibration bounds are already frozen")
        if key not in self._expected_critical_keys:
            raise KeyError(f"Unexpected critical projection: {key}")
        if key in self._bounds:
            raise RuntimeError(
                f"Critical projection was called more than once at step 0: {key}"
            )
        if not value.is_floating_point():
            raise TypeError("Calibration output must be floating point")
        if value.numel() == 0:
            raise ValueError(f"Cannot calibrate an empty tensor: {key}")

        detached = value.detach()
        raw_min = float(detached.amin().item())
        raw_max = float(detached.amax().item())
        if not math.isfinite(raw_min) or not math.isfinite(raw_max):
            raise RuntimeError(
                f"Non-finite clean calibration activation at {key}: "
                f"min={raw_min}, max={raw_max}"
            )

        self._bounds[key] = Bounds(
            lower=raw_min * 2.0,
            upper=raw_max * 2.0,
        )

    def freeze_calibration(self) -> None:
        if self._calibration_frozen:
            return
        missing = self._expected_critical_keys.difference(self._bounds)
        extra = set(self._bounds).difference(self._expected_critical_keys)
        if missing or extra:
            raise RuntimeError(
                "Incomplete calibration: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        self._calibration_frozen = True

    def bounds_for(self, key: ProjectionKey) -> Bounds:
        if not self._calibration_frozen:
            raise RuntimeError("Protection requested before bounds were frozen")
        try:
            return self._bounds[key]
        except KeyError as exc:
            raise KeyError(f"No calibrated bounds for {key}") from exc

    def fault_for_current_event(
        self,
        layer_index: int,
        projection: str,
    ) -> Optional[FaultSpec]:
        if self._current_step < 0:
            raise RuntimeError("No model forward is active")
        return self._fault_by_event.get(
            (self._current_step, layer_index, projection)
        )

    def record_injection(self, fault: FaultSpec, record: InjectionRecord) -> None:
        stable_hash = fault.stable_hash
        try:
            count = self._injection_counts[stable_hash]
        except KeyError as exc:
            raise KeyError("Injected an unregistered FaultSpec") from exc
        if count != 0:
            raise RuntimeError(
                f"Fault {stable_hash} was injected more than once"
            )

        planned = (
            stable_hash,
            fault.token_step,
            fault.layer_index,
            fault.projection,
            fault.flat_index,
            fault.bit_index,
        )
        observed = (
            record.spec_id,
            record.observed_step,
            record.layer_index,
            record.projection,
            record.flat_index,
            record.bit_index,
        )
        if observed != planned:
            raise RuntimeError(
                "Injection telemetry does not match plan: "
                f"planned={planned}, observed={observed}"
            )
        if not record.bit_flip_verified:
            raise RuntimeError(
                "Fault injection did not produce the planned one-bit XOR"
            )
        self._injection_counts[stable_hash] = 1
        self._injection_records[stable_hash] = record

    def assert_complete(self) -> None:
        if self._current_step < 0:
            raise AssertionError("The model was never executed")
        if not self._calibration_frozen:
            self.freeze_calibration()

        bad = {
            stable_hash: count
            for stable_hash, count in self._injection_counts.items()
            if count != 1
        }
        if bad:
            raise AssertionError(
                "Every planned fault must be injected exactly once; "
                f"invalid counts={bad}"
            )

        expected_records = {
            stable_hash
            for stable_hash, count in self._injection_counts.items()
            if count == 1
        }
        if set(self._injection_records) != expected_records:
            raise AssertionError(
                "Injection telemetry is incomplete: "
                f"expected={sorted(expected_records)}, "
                f"observed={sorted(self._injection_records)}"
            )
        if any(not record.bit_flip_verified for record in self._injection_records.values()):
            raise AssertionError("At least one recorded bit flip failed verification")
