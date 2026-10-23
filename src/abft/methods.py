"""ABFT method registry and explicit contracts for compiled CUDA policies."""
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from cutlass_gemm import backend


_PRECISIONS = frozenset({"fp32", "bf16", "fp8", "nvfp4"})


@dataclass(frozen=True)
class CompiledPolicy:
    algorithm: int
    channels: int

    @property
    def reference_formats(self):
        # These describe installed kernel formats, not a configurable claim
        # that low-precision inputs imply same-precision checksum arithmetic.
        return (("input_fused", "fp32_projection_fp32_accumulation"),
                ("augmented", "bf16_encoding_fp32_accumulation"))

    def __post_init__(self):
        if type(self.algorithm) is not int or type(self.channels) is not int or (
            self.algorithm, self.channels
        ) not in {(0, 2), (1, 1)}:
            raise ValueError("Compiled ABFT policy must match an installed CUDA algorithm/channels pair")

    def metadata(self):
        return {"id": self.algorithm, "channels": self.channels,
                "reference_formats": dict(self.reference_formats),
                "native_localization": False}


class AbftMethod(ABC):
    stateful: bool = False
    # Internal statistics required by commit, independent of report verbosity.
    needs_diagnostics: bool = False
    state_schema_version: int = 1
    compiled_policy = None
    supported_references = {"separate": _PRECISIONS}
    # Enabled only for the separate profiling run; no timing or I/O by default.
    _profile_scope = staticmethod(nullcontext)

    def validate_config(self, config):
        """Validate method-specific parameters before preparing evidence."""

    def validate_execution(self, operands, config):
        if type(self.needs_diagnostics) is not bool:
            raise TypeError("ABFT method needs_diagnostics must be a bool")
        self.validate_config(config)
        if config.reference_mode not in self.supported_references:
            raise NotImplementedError(f"{config.method} does not support {config.reference_mode}")
        if operands.precision not in self.supported_references[config.reference_mode]:
            raise ValueError(f"{config.method} does not support {config.reference_mode} with {operands.precision}")
        if config.implementation == "native":
            # A Python subclass overriding verify must explicitly opt into a
            # compiled policy; inherited Classic metadata must not bypass it.
            policy = type(self).__dict__.get("compiled_policy")
            if "compiled_policy" in vars(self) or not isinstance(policy, CompiledPolicy):
                raise ValueError(f"{config.method} has no explicit compiled CUDA policy for native implementation")
            if config.reference_mode not in dict(policy.reference_formats):
                raise ValueError(f"{config.method} native implementation does not support {config.reference_mode}")
            return policy
        return self.compiled_policy

    def prepare_fused(self, operands, config):
        """Prepare native projection metadata and optional frozen thresholds.

        Native methods may return thresholds as device FP32 [1] or [channels,N].
        Snapshot mutable history here; verification and retry share this tensor.
        """
        raise NotImplementedError("Method does not support input_fused reference")

    def prepare_augmented(self, operands, config):
        """Choose same-precision encoding weights for a native augmented GEMM."""
        raise NotImplementedError("Method does not support augmented GEMM")

    def bind_reference(self, predicted_partials, magnitude_partials, state):
        """Return state bound to the first execution's native reference."""
        raise NotImplementedError("Method does not support input_fused reference")

    @abstractmethod
    def prepare(self, operands, config, mm=None):
        """Build clean-input evidence; keep this call state read-only until commit."""

    @abstractmethod
    def propagate(self, partials, state):
        """Reduce native pre-cast partial projections."""

    @abstractmethod
    def verify(self, observed, output, state, config):
        """Return device tensors describing anomalies and candidate locations."""

    def recover(self, operands, state):
        # Retry the clean, already prepared operands, without quantizing again.
        policy = self.compiled_policy
        if policy is None or policy.algorithm == 0:
            return operands.collected_native(state["weights"])
        return operands.collected_native(state["weights"], algorithm=policy.algorithm)

    def commit(self, state, decision, context):
        """Update persistent history once, after verification and any retry.

        decision/context contain only observed decisions and execution metadata,
        never the configured fault, injection hit, or fault-selection truth.
        """

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("This method has no persistent state")


_METHODS = {}


def register_method(name, factory):
    if not name or name == "none" or name in _METHODS:
        raise ValueError(f"Invalid or duplicate ABFT method: {name}")
    _METHODS[name] = factory


def get_method(name):
    """Create a method; callers may retain it through method_instance."""
    if name not in _METHODS:
        raise ValueError(f"ABFT method is not implemented: {name}")
    method = _METHODS[name]()
    if not isinstance(method, AbftMethod):
        raise TypeError("ABFT factories must return an AbftMethod")
    method._registered_name = name
    return method


def validate_method_instance(name, method):
    if not isinstance(method, AbftMethod):
        raise TypeError("method_instance must be an AbftMethod")
    registered = getattr(method, "_registered_name", None)
    if registered is not None:
        if registered != name:
            raise ValueError("method_instance does not match the configured ABFT method")
    elif _METHODS.get(name) is not type(method):
        raise ValueError("Create this method_instance with get_method for the configured method")
    return method


class Classic(AbftMethod):
    compiled_policy = CompiledPolicy(0, 2)
    supported_references = {"separate": _PRECISIONS, "input_fused": _PRECISIONS,
                            "augmented": frozenset({"bf16"})}
    channels = 2

    def validate_config(self, config):
        if config.parameters:
            raise ValueError("classic does not accept method parameters")

    def projection_weights(self, operands):
        return torch.stack((torch.ones(operands.m, device=operands.a.device, dtype=torch.float32),
                               torch.arange(1, operands.m+1, device=operands.a.device, dtype=torch.float32)))

    def prepare_fused(self, operands, config):
        weights = self.projection_weights(operands)
        return {"weights": weights, "m": operands.m, "n": operands.n}

    def prepare_augmented(self, operands, config):
        if operands.precision != "bf16":
            raise ValueError("augmented currently requires BF16 GEMM")
        state = self.prepare_fused(operands, config)
        # Encoding weights, checksum rows and A use the same BF16 input format.
        return {**state, "weights": state["weights"].to(torch.bfloat16), "augmented": True}

    def bind_reference(self, predicted_partials, magnitude_partials, state):
        return {**state, "expected": self.propagate(predicted_partials, state),
                "magnitude": self.propagate(magnitude_partials, state)}

    def prepare(self, operands, config, mm=None):
        # CUDA uses the existing FP32 CUTLASS GEMM, never Torch's TF32 matmul.
        mm = backend.mm if mm is None else mm
        weights = self.projection_weights(operands)
        expected = torch.zeros((self.channels, operands.n), device=operands.a.device, dtype=torch.float32)
        magnitude = torch.zeros_like(expected)
        for start in range(0, operands.k, config.reference_block_k):
            end = min(operands.k, start + config.reference_block_k)
            a, b = operands.decode_block(start, end)
            # A K slice can retain the full operand's leading stride. Native
            # reference GEMMs require packed matrices, unlike torch.mm.
            a, b = a.contiguous(), b.contiguous()
            expected += mm(mm(weights, a).contiguous(), b.contiguous())
            magnitude += mm(mm(weights.abs(), a.abs()).contiguous(), b.abs().contiguous())
        return {"weights": weights, "expected": expected, "magnitude": magnitude,
                "m": operands.m, "n": operands.n}

    def propagate(self, partials, state):
        return partials.sum(dim=1)[:, :state["n"]]

    def verify(self, observed, output, state, config):
        with self._profile_scope("abft.detect"):
            residual = observed - state["expected"]
            threshold = config.atol + config.rtol * state["magnitude"]
            nonfinite = (~torch.isfinite(observed)).any() | (~torch.isfinite(state["expected"])).any()
            nonfinite |= (~torch.isfinite(threshold)).any() | (~torch.isfinite(output)).any()
            nonfinite |= (~torch.isfinite(residual)).any()
            failed = residual.abs() > threshold
            detected = failed.any() | nonfinite
        with self._profile_scope("abft.locate"):
            if state.get("augmented", False):
                # BF16 row-number weights can collide and encoding adds rounding.
                # The FP32 ratio-based row locator is not valid for this path.
                return {"detected": detected, "nonfinite": nonfinite,
                        "located": torch.zeros((), device=output.device, dtype=torch.bool),
                        "location": torch.full((2,), -1, device=output.device, dtype=torch.int64),
                        "residual": residual, "threshold": threshold}
            columns = failed.any(dim=0)
            col = columns.to(torch.int64).argmax()
            d0, d1 = residual[:, col].unbind()
            # A candidate location under the single-output-error assumption, not a
            # certificate excluding structured/multiple errors in the nullspace.
            ratio = d1 / torch.where(d0 == 0, torch.ones_like(d0), d0)
            row = ratio.round().to(torch.int64) - 1
            consistency = (d1 - (row+1).float()*d0).abs()
            allowance = threshold[1, col] + (row+1).abs().float()*threshold[0, col]
            located = detected & ~nonfinite & (columns.sum() == 1) & (d0.abs() > threshold[0, col])
            located &= (row >= 0) & (row < state["m"]) & (consistency <= allowance)
            # Reject an ambiguous row: both projection errors have independent
            # bounds. Their ratio interval must lie inside one rounding cell.
            uncertainty = (threshold[1, col] + ratio.abs() * threshold[0, col]) / (
                d0.abs() - threshold[0, col]
            ).clamp_min(torch.finfo(torch.float32).tiny)
            located &= (ratio - (row + 1)).abs() + uncertainty < 0.5
            location = torch.stack((torch.where(located, row, -1), torch.where(located, col, -1)))
        return {"detected": detected, "nonfinite": nonfinite, "located": located,
                "location": location, "residual": residual, "threshold": threshold}


register_method("classic", Classic)


class Sum(Classic):
    """One full all-ones column projection; detection only, no localization."""
    compiled_policy = CompiledPolicy(1, 1)
    channels = 1

    def validate_config(self, config):
        if config.parameters:
            raise ValueError("sum does not accept method parameters")

    def projection_weights(self, operands):
        return torch.ones((1, operands.m), device=operands.a.device, dtype=torch.float32)

    def verify(self, observed, output, state, config):
        with self._profile_scope("abft.detect"):
            residual = observed - state["expected"]
            threshold = config.atol + config.rtol * state["magnitude"]
            nonfinite = (~torch.isfinite(observed)).any() | (~torch.isfinite(state["expected"])).any()
            nonfinite |= (~torch.isfinite(threshold)).any() | (~torch.isfinite(output)).any()
            nonfinite |= (~torch.isfinite(residual)).any()
            detected = (residual.abs() > threshold).any() | nonfinite
        with self._profile_scope("abft.locate"):
            located = torch.zeros((), device=output.device, dtype=torch.bool)
            location = torch.full((2,), -1, device=output.device, dtype=torch.int64)
        return {"detected": detected, "nonfinite": nonfinite, "located": located,
                "location": location, "residual": residual, "threshold": threshold}


register_method("sum", Sum)
