"""Method registry. Classic is a two-projection, single-output-error example."""
from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch

from cutlass_gemm import backend


class AbftMethod(ABC):
    # Enabled only for the separate profiling run; no timing or I/O by default.
    _profile_scope = staticmethod(nullcontext)

    def validate_config(self, config):
        """Validate method-specific parameters before preparing evidence."""

    def prepare_fused(self, operands, config):
        """Prepare projection metadata for native input-side reference generation."""
        raise NotImplementedError("Method does not support input_fused reference")

    def bind_reference(self, predicted_partials, magnitude_partials, state):
        """Return state bound to the first execution's native reference."""
        raise NotImplementedError("Method does not support input_fused reference")

    @abstractmethod
    def prepare(self, operands, config, mm=None):
        """Build clean-input evidence before any injected mutation."""

    @abstractmethod
    def propagate(self, partials, state):
        """Reduce native pre-cast partial projections."""

    @abstractmethod
    def verify(self, observed, output, state, config):
        """Return device tensors describing anomalies and candidate locations."""

    def recover(self, operands, state):
        # Retry the clean, already prepared operands, without quantizing again.
        return operands.collected_native(state["weights"])

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
    """Create a per-call method; persistent adaptive history is not yet wired.

    Stateful plugins must manage their lifecycle explicitly through their
    registered factory. execute does not checkpoint plugin state.
    """
    if name not in _METHODS:
        raise ValueError(f"ABFT method is not implemented: {name}")
    return _METHODS[name]()


class Classic(AbftMethod):
    def validate_config(self, config):
        if config.parameters:
            raise ValueError("classic does not accept method parameters")

    def prepare_fused(self, operands, config):
        weights = torch.stack((torch.ones(operands.m, device=operands.a.device),
                               torch.arange(1, operands.m+1, device=operands.a.device, dtype=torch.float32)))
        return {"weights": weights, "m": operands.m, "n": operands.n}

    def bind_reference(self, predicted_partials, magnitude_partials, state):
        return {**state, "expected": self.propagate(predicted_partials, state),
                "magnitude": self.propagate(magnitude_partials, state)}

    def prepare(self, operands, config, mm=None):
        # CUDA uses the existing FP32 CUTLASS GEMM, never Torch's TF32 matmul.
        mm = backend.mm if mm is None else mm
        weights = torch.stack((torch.ones(operands.m, device=operands.a.device),
                               torch.arange(1, operands.m+1, device=operands.a.device, dtype=torch.float32)))
        expected = torch.zeros((2, operands.n), device=operands.a.device, dtype=torch.float32)
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
