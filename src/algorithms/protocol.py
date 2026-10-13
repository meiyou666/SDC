"""Interface for adding paper mechanisms without changing the GEMM core."""

from typing import Protocol

import torch

from gemm.types import PackedOperand


class Detector(Protocol):
    def prepare(self, a: PackedOperand, b: PackedOperand, *, phase: str, seed: int):
        """Generate reference checks before operand injection."""
        ...

    def verify(self, output: torch.Tensor, prepared, *, correct: bool):
        """Return status[6], residual and threshold on the output's device."""
        ...

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Online state and projection state saved with a training checkpoint."""
        ...

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        ...
