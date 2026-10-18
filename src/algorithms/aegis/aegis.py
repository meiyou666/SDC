"""AEGIS Matmul cSensor reference (§5.1.1, §5.1.3, §6).

Call prepare before fault injection, then sense with an actual pre-cast FP32
accumulator or output. This module neither executes nor replaces a training GEMM.
Reference checksum precision, infinity norm and threshold defaults are explicit
reproduction choices; see docs/reproduction/aegis.md.
"""
from dataclasses import dataclass
import math

import torch


def _matrix(value: torch.Tensor) -> None:
    if value.ndim != 2 or not value.is_floating_point() or 0 in value.shape:
        raise ValueError("expected a nonempty floating-point matrix")


@dataclass(frozen=True)
class Checksums:
    row: torch.Tensor
    column: torch.Tensor | None
    shape: tuple[int, int]


@torch.no_grad()
def prepare(a: torch.Tensor, b: torch.Tensor, *, columns: bool = False) -> Checksums:
    """Freeze FP32 predicted checksums from effective, already quantized inputs."""
    _matrix(a)
    _matrix(b)
    if a.shape[1] != b.shape[0] or a.device != b.device:
        raise ValueError("incompatible operands")
    a, b = a.float(), b.float()
    row = a @ b.sum(dim=1, dtype=torch.float32)
    column = a.sum(dim=0, dtype=torch.float32) @ b if columns else None
    return Checksums(row.clone(), None if column is None else column.clone(),
                     (a.shape[0], b.shape[1]))


@dataclass(frozen=True)
class Evidence:
    row: torch.Tensor
    column: torch.Tensor | None
    stat: float
    row_index: int
    column_index: int | None


@torch.no_grad()
def sense(checksums: Checksums, observed: torch.Tensor) -> Evidence:
    """FP32 reduction; FP64 subtraction isolates difference-rounding effects.

    For the AEGIS path observed must be the pre-cast FP32 accumulator.
    Passing a low-precision output is an explicit comparison experiment.
    Nonfinite checksums produce an alarm statistic of +inf.
    """
    _matrix(observed)
    if tuple(observed.shape) != checksums.shape or observed.device != checksums.row.device:
        raise ValueError("observed matrix does not match checksums")
    row = observed.float().sum(dim=1, dtype=torch.float32).double() - checksums.row.double()
    column = None
    if checksums.column is not None:
        column = (observed.float().sum(dim=0, dtype=torch.float32).double()
                  - checksums.column.double())

    def maximum(residual):
        magnitude = residual.abs()
        magnitude = torch.where(torch.isfinite(magnitude), magnitude, torch.inf)
        index = int(magnitude.argmax().item())
        return float(magnitude[index].item()), index

    stat, row_index = maximum(row)
    column_index = None
    if column is not None:
        column_stat, column_index = maximum(column)
        stat = max(stat, column_stat)
    return Evidence(row, column, stat, row_index, column_index)


def calibrate(stats: list[float], alpha: float) -> float:
    """Empirical inverse-CDF (1-alpha) quantile, using independent normal calls."""
    if not stats or not 0 <= alpha < 1:
        raise ValueError("need nonempty calibration and 0 <= alpha < 1")
    if any(not math.isfinite(x) or x < 0 for x in stats):
        raise ValueError("calibration statistics must be finite and nonnegative")
    return sorted(stats)[max(0, math.ceil((1 - alpha) * len(stats)) - 1)]


class Threshold:
    """Compare against the old threshold, then update the runtime estimate.

    A positive floor avoids the absorbing zero state of multiplicative updates.
    Nonfinite signals alarm without contaminating finite history.
    """

    def __init__(self, baseline: float, *, adaptive: bool = True,
                 window: int = 50, factor: float = 2.0):
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError("baseline must be finite and positive")
        if type(window) is not int or window < 1 or not math.isfinite(factor) or factor <= 1:
            raise ValueError("window must be positive and factor > 1")
        self.baseline, self.adaptive = baseline, adaptive
        self.window, self.factor = window, factor
        self.tau, self.quiet = baseline, 0

    def update(self, stat: float) -> bool:
        if not math.isfinite(stat):
            return True
        if stat < 0:
            raise ValueError("statistic must be nonnegative")
        alarm = stat > self.tau
        if not self.adaptive:
            return alarm
        if alarm:
            self.tau = min(self.tau * self.factor, float.fromhex('0x1.fffffffffffffp+1023'))
            self.quiet = 0
        else:
            self.quiet += 1
            if self.quiet >= self.window:
                self.tau = max(self.tau / self.factor, self.baseline)
                self.quiet = 0
        return alarm

    def state_dict(self) -> dict:
        return dict(baseline=self.baseline, adaptive=self.adaptive, window=self.window,
                    factor=self.factor, tau=self.tau, quiet=self.quiet)

    def load_state_dict(self, state: dict) -> None:
        expected = self.state_dict()
        if state.keys() != expected.keys() or any(
            state[key] != expected[key] for key in ('baseline', 'adaptive', 'window', 'factor')
        ):
            raise ValueError("threshold configuration mismatch")
        if (not math.isfinite(state['tau']) or state['tau'] < self.baseline
                or type(state['quiet']) is not int or not 0 <= state['quiet'] < self.window):
            raise ValueError("invalid threshold state")
        self.tau, self.quiet = state['tau'], state['quiet']
