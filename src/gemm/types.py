"""Backend-independent precision, event and result definitions."""

from dataclasses import dataclass
import hashlib
from typing import Literal

import torch


Phase = Literal["forward", "dgrad", "wgrad"]


@dataclass(frozen=True)
class Precision:
    name: str
    code: int
    bits: int
    scale_group: int
    scale_format: str | None


PROFILES = {
    "fp32": Precision("fp32", 0, 32, 0, None),
    "bf16": Precision("bf16", 1, 16, 0, None),
    "mxfp8": Precision("mxfp8", 2, 8, 32, "E8M0"),
    "nvfp4": Precision("nvfp4", 3, 4, 16, "E4M3+FP32"),
}
PHASES = ("forward", "dgrad", "wgrad")


@dataclass(frozen=True)
class ExecutionContext:
    seed: int = 2026
    module: str = "standalone"
    call: int = 0

    def quant_seed(self, phase: str) -> int:
        return self.stream_seed("quant", phase)

    def stream_seed(self, stream: str, phase: str) -> int:
        text = f"{stream}:v1:{self.seed}:{self.module}:{self.call}:{phase}"
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


@dataclass(frozen=True)
class Fault:
    site: Literal["a", "b", "accumulator"]
    row: int
    col: int
    bit: int
    enabled: bool = True

    def validate(self, m: int, n: int, k: int, profile: Precision) -> None:
        shapes = {"a": (m, k), "b": (k, n), "accumulator": (m, n)}
        if self.site not in shapes:
            raise ValueError("故障位置必须是 a、b 或 accumulator。")
        rows, cols = shapes[self.site]
        width = 32 if self.site == "accumulator" else profile.bits
        if any(type(v) is not int for v in (self.row, self.col, self.bit)) or type(self.enabled) is not bool:
            raise ValueError("故障坐标和 bit 必须是整数。")
        if not (0 <= self.row < rows and 0 <= self.col < cols and 0 <= self.bit < width):
            raise ValueError("故障坐标或 bit 超出范围。")


@dataclass
class PackedOperand:
    codes: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    rows: int
    columns: int
    padded_rows: int
    padded_columns: int
    profile: Precision


@dataclass
class GemmResult:
    output: torch.Tensor
    # [detected, located, row, col, correction_attempted, postcheck_failed]
    status: torch.Tensor
    residual: torch.Tensor
    threshold: torch.Tensor
    fault_bits: torch.Tensor
    profile: str
    algorithm: str
    backend: str
    phase: str
    recomputed: bool = False
    fault_values: torch.Tensor | None = None

    @property
    def detected(self) -> torch.Tensor:
        return self.status[0].bool()

    @property
    def unrecovered(self) -> torch.Tensor:
        return self.status[5].bool()
