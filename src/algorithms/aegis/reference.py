"""Small CPU mathematical checks, not a hardware GEMM or training backend.

Only FP32/BF16 storage is modeled. FP8/NVFP4 research must use official CUTLASS
conversion, scales and GEMM; no lookup-table substitute is provided here.
"""
from dataclasses import dataclass

import torch


DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16}


@dataclass(frozen=True)
class Fault:
    target: str
    bit: int
    coord: int = 0
    zero_mask: bool = False


def flip(value: torch.Tensor, bit: int, coord: int, *, zero_mask: bool = False) -> torch.Tensor:
    """XOR raw storage, including sign bits, NaNs and signed zero."""
    if value.dtype not in DTYPES.values():
        raise ValueError("reference injection supports only FP32/BF16")
    width = value.element_size() * 8
    if not 0 <= bit < width or not 0 <= coord < value.numel():
        raise ValueError("bit or coordinate out of range")
    result = value.detach().clone().contiguous()
    storage = result.view(torch.int32 if width == 32 else torch.int16).view(-1)
    mask = 0 if zero_mask else (-(1 << bit) if bit == width - 1 else 1 << bit)
    storage[coord] ^= mask
    return result


def bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@torch.no_grad()
def gemm(a: torch.Tensor, b: torch.Tensor, *, output_dtype: torch.dtype,
         k_blocks: int = 1, fault: Fault | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 block product rounded to FP32, then FP32 block accumulation.

    This fixes an arithmetic order for mechanism checks; it does not model MMA
    rounding. Faults affect input storage or the final pre-cast accumulator.
    """
    if a.device.type != 'cpu' or b.device.type != 'cpu':
        raise ValueError("mathematical reference runs on CPU in the project container")
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
            or min(*a.shape, *b.shape) < 1 or type(k_blocks) is not int
            or k_blocks < 1 or a.shape[1] % k_blocks):
        raise ValueError("incompatible matrices or K blocks")
    if a.dtype not in DTYPES.values() or b.dtype != a.dtype or output_dtype not in DTYPES.values():
        raise ValueError("reference supports matching FP32/BF16 operands and FP32/BF16 output")
    if fault is not None:
        if fault.target not in ('A', 'B', 'accumulator'):
            raise ValueError("unknown fault target")
        if fault.target == 'A':
            a = flip(a, fault.bit, fault.coord, zero_mask=fault.zero_mask)
        if fault.target == 'B':
            b = flip(b, fault.bit, fault.coord, zero_mask=fault.zero_mask)
    acc = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    width = a.shape[1] // k_blocks
    for start in range(0, a.shape[1], width):
        acc += (a[:, start:start + width].double() @ b[start:start + width].double()).float()
    if fault is not None and fault.target == 'accumulator':
        acc = flip(acc, fault.bit, fault.coord, zero_mask=fault.zero_mask)
    return acc, acc.to(output_dtype)
