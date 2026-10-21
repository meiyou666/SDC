"""Prepare once, decode bounded K blocks, and reuse clean native operands."""
from dataclasses import dataclass, replace
from typing import Any

import torch

from cutlass_gemm import backend


INPUT_REFERENCE_PRECISIONS = frozenset({"fp32", "bf16", "fp8", "nvfp4"})


@dataclass
class PreparedGemm:
    a: torch.Tensor
    b: torch.Tensor
    precision: str
    m: int
    n: int
    k: int
    qa: Any = None
    qb: Any = None
    alpha: Any = None
    sfa: Any = None
    sfb: Any = None

    @classmethod
    def prepare(cls, a, b, precision, scaling, global_scale="amax"):
        expected = {"fp32": "none", "bf16": "none", "fp8": "tensorwise", "nvfp4": "block16"}
        if precision not in expected or scaling != expected[precision]:
            raise ValueError("Incompatible precision/scaling")
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.device != b.device:
            raise ValueError("GEMM requires compatible matrices on one device")
        m, k = a.shape
        n = b.shape[1]
        if min(m, n, k) <= 0:
            raise ValueError("Empty GEMM is not supported")
        if precision in {"fp32", "bf16"}:
            dtype = torch.float32 if precision == "fp32" else torch.bfloat16
            # Preserve the original native GEMM's packed row/column layouts.
            return cls(a.to(dtype), b.to(dtype), precision, m, n, k)
        from cutlass_gemm.quantization import quantize
        backend.require_device(a.device)
        qa = quantize(a.contiguous(), precision, global_scale)
        qb = quantize(b.T.contiguous(), precision, global_scale)
        result = cls(qa.codes, qb.codes, precision, m, n, k, qa, qb,
                     (qa.multiplier * qb.multiplier).reciprocal())
        if precision == "nvfp4":
            ia, ib, sa, sb = backend._scale_layout(m, n, k, a.device)
            result.sfa = backend._pack_scales(qa.scales, ia, sa)
            result.sfb = backend._pack_scales(qb.scales, ib, sb)
        return result

    def raw_native(self):
        if self.precision in {"fp32", "bf16"}:
            return backend.mm(self.a, self.b)
        if self.precision == "fp8":
            return torch.ops.sdc_cutlass.fp8_mm(self.a.view(torch.float8_e4m3fn), self.b.view(torch.float8_e4m3fn), self.alpha)
        return torch.ops.sdc_cutlass.nvfp4_mm(self.a, self.b, self.sfa, self.sfb, self.k, self.alpha)

    def collected_native(self, weights, fault=(-1, -1, -1)):
        if self.precision in {"fp32", "bf16"}:
            backend.require_device(self.a.device)
            return torch.ops.sdc_cutlass.abft_mm(self.a, self.b, weights, *fault)
        if self.precision == "fp8":
            return torch.ops.sdc_cutlass.abft_fp8_mm(self.a.view(torch.float8_e4m3fn), self.b.view(torch.float8_e4m3fn), self.alpha, weights, *fault)
        return torch.ops.sdc_cutlass.abft_nvfp4_mm(self.a, self.b, self.sfa, self.sfb, self.k, self.alpha, weights, *fault)

    def collected_reference_native(self, weights, fault=(-1, -1, -1)):
        """Return output, observed, hit, predicted and magnitude partials."""
        if self.precision in {"fp32", "bf16"}:
            backend.require_device(self.a.device)
            return torch.ops.sdc_cutlass.abft_reference_mm(self.a, self.b, weights, *fault)
        if self.precision not in INPUT_REFERENCE_PRECISIONS:
            raise NotImplementedError(f"input_fused reference is not implemented for {self.precision}")
        if self.precision == "nvfp4":
            return torch.ops.sdc_cutlass.abft_nvfp4_reference_mm(
                self.a, self.b, self.sfa, self.sfb, self.k, self.alpha, weights, *fault,
            )
        return torch.ops.sdc_cutlass.abft_fp8_reference_mm(
            self.a.view(torch.float8_e4m3fn), self.b.view(torch.float8_e4m3fn),
            self.alpha, weights, *fault,
        )

    def augmented_native(self, weights, fault=(-1, -1, -1)):
        """BF16 tile encoding and augmented main MMA, without global A re-reads."""
        if self.precision != "bf16":
            raise ValueError("augmented currently requires BF16 GEMM")
        backend.require_device(self.a.device)
        return torch.ops.sdc_cutlass.abft_augmented_mm(self.a, self.b, weights, *fault)

    def decode_block(self, start, end):
        if not 0 <= start < end <= self.k:
            raise ValueError("Invalid K block")
        if self.precision == "nvfp4" and (start % 16 or end % 16):
            raise ValueError("NVFP4 reference blocks must align to block16 scales")
        if self.precision in {"fp32", "bf16"}:
            return self.a[:, start:end].float(), self.b[start:end, :].float()
        def decode(codes, q):
            if self.precision == "fp8":
                return codes[:, start:end].view(torch.float8_e4m3fn).float() / q.multiplier
            packed = codes[:, start // 2:end // 2]
            nibble = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(codes.shape[0], end-start).long()
            table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=codes.device)
            values = table[nibble]
            scale = q.scales[:, start // 16:end // 16].view(torch.float8_e4m3fn).float()
            return (values.reshape(values.shape[0], -1, 16) * scale.unsqueeze(-1)).reshape_as(values) / q.multiplier
        return decode(self.a, self.qa), decode(self.b, self.qb).T

    def injected(self, config):
        """Clone only the targeted buffer; logical B coordinates map to B_rows."""
        if config.location not in {"a", "b"}:
            raise ValueError("Prepared input injection requires location a or b")
        rows, cols = (self.m, self.k) if config.location == "a" else (self.k, self.n)
        width = {"fp32": 32, "bf16": 16, "fp8": 8, "nvfp4": 4}[self.precision]
        if config.row >= rows or config.col >= cols or config.bit >= width:
            raise ValueError("Input fault coordinate/bit outside actual representation")
        source = self.a if config.location == "a" else self.b
        target = source.clone()
        row, col = config.row, config.col
        if config.location == "b" and self.precision in {"fp8", "nvfp4"}:
            row, col = col, row
        if self.precision == "nvfp4":
            target[row, col // 2] ^= 1 << (config.bit + 4 * (col % 2))
        else:
            dtype = {32: torch.int32, 16: torch.int16, 8: torch.uint8}[width]
            mask = 1 << config.bit
            if width > 8 and config.bit == width-1:
                mask = -(1 << config.bit)
            target.view(dtype)[row, col] ^= mask
        return replace(self, **{config.location: target})
