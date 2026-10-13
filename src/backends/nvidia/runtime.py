"""Shared packing, CUTLASS dispatch and classic ABFT launch order."""

import torch

from gemm.quantization import bit_values, operand_scale
from gemm.types import PROFILES, GemmResult, PackedOperand

from .build import load_backend


def require_device(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("请在 NVIDIA 容器中使用 RTX 5090 运行。")
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("当前内核目标为 SM120（RTX 5090）。")
    load_backend()


def pack_operands(a, b, profile, phase, context):
    require_device(a.device)
    ops = torch.ops.sdc
    p = PROFILES[profile]
    seed = context.quant_seed(phase)
    torch._assert_async(torch.isfinite(a).all() & torch.isfinite(b).all(), "GEMM 输入包含 NaN 或 Inf")
    pa = ops.pack(a.float().contiguous(), p.code, False,
                  profile == "nvfp4" and phase != "forward", seed)
    pb = ops.pack(b.T.float().contiguous(), p.code, profile == "nvfp4" and phase in {"forward", "dgrad"},
                  False, seed ^ 0x9E3779B9)
    return pa, pb


def multiply(pa, pb, m, n, k, profile, mask=None, bits=None):
    ops = torch.ops.sdc
    instrumented = mask is not None
    mask = mask if instrumented else torch.empty(0, device=pa[0].device, dtype=torch.float32)
    bits = bits if bits is not None else torch.full((2,), -1, device=pa[0].device, dtype=torch.int64)
    p = PROFILES[profile]
    if p.code < 2:
        out = ops.dense(pa[0], pb[0], mask, bits, m, n, k, p.code, instrumented)
    else:
        op = ops.mxfp8 if profile == "mxfp8" else ops.nvfp4
        out = op(*pa, *pb, mask, bits, m, n, k, instrumented)
    return out[:m, :n]


def execute(a, b, *, profile, method, phase, context, fault, correct, detector=None):
    ops = torch.ops.sdc
    p = PROFILES[profile]
    m, k = a.shape
    n = b.shape[1]
    pa, pb = pack_operands(a, b, profile, phase, context)
    if detector is not None:
        qa = PackedOperand(*pa, m, k, (m+127)//128*128, (k+127)//128*128, p)
        qb = PackedOperand(*pb, n, k, (n+127)//128*128, (k+127)//128*128, p)
        expected = detector.prepare(qa, qb, phase=phase, seed=context.stream_seed("projection", phase))
    else:
        expected = ops.reference(*pa, *pb, m, n, k, p.code) if method.name != "none" else None
    bits = torch.full((2,), -1, device=a.device, dtype=torch.int64)
    mask = None
    instrumented = fault is not None and fault.site == "accumulator"
    fault_values = None
    if fault and fault.site in {"a", "b"}:
        row, col = (fault.row, fault.col) if fault.site == "a" else (fault.col, fault.row)
        target = pa if fault.site == "a" else pb
        bits = ops.flip(target[0], row * ((k + 127) // 128 * 128) + col, fault.bit, p.bits, fault.enabled)
        fault_values = bit_values(bits, profile, operand_scale(target[1], target[2], profile, (k+127)//128*128, row, col))
    elif instrumented:
        mask = ops.mask(m, n, fault.row, fault.col, fault.bit, fault.enabled, a)
    out = multiply(pa, pb, m, n, k, profile, mask, bits)
    if instrumented and fault.enabled:
        fault_values = bit_values(bits, "fp32")
    if detector is not None:
        status, residual, threshold = detector.verify(out, expected, correct=correct)
    elif method.name == "none":
        status = torch.zeros(6, device=a.device, dtype=torch.int64)
        status[2:4] = -1
        residual = threshold = torch.empty(0, device=a.device)
    else:
        status, residual, threshold = ops.verify(out, expected, method.atol, method.rtol, correct)
    return GemmResult(out, status, residual, threshold, bits, profile, method.name, "nvidia", phase,
                      fault_values=fault_values)
