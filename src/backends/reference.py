"""Explicit CPU oracle; never selected automatically by a device backend."""

import torch

from algorithms.classic.reference import locate_correct, reference, verify
from gemm.quantization import bit_values, decode_reference, operand_scale, pack_reference
from gemm.types import GemmResult


def flip_raw(codes, index, bit, width, enabled=True):
    if width == 4:
        shift = index % 2 * 4
        before = (int(codes[index // 2]) >> shift) & 15
        after = before ^ ((1 << bit) if enabled else 0)
        codes[index // 2] = (int(codes[index // 2]) & ~(15 << shift)) | (after << shift)
    else:
        start = index * width // 8
        before = sum(int(codes[start + j]) << (8 * j) for j in range(width // 8))
        after = before ^ ((1 << bit) if enabled else 0)
        for j in range(width // 8):
            codes[start + j] = (after >> (8 * j)) & 255
    return torch.tensor([before, after], dtype=torch.int64)


def execute(a, b, *, profile, method, phase, context, fault, correct, detector=None):
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("reference backend 只用于 CPU 单元测试。")
    seed = context.quant_seed(phase)
    pa = pack_reference(a, profile, tile_scale=False,
                        stochastic=profile == "nvfp4" and phase != "forward", seed=seed)
    pb = pack_reference(b.T, profile, tile_scale=profile == "nvfp4" and phase in {"forward", "dgrad"},
                        stochastic=False, seed=seed ^ 0x9E3779B9)
    aq, bq = decode_reference(pa), decode_reference(pb).T
    expected = detector.prepare(pa, pb, phase=phase, seed=context.stream_seed("projection", phase)) if detector is not None else reference(aq, bq) if method.name != "none" else None
    bits = torch.full((2,), -1, dtype=torch.int64)
    fault_values = None
    if fault and fault.site in {"a", "b"}:
        target = pa if fault.site == "a" else pb
        row, col = (fault.row, fault.col) if fault.site == "a" else (fault.col, fault.row)
        bits = flip_raw(target.codes, row * target.padded_columns + col, fault.bit, target.profile.bits, fault.enabled)
        fault_values = bit_values(bits, profile, operand_scale(target.scales, target.global_scale, profile, target.padded_columns, row, col))
        aq, bq = decode_reference(pa), decode_reference(pb).T
    # Mathematical oracle rounds once into FP32, distinct from the native reduction path.
    alpha = pa.global_scale * pb.global_scale
    out = ((aq.double() / pa.global_scale) @ (bq.double() / pb.global_scale)).float()
    if fault and fault.site == "accumulator":
        bits = flip_raw(out.view(torch.uint8).flatten(), fault.row * b.shape[1] + fault.col, fault.bit, 32, fault.enabled)
        fault_values = bit_values(bits, "fp32")
    out *= alpha
    status = torch.tensor([0, 0, -1, -1, 0, 0], dtype=torch.int64)
    residual, threshold = torch.empty(0), torch.empty(0)
    if detector is not None:
        status, residual, threshold = detector.verify(out, expected, correct=correct)
    elif method.name != "none":
        if correct:
            out, status, residual, threshold = locate_correct(out, expected, method)
        else:
            # Localization is available in detect-only mode without changing C.
            _, status, residual, threshold = locate_correct(out, expected, method)
            status[4] = 0
            status[5] = verify(out, expected, method)[2].any()
    return GemmResult(out, status, residual, threshold, bits, profile, method.name, "reference", phase,
                      fault_values=fault_values)
