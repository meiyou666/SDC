"""Native PyTorch CUDA library comparator on the same encoded operands."""

import torch


def library_gemm(pa, pb, m, n, k, profile):
    mp, np, kp = (value + 127 for value in (m, n, k))
    mp, np, kp = mp // 128 * 128, np // 128 * 128, kp // 128 * 128
    if profile in {"fp32", "bf16"}:
        dtype = torch.float32 if profile == "fp32" else torch.bfloat16
        a = pa[0].view(dtype).reshape(mp, kp)[:m, :k]
        b = pb[0].view(dtype).reshape(np, kp)[:n, :k].T
        if profile == "bf16":
            # torch.mm(out_dtype=...) uses FP32 output for half/BF16 CUDA inputs.
            return torch.mm(a, b, out_dtype=torch.float32)
        return torch.mm(a, b)
    if profile == "mxfp8":
        dtype, sf_dtype, packed_k = torch.float8_e4m3fn, torch.float8_e8m0fnu, kp
    else:
        dtype, sf_dtype, packed_k = torch.float4_e2m1fn_x2, torch.float8_e4m3fn, kp // 2
    a = pa[0].view(dtype).reshape(mp, packed_k)
    b = pb[0].view(dtype).reshape(np, packed_k).T
    result = torch._scaled_mm(a, b, scale_a=pa[1].view(sf_dtype), scale_b=pb[1].view(sf_dtype),
                              out_dtype=torch.float32, use_fast_accum=False)
    return (result * (pa[2] * pb[2]))[:m, :n]
