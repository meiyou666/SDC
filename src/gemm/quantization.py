"""Independent CPU representation oracle and shared NVFP4 transforms."""

import torch
import torch.nn.functional as F

from .types import PROFILES, PackedOperand


def pad128(value: int) -> int:
    return (value + 127) // 128 * 128


def scale_offsets(rows: int, columns: int, group: int) -> torch.Tensor:
    """CUTLASS K-major SF layout, indexed by [row, scale group]."""
    r = torch.arange(rows)[:, None]
    g = torch.arange(columns // group)[None, :]
    return (r // 128 * (columns // group // 4) + g // 4) * 512 + r % 32 * 16 + r // 32 % 4 * 4 + g % 4


def bit_values(bits, profile, scale=None):
    if profile == "fp32":
        values = bits.to(torch.int32).view(torch.float32)
    elif profile == "bf16":
        values = bits.to(torch.int16).view(torch.bfloat16).float()
    elif profile == "mxfp8":
        values = bits.to(torch.uint8).view(torch.float8_e4m3fn).float()
    else:
        table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=bits.device)
        values = table[bits & 7] * torch.where(bits & 8 != 0, -1, 1)
    return values if scale is None else values * scale


def operand_scale(scales, global_scale, profile, padded_columns, row, col):
    if profile in {"fp32", "bf16"}:
        return global_scale
    group = PROFILES[profile].scale_group
    g = col // group
    offset = (row // 128 * (padded_columns // group // 4) + g // 4) * 512 + row % 32 * 16 + row // 32 % 4 * 4 + g % 4
    code = scales[offset:offset+1]
    factor = torch.exp2(code.float()-127) if profile == "mxfp8" else code.view(torch.float8_e4m3fn).float()
    return factor * global_scale


def _uniform(seed: int, count: int, device) -> torch.Tensor:
    x = torch.arange(count, dtype=torch.int64, device=device) ^ seed
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    x = ((x ^ (x >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) >> 8).float() * 2**-24


def pack_reference(x, profile, *, tile_scale=False, stochastic=False, seed=0):
    """CPU oracle. Production packing is performed by the CUDA backend."""
    if x.device.type != "cpu":
        raise ValueError("表示参考实现仅用于 CPU 单元测试。")
    p = PROFILES[profile]
    rows, cols = x.shape
    rp, cp = pad128(rows), pad128(cols)
    values = F.pad(x.float(), (0, cp - cols, 0, rp - rows))
    global_scale = torch.ones((), dtype=torch.float32)
    scales = torch.empty(0, dtype=torch.uint8)
    if profile in {"fp32", "bf16"}:
        codes = values.to(torch.float32 if profile == "fp32" else torch.bfloat16).view(torch.uint8).flatten()
    else:
        groups = values.reshape(rp, cp // p.scale_group, p.scale_group)
        maximum = groups.abs().amax(-1)
        if tile_scale:
            maximum = maximum.reshape(rp // 16, 16, -1).amax(1, keepdim=True).expand(-1, 16, -1).reshape_as(maximum)
        if profile == "mxfp8":
            exponent = torch.where(maximum > 0, torch.ceil(torch.log2(maximum / 448)), 0).clamp(-127, 127)
            factors = torch.exp2(exponent)
            scale_codes = (exponent + 127).to(torch.uint8)
        else:
            global_scale = (values.abs().amax() / (448 * 6)).clamp_min(1e-30)
            target = torch.where(maximum > 0, maximum / (6 * global_scale), 1).clamp(2**-9, 448)
            scale_codes = target.to(torch.float8_e4m3fn).view(torch.uint8)
            factors = scale_codes.view(torch.float8_e4m3fn).float()
        normalized = groups / (factors[..., None] * global_scale)
        offsets = scale_offsets(rp, cp, p.scale_group)
        scales = torch.empty(offsets.numel(), dtype=torch.uint8)
        scales[offsets] = scale_codes
        if profile == "mxfp8":
            codes = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).flatten()
        else:
            magnitudes = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
            absolute = normalized.abs().flatten().clamp_max(6)
            hi = torch.searchsorted(magnitudes, absolute).clamp(1, 7)
            lo = hi - 1
            if stochastic:
                choose_hi = _uniform(seed, absolute.numel(), x.device) < (absolute - magnitudes[lo]) / (magnitudes[hi] - magnitudes[lo])
            else:
                middle = (magnitudes[hi] + magnitudes[lo]) / 2
                choose_hi = (absolute > middle) | ((absolute == middle) & (hi % 2 == 0))
            raw = torch.where(choose_hi, hi, lo).to(torch.uint8)
            raw |= torch.signbit(normalized.flatten()).to(torch.uint8) * 8
            codes = raw[::2] | (raw[1::2] << 4)
    return PackedOperand(codes, scales, global_scale, rows, cols, rp, cp, p)


def decode_reference(p: PackedOperand) -> torch.Tensor:
    rp, cp = p.padded_rows, p.padded_columns
    if p.profile.name in {"fp32", "bf16"}:
        dtype = torch.float32 if p.profile.name == "fp32" else torch.bfloat16
        values = p.codes.view(dtype).float().reshape(rp, cp)
    else:
        scale_codes = p.scales[scale_offsets(rp, cp, p.profile.scale_group)]
        if p.profile.name == "mxfp8":
            factors = torch.exp2(scale_codes.float() - 127)
            values = p.codes.view(torch.float8_e4m3fn).float().reshape(rp, cp)
        else:
            factors = scale_codes.view(torch.float8_e4m3fn).float()
            raw = torch.stack((p.codes & 15, p.codes >> 4), -1).flatten().long()
            lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
            values = (lut[raw & 7] * torch.where(raw & 8 != 0, -1, 1)).reshape(rp, cp)
        values *= factors.repeat_interleave(p.profile.scale_group, 1) * p.global_scale
    return values[:p.rows, :p.columns].contiguous()


def hadamard_pair(a, b, seed):
    """Paired H16 on the wgrad reduction axis: (A H)(H^T B) = A B."""
    k = a.shape[1]
    pad = (-k) % 16
    a, bt = F.pad(a, (0, pad)), F.pad(b.T, (0, pad))
    signs = torch.where(_uniform(seed, k + pad, a.device) < .5, -1., 1.)

    def transform(x):
        y = (x.float() * signs).reshape(-1, 16)
        for size in (1, 2, 4, 8):
            z = y.reshape(-1, 16 // (2 * size), 2, size)
            y = torch.stack((z[:, :, 0] + z[:, :, 1], z[:, :, 0] - z[:, :, 1]), 2).reshape(-1, 16)
        return (y / 4).reshape_as(x)

    return transform(a), transform(bt).T
