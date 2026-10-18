"""Data preparation using existing Torch ops and official CUTLASS casts.

CUTLASS 4.2.1 sources: ``cutlass/cute/testing.py::convert`` and
``cutlass/torch.py::cute_tensor_like`` provide GPU conversion and FP4 packing.
NVFP4 uses the generic reference formula in
``tools/util/include/cutlass/util/reference/host/gett.hpp``
``compute_1d_scaling_factor_and_quantized_output`` with E2M1 inputs and
E4M3 block16 scales. Example 79b is a related comparison; its output scales
use UE8M0. Scale tensors here retain their natural row layout; the GEMM
binding supplies the official interleaved-layout indices.
"""

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any

import torch


_CONVERTERS = {}


def _cute_view(storage, dtype=None):
    from cutlass.cute.runtime import from_dlpack

    view = from_dlpack(storage, assumed_align=16)
    if dtype is not None:
        view.element_type = dtype
    return view.mark_layout_dynamic(leading_dim=storage.ndim - 1)


def _convert(src, dst, layout_reference):
    """Cache the official launcher; plain @jit calls rebuild IR on every call.

    Both views use the reference's contiguous logical shape and strides, even
    when FP4's backing allocation has one byte per logical element. Pointer
    values remain runtime arguments and are deliberately absent from the key.
    """
    import cutlass.cute as cute

    leading = layout_reference.ndim - 1
    elements = 8 if min(src.element_type.width, dst.element_type.width) == 4 else 4
    if not layout_reference.is_contiguous() or layout_reference.shape[leading] % elements:
        raise ValueError("official conversion requires aligned contiguous tensors")
    layout = (tuple(layout_reference.shape), tuple(layout_reference.stride()), 16)
    key = (
        layout_reference.device, src.element_type, dst.element_type,
        layout, layout, leading, elements,
    )
    if key not in _CONVERTERS:
        _CONVERTERS[key] = cute.compile(cute.testing._convert, src, dst, leading, elements)
    _CONVERTERS[key](src, dst)


@dataclass(frozen=True)
class QuantizedMatrix:
    """Codes reconstruct as ``decoded_codes * scales / multiplier``.

    FP8 has no block scales. NVFP4 codes contain two E2M1 values per byte;
    scales hold E4M3 bytes in natural ``[rows, columns // 16]`` order.
    ``multiplier`` is a scalar CUDA float32 tensor, never a host scalar.
    """

    codes: torch.Tensor
    scales: torch.Tensor | None
    multiplier: torch.Tensor
    precision: str
    shape: tuple[int, int]
    _cute_codes: Any = field(repr=False, compare=False)


def _official_cast(values, dtype, *, decode=False):
    """Call unchanged CUTLASS GPU conversion on its default stream."""
    with torch.cuda.device(values.device):
        current = torch.cuda.current_stream(values.device)
        default = torch.cuda.default_stream(values.device)
        default.wait_stream(current)
        values.record_stream(default)
        # Match cute_tensor_like's public tensor representation, but compile
        # the unchanged official converter once instead of retracing its IR.
        with torch.cuda.stream(default):
            storage = torch.empty_like(values, dtype=torch.int8)
            cute_values = _cute_view(storage, dtype)
            _convert(_cute_view(values), cute_values, values)
            storage.record_stream(default)
            decoded = None
            if decode:
                decoded = torch.empty_like(values, dtype=torch.float32)
                _convert(cute_values, _cute_view(decoded), decoded)
                decoded.record_stream(default)
        current.wait_stream(default)
        storage.record_stream(current)
        if decoded is not None:
            decoded.record_stream(current)
    return cute_values, storage, decoded


def _multiplier(values, precision, policy):
    if isinstance(policy, str):
        if policy != "amax":
            raise ValueError("global_scale must be 'amax' or a positive finite number")
        maximum = values.abs().amax()
        limit = 448.0 if precision == "fp8" else 6.0 * 448.0
        return torch.where(maximum == 0, torch.ones_like(maximum), limit / maximum)
    if isinstance(policy, bool) or not isinstance(policy, Real):
        raise ValueError("global_scale must be 'amax' or a positive finite number")
    if not math.isfinite(policy) or policy <= 0:
        raise ValueError("global_scale must be positive and finite")
    return torch.full((), float(policy), dtype=torch.float32, device=values.device)


@torch.no_grad()
def quantize(matrix, precision, global_scale="amax"):
    """Quantize a CUDA FP32/BF16 matrix without defining a new GPU kernel.

    ``global_scale`` selects the multiplier ``st``: dynamic amax scaling or
    a positive number. GEMM must use ``alpha = (st_a * st_b).reciprocal()``.
    Nonfinite-input monitoring belongs to the training loop.
    """
    import cutlass

    if matrix.device.type != "cuda" or matrix.ndim != 2:
        raise ValueError("quantize requires a two-dimensional CUDA tensor")
    if matrix.dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("quantize requires FP32 or BF16 input")
    precision = "nvfp4" if precision == "fp4" else precision
    if precision not in {"fp8", "nvfp4"}:
        raise ValueError("precision must be fp8 or nvfp4")
    rows, columns = matrix.shape
    alignment = 4 if precision == "fp8" else 16
    if rows == 0 or columns == 0 or columns % alignment:
        raise ValueError(f"nonempty input columns must be divisible by {alignment}")
    values = matrix.float().contiguous()
    st = _multiplier(values, precision, global_scale)
    if precision == "fp8":
        cute_codes, storage, _ = _official_cast(values * st, cutlass.Float8E4M3FN)
        return QuantizedMatrix(
            storage.view(torch.uint8), None, st, precision, (rows, columns), cute_codes
        )

    blocks = values.reshape(rows, columns // 16, 16)
    # Match the reference's rounding sequence: max * (st * (1 / FP4_MAX)),
    # E4M3 round, reciprocal in float32, then multiply by st.
    scale_values = blocks.abs().amax(dim=-1) * (st * (1.0 / 6.0))
    count = scale_values.numel()
    padded = torch.nn.functional.pad(scale_values.reshape(-1), (0, (-count) % 4))
    _, scale_storage, decoded_scales = _official_cast(
        padded, cutlass.Float8E4M3FN, decode=True
    )
    scales = scale_storage[:count].view(torch.uint8).reshape(rows, columns // 16)
    decoded_scales = decoded_scales[:count].reshape(rows, columns // 16)
    acc_scale = (decoded_scales.reciprocal() * st).clamp_max(
        torch.finfo(torch.float32).max
    )
    normalized = (blocks * acc_scale.unsqueeze(-1)).reshape(rows, columns)
    cute_codes, storage, _ = _official_cast(normalized, cutlass.Float4E2M1FN)
    # cute_tensor_like allocates one int8 per logical element, then retypes
    # its CuTe view to FP4. Conversion writes only the first M*K/2 bytes.
    codes = storage.reshape(-1)[: rows * columns // 2].view(torch.uint8)
    codes = codes.reshape(rows, columns // 2)
    return QuantizedMatrix(codes, scales, st, precision, (rows, columns), cute_codes)


@torch.no_grad()
def dequantize(matrix: QuantizedMatrix):
    """Decode with the official conversion kernel, for reference checks."""
    with torch.cuda.device(matrix.codes.device):
        current = torch.cuda.current_stream(matrix.codes.device)
        default = torch.cuda.default_stream(matrix.codes.device)
        default.wait_stream(current)
        matrix.codes.record_stream(default)
        with torch.cuda.stream(default):
            values = torch.empty(matrix.shape, device=matrix.codes.device, dtype=torch.float32)
            _convert(matrix._cute_codes, _cute_view(values), values)
            values.record_stream(default)
        current.wait_stream(default)
        values.record_stream(current)
        if matrix.scales is not None:
            # Reinterpret E4M3 bytes, without another narrowing conversion.
            scales = matrix.scales.view(torch.float8_e4m3fn).float()
            values = (values.reshape(*scales.shape, 16) * scales.unsqueeze(-1)).reshape(
                matrix.shape
            )
        return values / matrix.multiplier
