"""Fused epilogue contracts; execute only on the configured SM120 GPU."""

import pytest
import torch

from cutlass_gemm import backend
from cutlass_gemm.quantization import dequantize, quantize


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual fused ABFT CUTLASS execution requires SM120",
)
PRECISIONS = ["fp32", "bf16", "fp8", "nvfp4"]


def _prepare(precision, m=136, n=144, k=128, unit=False, layouts=(False, False)):
    backend.require_device(torch.device("cuda"))
    torch.manual_seed(2026)
    a = torch.randint(-2, 3, (m, k), device="cuda").float()
    b = torch.randint(-2, 3, (n, k), device="cuda").float()
    # Unit multiplier and a max of 6 in each block make NVFP4 scales exactly 1.
    a[:, ::16] = 6
    b[:, ::16] = 6
    if unit:
        a.zero_()
        b.zero_()
        a[:, 0] = 1
        b[:, 0] = 1
    if precision in ("fp32", "bf16"):
        dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        a, b = a.to(dtype), b.T.contiguous().to(dtype)
        if layouts[0]:
            a = a.T.contiguous().T
        if layouts[1]:
            b = b.T.contiguous().T
        args = (a, b)
        reference = (a.double() @ b.double()).float()
        name, tile_n = "mm", 128
    else:
        qa, qb = quantize(a, precision, 1.0), quantize(b, precision, 1.0)
        alpha = torch.tensor(1.0 if unit else 0.25, device="cuda")
        # Reference consumes the SAME prepared codes/scales as the native call.
        reference = ((dequantize(qa).double() @ dequantize(qb).double().T) * alpha.double()).float()
        if precision == "fp8":
            args = (qa.codes.view(torch.float8_e4m3fn), qb.codes.view(torch.float8_e4m3fn), alpha)
            name, tile_n = "fp8_mm", 64
        else:
            ia, ib, size_a, size_b = backend._scale_layout(m, n, k, a.device)
            args = (qa.codes, qb.codes, backend._pack_scales(qa.scales, ia, size_a),
                    backend._pack_scales(qb.scales, ib, size_b), k, alpha)
            name, tile_n = "nvfp4_mm", 128
    return getattr(torch.ops.sdc_cutlass, name), getattr(torch.ops.sdc_cutlass, "abft_" + name), args, reference, tile_n


def _weights(m):
    # Independent signed, non-integral projection vectors, not all-ones/checksum indices.
    return torch.randn(2, m, device="cuda", dtype=torch.float32)


def _assert_partial(partial, weights, reference, tile_n):
    m, n = reference.shape
    assert partial.dtype == torch.float32 and partial.device == reference.device
    assert partial.shape == (2, (m + 127) // 128, ((n + tile_n - 1) // tile_n) * tile_n)
    for tile, start in enumerate(range(0, m, 128)):
        expected = weights[:, start:start + 128].double() @ reference[start:start + 128].double()
        torch.testing.assert_close(partial[:, tile, :n].double(), expected, rtol=2e-5, atol=2e-3)
    assert torch.count_nonzero(partial[:, :, n:]).item() == 0


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("shape", [(128, 128), (136, 144), (264, 272)])
def test_disabled_output_and_arbitrary_weight_cta_projections(precision, shape):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        base, fused, args, reference, tile_n = _prepare(precision, *shape)
        weights = _weights(shape[0])
        baseline = base(*args)
        output, partial, hit = fused(*args, weights, -1, -1, -1)
    stream.synchronize()
    assert output.stride() == baseline.stride()
    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    torch.testing.assert_close(output, reference.to(output.dtype), rtol=0, atol=0)
    assert hit.numel() == 1 and hit.item() == 0
    _assert_partial(partial, weights, reference, tile_n)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("layouts", [(False, True), (True, False), (True, True)])
def test_legacy_packed_operand_layouts(precision, layouts):
    base, fused, args, reference, tile_n = _prepare(precision, layouts=layouts)
    weights = _weights(reference.shape[0])
    output, partial, hit = fused(*args, weights, -1, -1, -1)
    torch.testing.assert_close(output, base(*args), rtol=0, atol=0)
    _assert_partial(partial, weights, reference, tile_n)
    assert hit.item() == 0


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("location", [(0, 0), (127, 127), (128, 128), (135, 143)])
def test_bitflip_exact_global_position_and_projection(precision, location):
    _, fused, args, reference, tile_n = _prepare(precision)
    weights = _weights(reference.shape[0])
    row, col = location
    # Sign bit remains visible after BF16 conversion and must affect exactly one cell.
    expected = reference.clone()
    expected.view(torch.int32)[row, col] ^= -2147483648
    output, partial, hit = fused(*args, weights, row, col, 31)
    assert hit.item() == 1
    torch.testing.assert_close(output, expected.to(output.dtype), rtol=0, atol=0)
    _assert_partial(partial, weights, expected, tile_n)


@pytest.mark.parametrize("precision", ["bf16", "nvfp4"])
def test_projection_precedes_bf16_rounding(precision):
    _, fused, args, reference, tile_n = _prepare(precision)
    row, col = 128, 128
    # Isolate the affected row so an FP32 mantissa flip cannot disappear in a sum.
    weights = torch.zeros(2, reference.shape[0], device="cuda")
    weights[:, row] = torch.tensor([1., -2.], device="cuda")
    original, _, _ = fused(*args, weights, -1, -1, -1)
    output, partial, hit = fused(*args, weights, row, col, 0)
    expected = reference.clone()
    expected.view(torch.int32)[row, col] ^= 1
    assert hit.item() == 1
    torch.testing.assert_close(output, original, rtol=0, atol=0)
    torch.testing.assert_close(partial[:, row // 128, col], weights[:, row] * expected[row, col], rtol=0, atol=0)
    assert partial[0, row // 128, col] != reference[row, col]
    _assert_partial(partial, weights, expected, tile_n)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_nonfinite_injection_is_not_masked(precision):
    _, fused, args, reference, _ = _prepare(precision, unit=True)
    # Derive exponent-bit injection from the actual prepared-operand reference.
    # Legacy/FP8 unit dot is exactly 1; NVFP4 block scaling can perturb it.
    value = reference[128, 128]
    bits = value.view(torch.int32).item()
    candidates = [bit for bit in range(23, 31) if ((bits ^ (1 << bit)) & 0x7f800000) == 0x7f800000]
    assert candidates, "Unit reference must permit a one-bit nonfinite injection"
    weights = torch.ones(2, reference.shape[0], device="cuda")
    output, partial, hit = fused(*args, weights, 128, 128, candidates[0])
    assert hit.item() == 1
    assert not torch.isfinite(output[128, 128])
    assert not torch.isfinite(partial[:, 1, 128]).any()
    output[128, 128] = 0
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("fault", [(0, -1, -1), (-1, 0, 0), (136, 0, 0), (0, 144, 0), (0, 0, 32), (0, 0, -2)])
def test_invalid_fault_rejected(precision, fault):
    _, fused, args, reference, _ = _prepare(precision)
    with pytest.raises(RuntimeError, match="fault"):
        fused(*args, _weights(reference.shape[0]), *fault)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("invalid", ["dtype", "shape", "noncontiguous", "cpu"])
def test_invalid_weights_rejected(precision, invalid):
    _, fused, args, reference, _ = _prepare(precision)
    m = reference.shape[0]
    weights = _weights(m)
    if invalid == "dtype":
        weights = weights.bfloat16()
    elif invalid == "shape":
        weights = weights[:, :-1].contiguous()
    elif invalid == "noncontiguous":
        weights = torch.ones(m, 2, device="cuda").T
    else:
        weights = weights.cpu()
    with pytest.raises(RuntimeError, match="weights|CUDA device"):
        fused(*args, weights, -1, -1, -1)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_execute_clean_and_recovered_with_multiblock_reference(precision):
    from abft import execute

    backend.require_device(torch.device("cuda"))
    scaling = {"fp32": "none", "bf16": "none", "fp8": "tensorwise", "nvfp4": "block16"}[precision]
    a = torch.ones(136, 288, device="cuda")
    b = torch.ones(288, 144, device="cuda")
    config = {"method": "classic", "rtol": 1e-5, "atol": 1e-3,
              "reference_block_k": 128, "diagnostics": True}
    baseline = backend.gemm(a, b, precision, scaling, global_scale=1.0)
    clean, report = execute(a, b, precision, scaling, global_scale=1.0, abft_config=config)
    assert clean.dtype == baseline.dtype and clean.stride() == baseline.stride()
    torch.testing.assert_close(clean, baseline, rtol=0, atol=0)
    assert not report.detected.item() and not report.nonfinite.item()
    assert report.fault_hit.item() == 0
    assert torch.isfinite(report.residual).all()
    assert (report.residual.abs() <= report.threshold).all()

    recovered, fault_report = execute(
        a, b, precision, scaling, global_scale=1.0,
        abft_config={**config, "recovery": "recompute"},
        fault_config={"location": "accumulator", "row": 135, "col": 143, "bit": 31},
    )
    assert fault_report.fault_hit.item() == 1 and fault_report.detected.item()
    assert fault_report.located.item()
    assert fault_report.location.tolist() == [135, 143]
    assert fault_report.recovered.item() and not fault_report.recheck_failed.item()
    assert torch.isfinite(fault_report.retry_residual).all()
    assert (fault_report.retry_residual.abs() <= fault_report.threshold).all()
    assert recovered.dtype == clean.dtype and recovered.stride() == clean.stride()
    torch.testing.assert_close(recovered, clean, rtol=0, atol=0)
    assert torch.all(a == 1) and torch.all(b == 1)


def test_fp32_odd_native_tail():
    base, fused, args, reference, tile_n = _prepare("fp32", m=137, n=139, k=33)
    weights = _weights(137)
    output, partial, hit = fused(*args, weights, -1, -1, -1)
    torch.testing.assert_close(output, base(*args), rtol=0, atol=0)
    _assert_partial(partial, weights, reference, tile_n)
    assert hit.item() == 0
    expected = reference.clone()
    expected.view(torch.int32)[136, 138] ^= -2147483648
    corrupted, partial, hit = fused(*args, weights, 136, 138, 31)
    torch.testing.assert_close(corrupted, expected, rtol=0, atol=0)
    _assert_partial(partial, weights, expected, tile_n)
    assert hit.item() == 1


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_execute_column_major_prepared_operands(precision):
    from abft import execute
    from abft.operands import PreparedGemm

    backend.require_device(torch.device("cuda"))
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    a = torch.ones(288, 136, device="cuda", dtype=dtype).T
    b = torch.ones(144, 288, device="cuda", dtype=dtype).T
    prepared = PreparedGemm.prepare(a, b, precision, "none", 1.0)
    assert prepared.a.stride() == a.stride() and prepared.b.stride() == b.stride()
    baseline = backend.gemm(a, b, precision, "none", 1.0)
    output, report = execute(
        a, b, precision, "none", global_scale=1.0,
        abft_config={"method": "classic", "rtol": 1e-5, "atol": 1e-3,
                     "reference_block_k": 128, "diagnostics": True},
    )
    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    assert not report.detected.item()
    assert (report.residual.abs() <= report.threshold).all()
