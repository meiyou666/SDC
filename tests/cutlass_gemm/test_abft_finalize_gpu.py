"""Device-side augmented ABFT merging, detection and retry-reference contracts."""

import pytest
import torch

from cutlass_gemm import backend


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="The configured CUTLASS extension requires SM120",
)


@pytest.fixture(scope="module", autouse=True)
def _load_extension():
    backend.require_device(torch.device("cuda"))


def _inputs(tiles=2, n=144, *, random=False):
    padded_n = ((n + 127) // 128) * 128
    generator = torch.Generator().manual_seed(529)
    shape = (3, 2, tiles, padded_n)
    if random:
        partials = torch.randn(shape, generator=generator, dtype=torch.float32)
        partials[2].abs_()
    else:
        partials = torch.randint(-8, 9, shape, generator=generator).float() / 8
        partials[0].copy_(partials[1])
        partials[2].abs_()
    # Padding must never participate in the logical-column comparison.
    partials[..., n:] = float("nan")
    return partials.cuda(), torch.zeros(tiles, padded_n // 128, device="cuda", dtype=torch.int32)


def _finalize(partials, tile_flags, n=144, atol=0.0, rtol=0.0, reference=None, diagnostics=True):
    return torch.ops.sdc_cutlass.abft_finalize(
        partials, tile_flags, n, atol, rtol, reference, diagnostics,
    )


def _assert_state(outputs, *, detected=False, nonfinite=False, hit=0):
    assert len(outputs) == 6
    reference, residual, threshold, flags, actual_hit, location = outputs
    assert reference.dtype == residual.dtype == threshold.dtype == torch.float32
    assert flags.dtype == torch.bool and flags.shape == (5,)
    assert actual_hit.dtype == torch.int32 and actual_hit.shape == (1,)
    assert location.dtype == torch.int64 and location.shape == (2,)
    assert all(tensor.device == reference.device and tensor.is_contiguous() for tensor in outputs)
    assert reference.device.type == "cuda"
    assert flags.tolist() == [detected, nonfinite, False, False, False]
    assert actual_hit.item() == hit
    assert location.tolist() == [-1, -1]


@pytest.mark.parametrize("tiles", [1, 2, 17])
@pytest.mark.parametrize("n", [16, 144, 1024])
def test_finalize_exact_dyadic_global_columns_and_current_stream(tiles, n):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        partials, tile_flags = _inputs(tiles, n)
        outputs = _finalize(partials, tile_flags, n, atol=0.125, rtol=0.25)
    stream.synchronize()
    reference, residual, threshold = outputs[:3]
    # CPU FP64 provides an independent exact sum for these bounded dyadic inputs.
    expected = partials.cpu().double()[..., :n].sum(dim=2)
    assert reference.shape == (2, 2, n)
    assert residual.shape == threshold.shape == (2, n)
    torch.testing.assert_close(reference.cpu(), expected[1:].float(), rtol=0, atol=0)
    torch.testing.assert_close(residual.cpu(), (expected[0] - expected[1]).float(), rtol=0, atol=0)
    torch.testing.assert_close(threshold.cpu(), (0.125 + 0.25 * expected[2]).float(), rtol=0, atol=0)
    _assert_state(outputs)


@pytest.mark.parametrize("tiles,n", [(1, 16), (2, 144), (17, 1024)])
def test_finalize_random_values_allow_only_fp32_reduction_error(tiles, n):
    partials, tile_flags = _inputs(tiles, n, random=True)
    # Dyadic tolerances eliminate host scalar-conversion ambiguity.
    atol, rtol = 1 / 1024, 1 / 256
    outputs = _finalize(partials, tile_flags, n, atol, rtol)
    reference, residual, threshold = (tensor.cpu().double() for tensor in outputs[:3])
    values = partials.cpu().double()[..., :n]
    exact = values.sum(dim=2)
    absolute_sums = values.abs().sum(dim=2)
    unit = torch.finfo(torch.float32).eps / 2
    gamma = (tiles - 1) * unit / (1 - (tiles - 1) * unit)
    sum_error = gamma * absolute_sums
    assert torch.all((reference - exact[1:]).abs() <= sum_error[1:])
    exact_residual = exact[0] - exact[1]
    residual_error = sum_error[0] + sum_error[1]
    residual_error += unit * (exact_residual.abs() + residual_error)
    assert torch.all((residual - exact_residual).abs() <= residual_error)
    exact_threshold = atol + rtol * exact[2]
    # rtol is a power of two, so the product introduces no new rounding here.
    threshold_error = rtol * sum_error[2]
    threshold_error += unit * (exact_threshold.abs() + threshold_error)
    assert torch.all((threshold - exact_threshold).abs() <= threshold_error)
    assert bool((residual.abs() > threshold).any())
    _assert_state(outputs, detected=True)


@pytest.mark.parametrize("field", [0, 1, 2], ids=["observed", "predicted", "magnitude"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_finalize_nonfinite_checks_all_checksum_fields(field, value):
    partials, tile_flags = _inputs()
    partials[field, 1, 1, 143] = value
    outputs = _finalize(partials, tile_flags, atol=1e30)
    _assert_state(outputs, detected=True, nonfinite=True)


@pytest.mark.parametrize("case", ["sum", "residual", "threshold_multiply", "threshold_add"])
def test_finalize_nonfinite_arithmetic_is_not_silently_accepted(case):
    partials, tile_flags = _inputs()
    partials.zero_()
    maximum = torch.finfo(torch.float32).max
    atol, rtol = 0.0, 0.0
    if case == "sum":
        partials[0, 1, :, 143] = maximum
    elif case == "residual":
        partials[0, 1, 0, 143] = maximum
        partials[1, 1, 0, 143] = -maximum
    else:
        partials[2, 1, 0, 143] = maximum
        if case == "threshold_multiply":
            rtol = 2.0
        else:
            atol, rtol = maximum, 1.0
    outputs = _finalize(partials, tile_flags, atol=atol, rtol=rtol)
    diagnostic = outputs[2] if case.startswith("threshold") else outputs[1]
    assert not torch.isfinite(diagnostic[1, 143])
    _assert_state(outputs, detected=True, nonfinite=True)


@pytest.mark.parametrize("tile_value,detected,nonfinite,hit", [
    (0, False, False, 0), (1, False, False, 1), (2, True, True, 0), (3, True, True, 1),
])
def test_finalize_tile_flags_include_bf16_output_overflow(tile_value, detected, nonfinite, hit):
    partials, tile_flags = _inputs(17, 1024)
    # Finite, matching pre-cast checksums alone do not expose BF16 output overflow.
    tile_flags[-1, -1] = tile_value
    outputs = _finalize(partials, tile_flags, n=1024)
    assert torch.isfinite(outputs[0]).all() and torch.isfinite(outputs[1]).all()
    _assert_state(outputs, detected=detected, nonfinite=nonfinite, hit=hit)


def test_finalize_reduces_flags_from_different_threads_and_tiles():
    partials, tile_flags = _inputs(17, 1024)
    tile_flags[0, 0] = 1
    tile_flags[-1, -1] = 2
    _assert_state(_finalize(partials, tile_flags, n=1024), detected=True, nonfinite=True, hit=1)


def test_finalize_diagnostics_disabled_keeps_complete_device_state():
    partials, tile_flags = _inputs()
    partials[0, 1, -1, 143] += 1
    tile_flags[-1, -1] = 1
    outputs = _finalize(partials, tile_flags, diagnostics=False)
    assert outputs[1].shape == outputs[2].shape == (0,)
    assert outputs[0].shape == (2, 2, 144)
    _assert_state(outputs, detected=True, hit=1)


def test_finalize_retry_uses_original_reference_and_preserves_its_alias():
    partials, tile_flags = _inputs()
    first = _finalize(partials, tile_flags, rtol=0.25)
    original = first[0]
    saved = original.clone()
    retry = partials.clone()
    # A co-varying erroneous retry would pass if its new prediction were accepted.
    retry[0, 1, 0, 143] += 8
    retry[1, 1, 0, 143] += 8
    retry[2, 1, 0, 143] += 1e10
    outputs = _finalize(retry, tile_flags, rtol=0.25, reference=original)
    assert outputs[0].data_ptr() == original.data_ptr()
    torch.testing.assert_close(original, saved, rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], first[2], rtol=0, atol=0)
    assert outputs[1][1, 143].item() == 8
    _assert_state(outputs, detected=True)


def test_finalize_retry_does_not_consume_new_prediction_or_magnitude():
    partials, tile_flags = _inputs()
    original = _finalize(partials, tile_flags)[0]
    saved = original.clone()
    partials[1].fill_(float("nan"))
    partials[2].fill_(float("inf"))
    outputs = _finalize(partials, tile_flags, reference=original)
    assert outputs[0].data_ptr() == original.data_ptr()
    torch.testing.assert_close(original, saved, rtol=0, atol=0)
    _assert_state(outputs)


@pytest.mark.parametrize("field", [0, 1], ids=["original_prediction", "original_magnitude"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_finalize_retry_checks_nonfinite_original_reference(field, value):
    partials, tile_flags = _inputs()
    reference = _finalize(partials, tile_flags)[0]
    reference[field, 1, 143] = value
    outputs = _finalize(partials, tile_flags, reference=reference)
    _assert_state(outputs, detected=True, nonfinite=True)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("above", [False, True])
def test_finalize_threshold_is_strict_greater_than(sign, above):
    partials, tile_flags = _inputs(1, 16)
    partials.zero_()
    value = torch.tensor(float(sign), dtype=torch.float32)
    if above:
        value = torch.nextafter(value, torch.tensor(float("inf") * sign))
    partials[0, 1, 0, 15] = value
    outputs = _finalize(partials, tile_flags, n=16, atol=1.0)
    _assert_state(outputs, detected=above)


def test_finalize_accepts_maximum_finite_tolerance_without_false_nonfinite():
    partials, tile_flags = _inputs(1, 16)
    partials.zero_()
    maximum = torch.finfo(torch.float32).max
    partials[0, 0, 0, 15] = maximum
    outputs = _finalize(partials, tile_flags, n=16, atol=maximum)
    assert outputs[2][0, 15].item() == maximum
    _assert_state(outputs)


def test_finalize_preserves_subnormal_tolerance():
    partials, tile_flags = _inputs(1, 16)
    partials.zero_()
    partials[2, 0, 0, 15] = 1
    partials[0, 0, 0, 15] = 2**-149
    outputs = _finalize(partials, tile_flags, n=16, rtol=2**-149)
    assert outputs[2][0, 15].item() == 2**-149
    _assert_state(outputs)


def test_finalize_threshold_keeps_separate_multiply_and_add_not_fma():
    partials, tile_flags = _inputs(1, 16)
    partials.zero_()
    magnitude = rtol = 1 + 2**-23
    atol = 2**-24
    # Multiplication rounds away 2**-46. The subsequent addition lands exactly
    # on an even mantissa midpoint; a fused operation would round one ULP up.
    separate = 1 + 2**-22
    fused = torch.tensor(atol + rtol * magnitude, dtype=torch.float32).item()
    assert fused == 1 + 3 * 2**-23 and fused > separate
    partials[2, 0, 0, 15] = magnitude
    partials[0, 0, 0, 15] = fused
    outputs = _finalize(partials, tile_flags, n=16, atol=atol, rtol=rtol)
    assert outputs[2][0, 15].item() == separate
    _assert_state(outputs, detected=True)


@pytest.mark.parametrize("invalid", [
    "partials_rank", "partials_fields", "partials_channels", "partials_empty_tiles",
    "partials_padding", "partials_dtype", "partials_noncontiguous", "partials_cpu", "partials_sparse",
    "flags_rank", "flags_tiles", "flags_columns", "flags_dtype", "flags_noncontiguous", "flags_cpu",
    "flags_sparse", "n_zero", "n_negative", "n_padding", "n_int32_overflow",
])
def test_finalize_rejects_invalid_native_inputs(invalid):
    partials, tile_flags = _inputs()
    n = 144
    if invalid == "partials_rank":
        partials = partials.flatten(1, 2)
    elif invalid == "partials_fields":
        partials = partials[:2]
    elif invalid == "partials_channels":
        partials = partials[:, :1].contiguous()
    elif invalid == "partials_empty_tiles":
        partials = partials[:, :, :0].contiguous()
    elif invalid == "partials_padding":
        partials = partials[..., :-1].contiguous()
    elif invalid == "partials_dtype":
        partials = partials.bfloat16()
    elif invalid == "partials_noncontiguous":
        partials = partials.repeat_interleave(2, dim=-1)[..., ::2]
    elif invalid == "partials_cpu":
        partials = partials.cpu()
    elif invalid == "partials_sparse":
        partials = partials.to_sparse()
    elif invalid == "flags_rank":
        tile_flags = tile_flags.flatten()
    elif invalid == "flags_tiles":
        tile_flags = tile_flags[:1]
    elif invalid == "flags_columns":
        tile_flags = tile_flags[:, :1].contiguous()
    elif invalid == "flags_dtype":
        tile_flags = tile_flags.long()
    elif invalid == "flags_noncontiguous":
        tile_flags = tile_flags.repeat_interleave(2, dim=1)[:, ::2]
    elif invalid == "flags_cpu":
        tile_flags = tile_flags.cpu()
    elif invalid == "flags_sparse":
        tile_flags = tile_flags.to_sparse()
    else:
        n = {"n_zero": 0, "n_negative": -1, "n_padding": 257, "n_int32_overflow": 2**31 - 128}[invalid]
    with pytest.raises((RuntimeError, NotImplementedError), match="ABFT finalize|SparseCUDA"):
        _finalize(partials, tile_flags, n=n)


@pytest.mark.parametrize("invalid", ["rank", "shape", "dtype", "noncontiguous", "cpu", "sparse"])
def test_finalize_rejects_invalid_original_reference(invalid):
    partials, tile_flags = _inputs()
    reference = _finalize(partials, tile_flags)[0]
    if invalid == "rank":
        reference = reference.flatten(0, 1)
    elif invalid == "shape":
        reference = reference[..., :-1].contiguous()
    elif invalid == "dtype":
        reference = reference.bfloat16()
    elif invalid == "noncontiguous":
        reference = reference.repeat_interleave(2, dim=-1)[..., ::2]
    elif invalid == "cpu":
        reference = reference.cpu()
    else:
        reference = reference.to_sparse()
    with pytest.raises((RuntimeError, NotImplementedError), match="reference|SparseCUDA"):
        _finalize(partials, tile_flags, reference=reference)


@pytest.mark.parametrize("name", ["atol", "rtol"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), torch.finfo(torch.float32).max * 2])
def test_finalize_rejects_invalid_tolerances(name, value):
    partials, tile_flags = _inputs()
    with pytest.raises(RuntimeError, match="tolerances"):
        _finalize(partials, tile_flags, **{name: value})


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires a second CUDA device")
@pytest.mark.parametrize("argument", ["tile_flags", "reference"])
def test_finalize_rejects_different_cuda_device(argument):
    partials, tile_flags = _inputs()
    other_device = (partials.device.index + 1) % torch.cuda.device_count()
    reference = None
    if argument == "tile_flags":
        tile_flags = tile_flags.to(torch.device("cuda", other_device))
    else:
        reference = _finalize(partials, tile_flags)[0].to(torch.device("cuda", other_device))
    with pytest.raises(RuntimeError, match="same CUDA device"):
        _finalize(partials, tile_flags, reference=reference)
