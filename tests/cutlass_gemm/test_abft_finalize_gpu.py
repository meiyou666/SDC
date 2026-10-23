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


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.cpu().contiguous().view(torch.uint8),
                       expected.cpu().contiguous().view(torch.uint8))


@pytest.mark.parametrize("tiles,n", [
    (tiles, n) for tiles in (1, 2, 17) for n in (16, 144, 1024)
] + [(1, 8191), (1, 8192), (1, 8193), (33, 257)])
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


@pytest.mark.parametrize("tiles,n", [(1, 16), (2, 144), (17, 1024), (33, 257)])
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
    # Parallel ownership of columns must not reorder each column's FP32 sum.
    # CPU tensor additions retain the explicitly specified ascending tile order.
    sequential = torch.zeros(3, 2, n, dtype=torch.float32)
    source = partials.cpu()[..., :n]
    for tile in range(tiles):
        sequential.add_(source[:, :, tile])
    _assert_bits(outputs[0], sequential[1:])
    _assert_bits(outputs[1], sequential[0] - sequential[1])
    _assert_bits(outputs[2], sequential[2].mul(rtol).add(atol))
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


@pytest.mark.parametrize("tiles,n", [(2, 144), (33, 257)])
def test_finalize_diagnostics_disabled_keeps_complete_device_state(tiles, n):
    partials, tile_flags = _inputs(tiles, n)
    partials[0, 1, -1, n - 1] += 1
    tile_flags[-1, -1] = 1
    outputs = _finalize(partials, tile_flags, n=n, diagnostics=False)
    assert outputs[1].shape == outputs[2].shape == (0,)
    assert outputs[0].shape == (2, 2, n)
    _assert_state(outputs, detected=True, hit=1)


@pytest.mark.parametrize("tiles,n", [(2, 144), (33, 257)])
def test_finalize_retry_uses_original_reference_and_preserves_its_alias(tiles, n):
    partials, tile_flags = _inputs(tiles, n)
    first = _finalize(partials, tile_flags, n=n, rtol=0.25)
    original = first[0]
    saved = original.clone()
    retry = partials.clone()
    # A co-varying erroneous retry would pass if its new prediction were accepted.
    retry[0, 1, 0, n - 1] += 32
    retry[1, 1, 0, n - 1] += 32
    retry[2, 1, 0, n - 1] += 1e10
    outputs = _finalize(retry, tile_flags, n=n, rtol=0.25, reference=original)
    assert outputs[0].data_ptr() == original.data_ptr()
    _assert_bits(original, saved)
    _assert_bits(outputs[2], first[2])
    assert outputs[1][1, n - 1].item() == 32
    _assert_state(outputs, detected=True)


@pytest.mark.parametrize("tiles,n", [(2, 144), (33, 257)])
def test_finalize_retry_does_not_consume_new_prediction_or_magnitude(tiles, n):
    partials, tile_flags = _inputs(tiles, n)
    original = _finalize(partials, tile_flags, n=n)[0]
    saved = original.clone()
    partials[1].fill_(float("nan"))
    partials[2].fill_(float("inf"))
    outputs = _finalize(partials, tile_flags, n=n, reference=original)
    assert outputs[0].data_ptr() == original.data_ptr()
    _assert_bits(original, saved)
    _assert_state(outputs)


def test_finalize_retained_outputs_survive_later_calls_on_nondefault_stream():
    n = 257
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        partials, tile_flags = _inputs(33, n)
        first = _finalize(partials, tile_flags, n=n)
        snapshots = [tensor.clone() for tensor in first]
        changed = partials.clone()
        changed[0, 1, -1, n - 1] += 1
        later_flags = tile_flags.clone()
        later_flags[-1, -1] = 1
        second = _finalize(changed, later_flags, n=n)
        third = _finalize(partials, tile_flags, n=n, diagnostics=False)
    stream.synchronize()
    for actual, snapshot, later in zip(first, snapshots, second, strict=True):
        _assert_bits(actual, snapshot)
        assert actual.data_ptr() != later.data_ptr()
    _assert_state(first)
    _assert_state(second, detected=True, hit=1)
    _assert_state(third)
    assert second[1][1, n - 1].item() == 1
    assert third[1].numel() == third[2].numel() == 0


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


def _policy_inputs(algorithm, tiles=3, n=145, *, random=False):
    partials, tile_flags = _inputs(tiles, n, random=random)
    if algorithm == 1:
        partials = partials[:, :1].contiguous()
    return partials, tile_flags


def _policy_finalize(partials, tile_flags, n, *, algorithm, atol=0.125, rtol=0.25,
                     reference=None, diagnostics=True, thresholds=None):
    return torch.ops.sdc_cutlass.abft_finalize(
        partials, tile_flags, n, atol, rtol, reference, diagnostics, algorithm, thresholds,
    )


def _generic_inputs(partials, n, padding=64):
    padded_n = ((n + padding - 1) // padding) * padding
    return {
        "observed": partials[0, ..., :padded_n].contiguous(),
        "predicted": partials[1, ..., :padded_n].contiguous(),
        "magnitude": partials[2, ..., :padded_n].contiguous(),
        "hit": torch.zeros(1, device=partials.device, dtype=torch.int32),
        "nonfinite": torch.zeros(1, device=partials.device, dtype=torch.int32),
        "n": n,
    }


def _generic_finalize(observed, predicted, magnitude, hit, nonfinite, n, *,
                      algorithm=0, atol=0.125, rtol=0.25,
                      reference=None, diagnostics=True, thresholds=None):
    return torch.ops.sdc_cutlass.abft_finalize_reference(
        observed, predicted, magnitude, hit, nonfinite, n, atol, rtol,
        reference, diagnostics, algorithm, thresholds,
    )


def _serial_policy_oracle(partials, n, *, atol=0.125, rtol=0.25, reference=None,
                          diagnostics=True, hit=0, nonfinite=False, thresholds=None):
    """CPU FP32 oracle: every logical column adds M tiles in ascending order."""
    source = partials.cpu()[..., :n]
    channels, tiles = source.shape[1:3]
    observed = torch.zeros(channels, n, dtype=torch.float32)
    for tile in range(tiles):
        observed.add_(source[0, :, tile])
    if reference is None:
        expected_reference = torch.zeros(2, channels, n, dtype=torch.float32)
        for tile in range(tiles):
            expected_reference.add_(source[1:, :, tile])
    else:
        expected_reference = reference.cpu().clone()
    predicted, magnitude = expected_reference.unbind()
    residual = observed - predicted
    threshold = (magnitude.mul(rtol).add(atol) if thresholds is None else
                 thresholds.cpu().expand(channels, n).clone())
    has_nonfinite = nonfinite or not bool(
        torch.isfinite(observed).all() and torch.isfinite(expected_reference).all()
        and torch.isfinite(residual).all() and torch.isfinite(threshold).all()
    )
    detected = has_nonfinite or bool((residual.abs() > threshold).any())
    flags = torch.tensor([detected, has_nonfinite, False, False, False], dtype=torch.bool)
    empty = torch.empty(0, dtype=torch.float32)
    return (expected_reference, residual if diagnostics else empty,
            threshold if diagnostics else empty, flags,
            torch.tensor([int(bool(hit))], dtype=torch.int32),
            torch.full((2,), -1, dtype=torch.int64))


def _assert_packet_bits(actual, expected):
    assert len(actual) == len(expected) == 6
    for left, right in zip(actual, expected, strict=True):
        _assert_bits(left, right)


@pytest.mark.parametrize("algorithm,channels", [(0, 2), (1, 1)])
@pytest.mark.parametrize("tiles,n", [(3, 65), (33, 257)])
def test_compiled_policy_finalizers_match_serial_cpu_oracle(algorithm, channels, tiles, n):
    partials, tile_flags = _policy_inputs(algorithm, tiles, n, random=True)
    # This order-sensitive column distinguishes serial RN additions from a
    # reassociated tree even when the other random values happen to agree.
    partials[0, 0, :3, 0] = torch.tensor([2**24, 1, -(2**24)], device=partials.device)
    outputs = _policy_finalize(partials, tile_flags, n, algorithm=algorithm)
    expected = _serial_policy_oracle(partials, n)
    assert outputs[0].shape == (2, channels, n)
    assert outputs[1].shape == outputs[2].shape == (channels, n)
    _assert_packet_bits(outputs, expected)
    generic = _generic_finalize(**_generic_inputs(partials, n), algorithm=algorithm)
    _assert_packet_bits(generic, expected)
    if algorithm == 0:
        # Omitting the new last argument must preserve the original Classic API.
        default = _finalize(partials, tile_flags, n, atol=0.125, rtol=0.25)
        _assert_packet_bits(outputs, default)


@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.parametrize("padding", [64, 128])
@pytest.mark.parametrize("tiles,n", [(3, 145), (9, 1025)])
def test_generic_policy_matches_packed_with_nan_padding_and_tail_columns(algorithm, padding, tiles, n):
    partials, tile_flags = _policy_inputs(algorithm, tiles, n)
    partials[0, -1, -1, n - 1] += 64
    values = _generic_inputs(partials, n, padding)
    assert values["observed"].shape[-1] == ((n + padding - 1) // padding) * padding
    assert torch.isnan(values["observed"][..., n:]).all()
    generic = _generic_finalize(**values, algorithm=algorithm)
    packed = _policy_finalize(partials, tile_flags, n, algorithm=algorithm)
    _assert_packet_bits(generic, packed)
    _assert_packet_bits(generic, _serial_policy_oracle(partials, n))
    assert generic[1][-1, n - 1].item() == 64
    _assert_state(generic, detected=True)


@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.parametrize("hit,nonfinite", [(1, 0), (0, 1), (1, 1)])
def test_generic_external_flags_change_only_status(algorithm, hit, nonfinite):
    n = 1025
    partials, tile_flags = _policy_inputs(algorithm, 9, n)
    values = _generic_inputs(partials, n)
    clean = _generic_finalize(**values, algorithm=algorithm)
    values["hit"].fill_(hit)
    values["nonfinite"].fill_(nonfinite)
    outputs = _generic_finalize(**values, algorithm=algorithm)
    for actual, expected in zip(outputs[:3], clean[:3], strict=True):
        _assert_bits(actual, expected)
    tile_flags[-1, -1] = hit | (nonfinite << 1)
    control = _policy_finalize(partials, tile_flags, n, algorithm=algorithm)
    _assert_packet_bits(outputs, control)
    _assert_state(outputs, detected=bool(nonfinite), nonfinite=bool(nonfinite), hit=hit)


@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.parametrize("entry", ["packed", "generic"])
def test_policy_retry_preserves_first_reference_and_ignores_new_evidence(algorithm, entry):
    n = 1025
    partials, tile_flags = _policy_inputs(algorithm, 9, n)

    def invoke(data, reference=None):
        if entry == "packed":
            return _policy_finalize(data, tile_flags, n, algorithm=algorithm, reference=reference)
        return _generic_finalize(**_generic_inputs(data, n), algorithm=algorithm, reference=reference)

    first = invoke(partials)
    snapshots = [tensor.clone() for tensor in first]
    retry_partials = partials.clone()
    retry_partials[1].fill_(float("nan"))
    retry_partials[2].fill_(float("inf"))
    retry = invoke(retry_partials, first[0])
    assert retry[0].data_ptr() == first[0].data_ptr()
    _assert_packet_bits(retry, first)
    retry_partials[0, -1, -1, n - 1] += 64
    damaged = invoke(retry_partials, first[0])
    assert damaged[0].data_ptr() == first[0].data_ptr()
    _assert_packet_bits(first, snapshots)
    _assert_packet_bits(damaged, _serial_policy_oracle(retry_partials, n, reference=snapshots[0]))
    _assert_bits(damaged[2], first[2])
    _assert_state(damaged, detected=True)


@pytest.mark.parametrize("algorithm,channels", [(0, 2), (1, 1)])
def test_generic_policy_diagnostics_disabled_keeps_reference_and_flags(algorithm, channels):
    n = 145
    partials, tile_flags = _policy_inputs(algorithm, 3, n)
    partials[0, -1, -1, n - 1] += 64
    values = _generic_inputs(partials, n)
    values["hit"].fill_(1)
    tile_flags[-1, -1] = 1
    outputs = _generic_finalize(**values, algorithm=algorithm, diagnostics=False)
    expected = _policy_finalize(partials, tile_flags, n, algorithm=algorithm, diagnostics=False)
    assert outputs[0].shape == (2, channels, n)
    assert outputs[1].shape == outputs[2].shape == (0,)
    _assert_packet_bits(outputs, expected)
    _assert_state(outputs, detected=True, hit=1)


@pytest.mark.parametrize("algorithm", [0, 1])
def test_generic_policy_retained_reports_on_nondefault_stream(algorithm):
    n = 1025
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        partials, _ = _policy_inputs(algorithm, 9, n)
        values = _generic_inputs(partials, n)
        first = _generic_finalize(**values, algorithm=algorithm)
        snapshots = [tensor.clone() for tensor in first]
        changed = dict(values)
        changed["observed"] = values["observed"].clone()
        changed["observed"][-1, -1, n - 1] += 64
        changed["hit"] = torch.ones_like(values["hit"])
        second = _generic_finalize(**changed, algorithm=algorithm)
        third = _generic_finalize(**values, algorithm=algorithm, diagnostics=False)
    stream.synchronize()
    _assert_packet_bits(first, snapshots)
    for original, later in zip(first, second, strict=True):
        assert original.data_ptr() != later.data_ptr()
    _assert_state(first)
    _assert_state(second, detected=True, hit=1)
    _assert_state(third)
    assert second[1][-1, n - 1].item() == 64
    assert third[1].numel() == third[2].numel() == 0


@pytest.mark.parametrize("algorithm", [-1, 2, 2**32])
@pytest.mark.parametrize("entry", ["packed", "generic"])
def test_finalize_rejects_unknown_compiled_algorithm_before_narrowing(algorithm, entry):
    partials, tile_flags = _policy_inputs(0)
    with pytest.raises(RuntimeError, match="Unknown compiled ABFT algorithm ID"):
        if entry == "packed":
            _policy_finalize(partials, tile_flags, 145, algorithm=algorithm)
        else:
            _generic_finalize(**_generic_inputs(partials, 145), algorithm=algorithm)


@pytest.mark.parametrize("algorithm,channels", [(0, 1), (1, 2)])
def test_compiled_policy_rejects_wrong_packed_channel_count(algorithm, channels):
    partials, tile_flags = _policy_inputs(0)
    partials = partials[:, :channels].contiguous()
    with pytest.raises(RuntimeError):
        _policy_finalize(partials, tile_flags, 145, algorithm=algorithm)


@pytest.mark.parametrize("invalid", [
    "observed_rank", "predicted_shape", "magnitude_channels", "empty_tiles", "padding",
    "observed_dtype", "predicted_dtype", "magnitude_dtype",
    "observed_noncontiguous", "predicted_noncontiguous", "magnitude_noncontiguous",
    "observed_cpu", "predicted_cpu", "magnitude_cpu", "observed_sparse",
    "hit_rank", "hit_size", "hit_dtype", "hit_cpu",
    "nonfinite_rank", "nonfinite_size", "nonfinite_dtype", "nonfinite_cpu",
    "reference_rank", "reference_channels", "reference_n", "reference_dtype",
    "reference_noncontiguous", "reference_cpu", "reference_sparse",
    "n_zero", "n_negative", "n_padding", "n_int32_overflow",
])
def test_generic_finalize_rejects_invalid_input_contract(invalid):
    partials, _ = _policy_inputs(1)
    values = _generic_inputs(partials, 145)
    reference = torch.zeros(2, 1, 145, device=partials.device, dtype=torch.float32)
    if invalid == "observed_rank":
        values["observed"] = values["observed"].flatten(0, 1)
    elif invalid == "predicted_shape":
        values["predicted"] = values["predicted"][:, :1].contiguous()
    elif invalid == "magnitude_channels":
        values["magnitude"] = values["magnitude"].repeat(2, 1, 1)
    elif invalid in ("empty_tiles", "padding"):
        for field in ("observed", "predicted", "magnitude"):
            tensor = values[field]
            values[field] = (tensor[:, :0] if invalid == "empty_tiles" else tensor[..., :-1]).contiguous()
    elif invalid == "observed_sparse":
        values["observed"] = values["observed"].to_sparse()
    elif invalid.startswith(("observed_", "predicted_", "magnitude_")):
        field, issue = invalid.split("_", 1)
        if issue == "dtype":
            values[field] = values[field].bfloat16()
        elif issue == "noncontiguous":
            values[field] = values[field].repeat_interleave(2, dim=-1)[..., ::2]
        else:
            values[field] = values[field].cpu()
    elif invalid.startswith(("hit_", "nonfinite_")):
        field, issue = invalid.split("_", 1)
        if issue == "rank":
            values[field] = values[field].reshape(1, 1)
        elif issue == "size":
            values[field] = values[field].repeat(2)
        elif issue == "dtype":
            values[field] = values[field].bool()
        else:
            values[field] = values[field].cpu()
    elif invalid.startswith("reference_"):
        issue = invalid.removeprefix("reference_")
        if issue == "rank":
            reference = reference.flatten(0, 1)
        elif issue == "channels":
            reference = reference.repeat(1, 2, 1)
        elif issue == "n":
            reference = reference[..., :-1].contiguous()
        elif issue == "dtype":
            reference = reference.bfloat16()
        elif issue == "noncontiguous":
            reference = reference.repeat_interleave(2, dim=-1)[..., ::2]
        elif issue == "sparse":
            reference = reference.to_sparse()
        else:
            reference = reference.cpu()
    else:
        values["n"] = {"n_zero": 0, "n_negative": -1, "n_padding": 193,
                       "n_int32_overflow": 2**31 - 128}[invalid]
    with pytest.raises((RuntimeError, NotImplementedError)):
        _generic_finalize(**values, algorithm=1, reference=reference)


@pytest.mark.parametrize("name", ["atol", "rtol"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), torch.finfo(torch.float32).max * 2])
def test_generic_finalize_rejects_invalid_tolerances(name, value):
    partials, _ = _policy_inputs(1)
    values = _generic_inputs(partials, 145)
    with pytest.raises(RuntimeError, match="tolerances"):
        _generic_finalize(**values, algorithm=1, **{name: value})


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires a second CUDA device")
@pytest.mark.parametrize("argument", ["observed", "predicted", "magnitude", "hit", "nonfinite", "reference"])
def test_generic_finalize_rejects_different_cuda_device(argument):
    partials, _ = _policy_inputs(1)
    values = _generic_inputs(partials, 145)
    reference = None
    other_device = torch.device("cuda", (partials.device.index + 1) % torch.cuda.device_count())
    if argument == "reference":
        reference = torch.zeros(2, 1, 145, device=other_device, dtype=torch.float32)
    else:
        values[argument] = values[argument].to(other_device)
    with pytest.raises(RuntimeError, match="same CUDA device"):
        _generic_finalize(**values, algorithm=1, reference=reference)


def _threshold_finalize(entry, partials, tile_flags, n, *, algorithm, **kwargs):
    if entry == "packed":
        return _policy_finalize(partials, tile_flags, n, algorithm=algorithm, **kwargs)
    return _generic_finalize(**_generic_inputs(partials, n), algorithm=algorithm, **kwargs)


@pytest.mark.parametrize("algorithm,channels", [(0, 2), (1, 1)])
@pytest.mark.parametrize("entry", ["packed", "generic"])
@pytest.mark.parametrize("scalar", [False, True], ids=["per_column", "scalar"])
def test_gpu_threshold_override_controls_decision_and_is_read_only(algorithm, channels, entry, scalar):
    n = 145
    partials, tile_flags = _policy_inputs(algorithm, 3, n)
    partials[0, 0, 0, 0] += 1
    partials[0, -1, -1, n - 1] += 4
    thresholds = torch.full((1,) if scalar else (channels, n),
                            0.5 if scalar else 8.0, device=partials.device, dtype=torch.float32)
    if not scalar:
        thresholds[-1, n - 1] = 0.5
    snapshot = thresholds.clone()
    outputs = _threshold_finalize(entry, partials, tile_flags, n, algorithm=algorithm,
                                  atol=1e12, rtol=1e12, thresholds=thresholds)
    expected = _serial_policy_oracle(partials, n, atol=1e12, rtol=1e12, thresholds=thresholds)
    _assert_packet_bits(outputs, expected)
    _assert_bits(thresholds, snapshot)
    _assert_state(outputs, detected=True)
    ordinary = _threshold_finalize(entry, partials, tile_flags, n, algorithm=algorithm,
                                   atol=1e12, rtol=1e12)
    _assert_state(ordinary)
    _assert_bits(outputs[0], ordinary[0])
    _assert_bits(outputs[1], ordinary[1])
    silent = _threshold_finalize(entry, partials, tile_flags, n, algorithm=algorithm,
                                 atol=1e12, rtol=1e12, thresholds=thresholds, diagnostics=False)
    assert silent[1].shape == silent[2].shape == (0,)
    _assert_bits(silent[0], outputs[0])
    _assert_state(silent, detected=True)


@pytest.mark.parametrize("algorithm,entry", [(0, "packed"), (1, "generic")])
def test_threshold_snapshot_and_first_reference_stay_frozen_on_retry(algorithm, entry):
    n = 1025
    partials, tile_flags = _policy_inputs(algorithm, 9, n)
    channels = partials.shape[1]
    supplied = torch.full((channels, n) if algorithm == 0 else (1,), 0.5,
                          device=partials.device, dtype=torch.float32)
    snapshot = supplied.clone()
    first = _threshold_finalize(entry, partials, tile_flags, n, algorithm=algorithm, thresholds=snapshot)
    saved_first = [tensor.clone() for tensor in first]
    saved_thresholds = snapshot.clone()
    # The caller freezes its threshold snapshot along with first-call evidence.
    # Later adaptive-threshold values must not replace that chosen snapshot.
    supplied.fill_(1e12)
    retry_partials = partials.clone()
    retry_partials[0, -1, -1, n - 1] += 4
    retry_partials[1].fill_(float("nan"))
    retry_partials[2].fill_(float("inf"))
    retry = _threshold_finalize(entry, retry_partials, tile_flags, n, algorithm=algorithm,
                                reference=first[0], thresholds=snapshot)
    assert retry[0].data_ptr() == first[0].data_ptr()
    _assert_packet_bits(first, saved_first)
    _assert_bits(snapshot, saved_thresholds)
    _assert_bits(retry[2], first[2])
    _assert_packet_bits(retry, _serial_policy_oracle(
        retry_partials, n, reference=saved_first[0], thresholds=saved_thresholds,
    ))
    _assert_state(retry, detected=True)


@pytest.mark.parametrize("entry", ["packed", "generic"])
@pytest.mark.parametrize("value,nonfinite", [
    (-1.0, False), (float("nan"), True), (float("inf"), True), (-float("inf"), True),
])
def test_gpu_threshold_negative_and_nonfinite_values_are_reported(entry, value, nonfinite):
    n = 145
    partials, tile_flags = _policy_inputs(1, 3, n)
    thresholds = torch.tensor([value], device=partials.device, dtype=torch.float32)
    snapshot = thresholds.clone()
    outputs = _threshold_finalize(entry, partials, tile_flags, n, algorithm=1, thresholds=thresholds)
    _assert_bits(outputs[2], thresholds.expand(1, n).contiguous())
    _assert_bits(thresholds, snapshot)
    _assert_state(outputs, detected=True, nonfinite=nonfinite)


@pytest.mark.parametrize("entry", ["packed", "generic"])
def test_threshold_override_skips_zero_times_infinite_magnitude(entry):
    n = 145
    partials, tile_flags = _policy_inputs(1, 3, n)
    partials[2, 0, -1, n - 1] = float("inf")
    thresholds = torch.tensor([0.5], device=partials.device, dtype=torch.float32)
    outputs = _threshold_finalize(entry, partials, tile_flags, n, algorithm=1,
                                  atol=1e12, rtol=0.0, thresholds=thresholds)
    _assert_bits(outputs[2], thresholds.expand(1, n).contiguous())
    assert torch.isfinite(outputs[2]).all()
    # Infinite evidence is still invalid, even though the override is finite.
    _assert_state(outputs, detected=True, nonfinite=True)


@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.parametrize("entry", ["packed", "generic"])
def test_none_threshold_and_omitted_threshold_are_bitwise_identical(algorithm, entry):
    n = 145
    partials, tile_flags = _policy_inputs(algorithm, 3, n, random=True)
    explicit = _threshold_finalize(entry, partials, tile_flags, n, algorithm=algorithm, thresholds=None)
    if entry == "packed":
        omitted = torch.ops.sdc_cutlass.abft_finalize(partials, tile_flags, n, 0.125, 0.25, None, True, algorithm)
    else:
        values = _generic_inputs(partials, n)
        omitted = torch.ops.sdc_cutlass.abft_finalize_reference(
            values["observed"], values["predicted"], values["magnitude"], values["hit"],
            values["nonfinite"], n, 0.125, 0.25, None, True, algorithm,
        )
    _assert_packet_bits(explicit, omitted)


@pytest.mark.parametrize("entry", ["packed", "generic"])
@pytest.mark.parametrize("invalid", ["scalar_rank", "channels", "columns", "flat", "dtype",
                                     "cpu", "noncontiguous", "sparse"])
def test_finalize_rejects_invalid_gpu_threshold_tensor(entry, invalid):
    n = 145
    partials, tile_flags = _policy_inputs(0, 3, n)
    thresholds = torch.ones(2, n, device=partials.device, dtype=torch.float32)
    if invalid == "scalar_rank":
        thresholds = thresholds[0, 0]
    elif invalid == "channels":
        thresholds = thresholds[:1]
    elif invalid == "columns":
        thresholds = thresholds[:, :-1].contiguous()
    elif invalid == "flat":
        thresholds = thresholds.flatten()
    elif invalid == "dtype":
        thresholds = thresholds.bfloat16()
    elif invalid == "cpu":
        thresholds = thresholds.cpu()
    elif invalid == "noncontiguous":
        thresholds = thresholds.repeat_interleave(2, dim=-1)[..., ::2]
    else:
        thresholds = thresholds.to_sparse()
    with pytest.raises((RuntimeError, NotImplementedError)):
        _threshold_finalize(entry, partials, tile_flags, n, algorithm=0, thresholds=thresholds)


@pytest.mark.parametrize("entry", ["packed", "generic"])
@pytest.mark.parametrize("algorithm,scalar", [(0, False), (1, True)])
def test_threshold_reference_overlap_rejected_but_disjoint_shared_storage_allowed(entry, algorithm, scalar):
    n = 145
    partials, tile_flags = _policy_inputs(algorithm, 3, n)
    channels = partials.shape[1]
    first = _threshold_finalize(entry, partials, tile_flags, n, algorithm=algorithm)
    overlapping = first[0].reshape(-1)[:1] if scalar else first[0][0]
    with pytest.raises(RuntimeError, match="overlap|memory location"):
        _threshold_finalize(entry, partials, tile_flags, n, algorithm=algorithm,
                            reference=first[0], thresholds=overlapping)
    reference_size = 2 * channels * n
    threshold_size = 1 if scalar else channels * n
    storage = torch.empty(reference_size + threshold_size, device=partials.device, dtype=torch.float32)
    reference = storage[:reference_size].view(2, channels, n)
    thresholds = storage[reference_size:].view(1) if scalar else storage[reference_size:].view(channels, n)
    reference.copy_(first[0])
    thresholds.fill_(0.5)
    snapshot = storage.clone()
    assert reference.untyped_storage().data_ptr() == thresholds.untyped_storage().data_ptr()
    outputs = _threshold_finalize(entry, partials, tile_flags, n, algorithm=algorithm,
                                  reference=reference, thresholds=thresholds)
    assert outputs[0].data_ptr() == reference.data_ptr()
    _assert_bits(storage, snapshot)
    _assert_bits(outputs[2], thresholds.expand(channels, n).contiguous())
    _assert_state(outputs)


@pytest.mark.parametrize("entry,algorithm,scalar", [("packed", 0, False), ("generic", 1, True)])
def test_gpu_threshold_changes_are_consumed_by_cuda_graph_replay(entry, algorithm, scalar):
    n = 145
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        partials, tile_flags = _policy_inputs(algorithm, 3, n)
        partials[0, -1, -1, n - 1] += 4
        channels = partials.shape[1]
        thresholds = torch.full((1,) if scalar else (channels, n), 8.0,
                                device=partials.device, dtype=torch.float32)
        for _ in range(3):
            _threshold_finalize(entry, partials, tile_flags, n, algorithm=algorithm, thresholds=thresholds)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = _threshold_finalize(entry, partials, tile_flags, n,
                                          algorithm=algorithm, thresholds=thresholds)
        graph.replay()
        stream.synchronize()
        _assert_state(outputs)
        _assert_bits(outputs[2], thresholds.expand(channels, n).contiguous())
        # Updating the same GPU tensor outside the graph must affect the next
        # replay; capture cannot hide a D2H read or freeze a Python scalar.
        thresholds.fill_(0.5)
        graph.replay()
        stream.synchronize()
        _assert_state(outputs, detected=True)
        _assert_bits(outputs[2], thresholds.expand(channels, n).contiguous())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires a second CUDA device")
@pytest.mark.parametrize("entry", ["packed", "generic"])
def test_finalize_rejects_threshold_on_another_cuda_device(entry):
    partials, tile_flags = _policy_inputs(1)
    other_device = torch.device("cuda", (partials.device.index + 1) % torch.cuda.device_count())
    thresholds = torch.ones(1, device=other_device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same CUDA device"):
        _threshold_finalize(entry, partials, tile_flags, 145, algorithm=1, thresholds=thresholds)
