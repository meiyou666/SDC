"""Compact augmented ABFT preserves the original computation and owns its evidence."""

import pytest
import torch

from cutlass_gemm import backend


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual compact augmented CUTLASS execution requires SM120",
)


@pytest.fixture(scope="module", autouse=True)
def _load_native():
    backend.require_device(torch.device("cuda"))


def _inputs(m, n, k, column_a=False, column_b=False, unit=False):
    generator = torch.Generator().manual_seed(724)
    a = (torch.randn(m, k, generator=generator) / 8).bfloat16().cuda()
    b = (torch.randn(k, n, generator=generator) / 8).bfloat16().cuda()
    weights = torch.randn(2, m, generator=generator).bfloat16().cuda()
    if unit:
        a.zero_()
        b.zero_()
        a[:, 0] = 1
        b[0, :] = 1
        # Nonzero dyadic coefficients also allow checking Inf injection without
        # accidentally introducing 0*Inf in the weighted output projection.
        values = torch.tensor([-1., -.5, .5, 1.], device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(m, device="cuda")
        weights = torch.stack((values[indices % 4], values[(indices * 3 + 1) % 4]))
    if column_a:
        a = a.T.contiguous().T
    if column_b:
        b = b.T.contiguous().T
    return a, b, weights


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.contiguous().view(dtype), expected.contiguous().view(dtype))


def _assert_against_legacy(compact, legacy, m, n):
    output, partials, tile_flags = compact
    reference, observed, hit, predicted, magnitude = legacy
    m_tiles, n_tiles = (m + 127) // 128, (n + 127) // 128
    assert output.dtype == torch.bfloat16 and output.shape == (m, n)
    assert partials.shape == (3, 2, m_tiles, n_tiles * 128)
    assert partials.dtype == torch.float32 and partials.is_contiguous()
    assert tile_flags.shape == (m_tiles, n_tiles) and tile_flags.dtype == torch.int32
    assert output.device == partials.device == tile_flags.device == reference.device
    _assert_bits(output, reference)
    for actual, expected in zip(partials, (observed, predicted, magnitude), strict=True):
        _assert_bits(actual, expected)
    # Empty allocation must not leak prior memory contents through any padded
    # column or projection plane, including the last partial-M CTA.
    padding = partials[..., n:]
    _assert_bits(padding, torch.zeros_like(padding))
    assert (tile_flags & 1).any().item() == bool(hit.item())
    assert not (tile_flags & ~3).any().item()


@pytest.mark.parametrize(
    "m,n,k,column_a,column_b",
    [
        (1, 8, 16, False, False),
        (128, 128, 64, False, True),
        (136, 144, 80, True, False),
        (136, 264, 192, True, True),
        (257, 136, 448, False, False),
        (1, 8, 8, False, False),
        (136, 144, 8, True, False),
        (129, 144, 72, False, False),
        (136, 144, 72, True, False),
    ],
    ids=["one-row-residue", "full-tile-B-column", "A-column-tail-residue",
         "both-column-three-stages", "partial-M-N-seven-stages",
         "half-atom-RR-one-row", "half-atom-A-column-tail",
         "half-atom-residue-RR-tail", "half-atom-residue-A-column-tail"],
)
def test_compact_bitwise_layouts_tails_stage_wrap_and_nondefault_stream(
    m, n, k, column_a, column_b,
):
    # Cases cover four packed layouts, stage wraps, and half-K16-atom residues,
    # rather than multiplying every shape by every layout without new coverage.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a, b, weights = _inputs(m, n, k, column_a, column_b)
        snapshots = [tensor.clone() for tensor in (a, b, weights)]
        compact = torch.ops.sdc_cutlass.abft_augmented_compact(a, b, weights, -1, -1, -1)
        legacy = torch.ops.sdc_cutlass.abft_augmented_mm(a, b, weights, -1, -1, -1)
        unprotected = torch.ops.sdc_cutlass.mm(a, b)
    stream.synchronize()
    _assert_against_legacy(compact, legacy, m, n)
    _assert_bits(compact[0], unprotected)
    assert not compact[2].any().item()
    for actual, before in zip((a, b, weights), snapshots, strict=True):
        _assert_bits(actual, before)


@pytest.mark.parametrize(
    "row,col,bit,expected_flag", [(127, 127, 31, 1), (128, 128, 0, 1), (256, 143, 30, 3)],
    ids=["last-full-tile-sign", "next-tile-invisible-lsb", "last-tail-tile-infinity"],
)
def test_compact_fault_status_is_local_and_matches_legacy(row, col, bit, expected_flag):
    m, n, k = 257, 144, 80
    a, b, weights = _inputs(m, n, k, unit=True)
    if bit == 0:
        # Isolate the one-row contribution so its FP32 LSB survives checksum
        # reduction, although BF16 output rounding masks the injected change.
        weights.zero_()
        weights[:, row] = torch.tensor([1., .5], dtype=torch.bfloat16, device="cuda")
    clean = torch.ops.sdc_cutlass.abft_augmented_compact(a, b, weights, -1, -1, -1)
    corrupted = torch.ops.sdc_cutlass.abft_augmented_compact(a, b, weights, row, col, bit)
    legacy = torch.ops.sdc_cutlass.abft_augmented_mm(a, b, weights, row, col, bit)
    _assert_against_legacy(corrupted, legacy, m, n)
    expected = torch.zeros_like(corrupted[2])
    expected[row // 128, col // 128] = expected_flag
    assert torch.equal(corrupted[2], expected)
    _assert_bits(corrupted[1][1], clean[1][1])  # Prediction precedes the injected error.
    if bit == 0:
        _assert_bits(corrupted[0], clean[0])
        assert corrupted[1][0, 0, row // 128, col] != clean[1][0, 0, row // 128, col]
    elif bit == 30:
        assert torch.isinf(corrupted[0][row, col]).item()


def test_compact_flags_finite_accumulator_overflowing_only_at_bf16_conversion():
    # BF16 max + half a BF16 ULP is finite FP32 but rounds to BF16 infinity.
    # Both input factors are exactly representable BF16 values.
    a = torch.zeros(1, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(16, 8, dtype=torch.bfloat16, device="cuda")
    a[0, 0], a[0, 1] = torch.finfo(torch.bfloat16).max, float(2 ** 119)
    b[:2] = 1
    weights = torch.ones(2, 1, dtype=torch.bfloat16, device="cuda")
    precast = (a.cpu().double() @ b.cpu().double()).float()
    assert torch.isfinite(precast).all().item()
    assert torch.isinf(precast.bfloat16()).all().item()
    output, partials, flags = torch.ops.sdc_cutlass.abft_augmented_compact(
        a, b, weights, -1, -1, -1,
    )
    assert torch.isinf(output).all().item()
    # This specifically requires inspecting the converted output, not merely
    # accumulators, inputs or checksum residuals. There was no injected fault.
    assert flags.tolist() == [[2]]
    _assert_bits(partials[..., 8:], torch.zeros_like(partials[..., 8:]))


def test_compact_nonfinite_flag_follows_actual_output_tiles():
    a, b, weights = _inputs(136, 144, 64, unit=True)
    a[135, 0] = float("nan")
    output, partials, flags = torch.ops.sdc_cutlass.abft_augmented_compact(
        a, b, weights, -1, -1, -1,
    )
    assert torch.isfinite(output[:135]).all().item()
    assert torch.isnan(output[135]).all().item()
    assert flags.tolist() == [[0, 0], [2, 2]]
    _assert_bits(partials[..., 144:], torch.zeros_like(partials[..., 144:]))


def test_compact_returns_owned_buffers_without_mutating_input_or_prior_evidence():
    a, b, weights = _inputs(136, 144, 192, unit=True)
    snapshots = [tensor.clone() for tensor in (a, b, weights)]
    first = torch.ops.sdc_cutlass.abft_augmented_compact(a, b, weights, -1, -1, -1)
    evidence = [tensor.clone() for tensor in first]
    second = torch.ops.sdc_cutlass.abft_augmented_compact(a, b, weights, 135, 143, 31)
    for actual, snapshot, later in zip(first, evidence, second, strict=True):
        _assert_bits(actual, snapshot)
        assert actual.data_ptr() != later.data_ptr()
    assert not first[2].any().item()
    assert second[2].tolist() == [[0, 0], [0, 1]]
    for actual, before in zip((a, b, weights), snapshots, strict=True):
        _assert_bits(actual, before)
