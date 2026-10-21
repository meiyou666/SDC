"""BF16 augmented rows share the official MMA and retain its FP32 accumulator.

The encoding is q = BF16(W_BF16 @ A_BF16), per 128-data-row CTA.  Predicted
checksums are q @ B, not an independent FP32 (W @ A) @ B reference.  Two extra
checksum rows use official BF16 MMA fragments without replacing any data row.
"""

import pytest
import torch

from cutlass_gemm import backend


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual augmented CUTLASS execution requires SM120",
)

DATA_ROWS = 128
TILE_N = 128
TILE_K = 64


def _weights(m):
    # Signed, non-integral BF16 projections, independent of classic row indices.
    values = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.bfloat16)
    index = torch.arange(m)
    return torch.stack((values[index % 4], values[(index * 3 + 1) % 4]))


def _prepare(m=129, n=144, k=80, column_b=False, column_a=False, unit=False):
    backend.require_device(torch.device("cuda"))
    generator = torch.Generator().manual_seed(2026)
    # |WA| <= 128/16 with a 1/32 quantum: every encoded value is exact in BF16.
    a = (torch.randint(-1, 2, (m, k), generator=generator).double() / 16).bfloat16()
    b = (torch.randint(-2, 3, (k, n), generator=generator).double() / 8).bfloat16()
    if unit:
        a.zero_()
        b.zero_()
        a[:, 0] = 1
        b[0, :] = 1
    a, b, weights = a.cuda(), b.cuda(), _weights(m).cuda()
    if column_a:
        a = a.T.contiguous().T
    if column_b:
        b = b.T.contiguous().T
    return a, b, weights


def _k_slices(k):
    # CUTLASS visits the residue first: K=80 consumes [0:16], then [16:80].
    first = k % TILE_K or TILE_K
    yield slice(0, first)
    for start in range(first, k, TILE_K):
        yield slice(start, min(start + TILE_K, k))


def _reference(a, b, weights):
    """Independent CPU double model; explicit BF16 rounding at the encoding edge."""
    a, b, weights = (tensor.detach().cpu().double() for tensor in (a, b, weights))
    m, k = a.shape
    n = b.shape[1]
    accumulator = torch.zeros(m, n, dtype=torch.float64)
    shape = (2, (m + DATA_ROWS - 1) // DATA_ROWS, ((n + TILE_N - 1) // TILE_N) * TILE_N)
    observed, predicted, magnitude, unrounded = [torch.zeros(shape, dtype=torch.float64) for _ in range(4)]
    encoding_rounded = False
    for span in _k_slices(k):
        accumulator += a[:, span] @ b[span]
        for tile, start in enumerate(range(0, m, DATA_ROWS)):
            stop = min(start + DATA_ROWS, m)
            q = weights[:, start:stop] @ a[start:stop, span]
            q_cast = q.bfloat16().double()
            encoding_rounded |= not torch.equal(q, q_cast)
            predicted[:, tile, :n] += q_cast @ b[span]
            unrounded[:, tile, :n] += q @ b[span]
    for tile, start in enumerate(range(0, m, DATA_ROWS)):
        stop = min(start + DATA_ROWS, m)
        observed[:, tile, :n] = weights[:, start:stop] @ accumulator[start:stop]
        magnitude[:, tile, :n] = weights[:, start:stop].abs() @ accumulator[start:stop].abs()
    return accumulator, observed, predicted, magnitude, unrounded, encoding_rounded


def _assert_partial(actual, reference, n, *, exact):
    assert actual.dtype == torch.float32 and actual.device.type == "cuda"
    assert actual.shape == reference.shape
    # Padding is part of the native interface, not ignored reference data.
    assert torch.count_nonzero(actual[:, :, n:]).item() == 0
    # Non-integral tests allow FP32 MMA/reduction order differences only; the
    # much larger BF16 q-cast difference is explicitly tested below.
    torch.testing.assert_close(
        actual.cpu().double(), reference,
        atol=0 if exact else 2e-3, rtol=0 if exact else 2e-5,
    )


@pytest.mark.parametrize("m", [1, 16, 128, 129, 256, 257])
@pytest.mark.parametrize("k", [16, 64, 80, 192])
@pytest.mark.parametrize("column_b", [False, True], ids=["B_row", "B_column"])
def test_augmented_dyadic_reference_layout_tails_and_stream(m, k, column_b):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a, b, weights = _prepare(m=m, k=k, column_b=column_b)
        baseline = torch.ops.sdc_cutlass.mm(a, b)
        d, observed, hit, predicted, magnitude = torch.ops.sdc_cutlass.abft_augmented_mm(
            a, b, weights, -1, -1, -1,
        )
    stream.synchronize()
    accumulator, obs_ref, pred_ref, mag_ref, _, rounded = _reference(a, b, weights)
    assert not rounded, "Dyadic fixtures must have exactly representable BF16 encodings"
    assert d.dtype == baseline.dtype == torch.bfloat16
    assert d.stride() == baseline.stride() and d.device == a.device
    torch.testing.assert_close(d, baseline, rtol=0, atol=0)
    assert torch.equal(d.view(torch.int16), baseline.view(torch.int16))
    torch.testing.assert_close(d.cpu(), accumulator.bfloat16(), rtol=0, atol=0)
    assert hit.numel() == 1 and hit.item() == 0
    for actual, reference in [(observed, obs_ref), (predicted, pred_ref), (magnitude, mag_ref)]:
        _assert_partial(actual, reference, b.shape[1], exact=True)
    torch.testing.assert_close(observed, predicted, rtol=0, atol=0)


@pytest.mark.parametrize("m,k", [(136, 16), (264, 80), (392, 192)])
@pytest.mark.parametrize(
    "column_a,column_b", [(False, False), (False, True), (True, False), (True, True)],
    ids=["A_row_B_row", "A_row_B_column", "A_column_B_row", "A_column_B_column"],
)
def test_augmented_packed_layout_combinations_preserve_tail_and_residue(m, k, column_a, column_b):
    # Column-major A requires M aligned to eight; each shape still has an
    # eight-row last CTA, and K=16/80 exercise residue-first shared stages.
    a, b, weights = _prepare(m=m, k=k, column_a=column_a, column_b=column_b)
    baseline = torch.ops.sdc_cutlass.mm(a, b)
    d, observed, hit, predicted, magnitude = torch.ops.sdc_cutlass.abft_augmented_mm(
        a, b, weights, -1, -1, -1,
    )
    _, obs_ref, pred_ref, mag_ref, _, rounded = _reference(a, b, weights)
    assert not rounded
    assert torch.equal(d.view(torch.int16), baseline.view(torch.int16))
    assert hit.item() == 0
    for actual, reference in [(observed, obs_ref), (predicted, pred_ref), (magnitude, mag_ref)]:
        _assert_partial(actual, reference, b.shape[1], exact=True)


@pytest.mark.parametrize("column_b", [False, True], ids=["B_row", "B_column"])
def test_augmented_nonintegral_reference_includes_bf16_encoding_rounding(column_b):
    backend.require_device(torch.device("cuda"))
    m, n, k = 257, 144, 80
    generator = torch.Generator().manual_seed(529)
    a = (torch.randint(-31, 32, (m, k), generator=generator).double() / 32).bfloat16().cuda()
    b = (torch.randint(-15, 16, (k, n), generator=generator).double() / 32).bfloat16().cuda()
    weights = (torch.randint(-7, 8, (2, m), generator=generator).double() / 8).bfloat16().cuda()
    if column_b:
        b = b.T.contiguous().T
    d, observed, hit, predicted, magnitude = torch.ops.sdc_cutlass.abft_augmented_mm(
        a, b, weights, -1, -1, -1,
    )
    _, obs_ref, pred_ref, mag_ref, unrounded, rounded = _reference(a, b, weights)
    assert rounded, "This fixture must exercise the BF16 checksum encoding cast"
    assert (pred_ref - unrounded).abs().max().item() > 0.01
    torch.testing.assert_close(d, torch.ops.sdc_cutlass.mm(a, b), rtol=0, atol=0)
    assert hit.item() == 0
    for actual, reference in [(observed, obs_ref), (predicted, pred_ref), (magnitude, mag_ref)]:
        _assert_partial(actual, reference, n, exact=False)
    assert not torch.allclose(predicted.cpu().double(), unrounded, atol=2e-3, rtol=2e-5)


@pytest.mark.parametrize("n,k", [(144, 18), (145, 80)])
def test_augmented_preserves_official_bf16_alignment_requirements(n, k):
    a, b, weights = _prepare(n=n, k=k)
    with pytest.raises(RuntimeError, match="can_implement|align"):
        torch.ops.sdc_cutlass.abft_augmented_mm(a, b, weights, -1, -1, -1)


@pytest.mark.parametrize("row,col", [(0, 0), (127, 127), (128, 128), (255, 143), (256, 0)])
def test_augmented_fault_hits_data_row_and_keeps_encoded_prediction(row, col):
    a, b, weights = _prepare(m=257, unit=True)
    operation = torch.ops.sdc_cutlass.abft_augmented_mm
    clean, observed, hit, predicted, magnitude = operation(a, b, weights, -1, -1, -1)
    d, corrupted, fault_hit, fault_predicted, fault_magnitude = operation(a, b, weights, row, col, 31)
    expected = clean.clone()
    expected[row, col] = -expected[row, col]
    torch.testing.assert_close(d, expected, rtol=0, atol=0)
    torch.testing.assert_close(fault_predicted, predicted, rtol=0, atol=0)
    # A sign flip changes the observed projection but not its absolute magnitude.
    torch.testing.assert_close(fault_magnitude, magnitude, rtol=0, atol=0)
    delta = torch.zeros_like(observed)
    delta[:, row // DATA_ROWS, col] = -2 * weights[:, row].float()
    torch.testing.assert_close(corrupted - observed, delta, rtol=0, atol=0)
    assert hit.item() == 0 and fault_hit.item() == 1


def test_augmented_projection_uses_accumulator_before_bf16_output_cast():
    a, b, weights = _prepare(unit=True)
    row, col = 128, 143
    weights.zero_()
    weights[:, row] = torch.tensor([1.0, -0.5], device="cuda", dtype=torch.bfloat16)
    operation = torch.ops.sdc_cutlass.abft_augmented_mm
    clean, observed, _, predicted, _ = operation(a, b, weights, -1, -1, -1)
    d, corrupted, hit, fault_predicted, _ = operation(a, b, weights, row, col, 0)
    expected = torch.tensor(1.0, dtype=torch.float32)
    expected.view(torch.int32).bitwise_xor_(1)
    torch.testing.assert_close(d, clean, rtol=0, atol=0)
    torch.testing.assert_close(fault_predicted, predicted, rtol=0, atol=0)
    torch.testing.assert_close(
        corrupted[:, row // DATA_ROWS, col], weights[:, row].float() * expected.item(), rtol=0, atol=0,
    )
    assert hit.item() == 1
    assert corrupted[0, row // DATA_ROWS, col] != observed[0, row // DATA_ROWS, col]


@pytest.mark.parametrize("invalid", ["fp32", "fp16", "mixed", "A_strided", "B_strided"])
def test_augmented_rejects_unsupported_precision_and_layout(invalid):
    a, b, weights = _prepare(m=128)
    if invalid == "fp32":
        a, b = a.float(), b.float()
    elif invalid == "fp16":
        a, b = a.half(), b.half()
    elif invalid == "mixed":
        b = b.float()
    elif invalid == "A_strided":
        a = a.repeat_interleave(2, dim=1)[:, ::2]
    else:
        b = b.repeat_interleave(2, dim=1)[:, ::2]
    with pytest.raises(RuntimeError, match="BF16|dtype|row.major|RowMajor|packed|stride|layout"):
        torch.ops.sdc_cutlass.abft_augmented_mm(a, b, weights, -1, -1, -1)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "noncontiguous", "cpu"])
def test_augmented_rejects_invalid_weights(invalid):
    a, b, weights = _prepare()
    if invalid == "dtype":
        weights = weights.float()
    elif invalid == "shape":
        weights = weights[:, :-1].contiguous()
    elif invalid == "noncontiguous":
        weights = weights.T.contiguous().T
    else:
        weights = weights.cpu()
    with pytest.raises(RuntimeError, match="weights|BF16|CUDA device"):
        torch.ops.sdc_cutlass.abft_augmented_mm(a, b, weights, -1, -1, -1)


@pytest.mark.parametrize("fault", [(0, -1, -1), (-1, 0, 0), (129, 0, 0), (0, 144, 0), (0, 0, 32), (0, 0, -2)])
def test_augmented_rejects_invalid_fault(fault):
    a, b, weights = _prepare()
    with pytest.raises(RuntimeError, match="fault"):
        torch.ops.sdc_cutlass.abft_augmented_mm(a, b, weights, *fault)


def _integration_config():
    # These unit/dyadic fixtures have exactly representable BF16 encodings.
    # Zero tolerance ensures integration does not hide an incorrect projection.
    return {"method": "classic", "reference_mode": "augmented", "threshold_scale": "output",
            "diagnostics": True, "atol": 0.0, "rtol": 0.0}


def _assert_augmented_report(report):
    assert report.context["reference_mode"] == "augmented"
    assert report.context["checksum_input_precision"] == "bf16"
    assert report.context["threshold_scale"] == "output"
    assert not report.located.item()
    assert report.location.tolist() == [-1, -1]


def test_augmented_engine_recompute_reuses_clean_prepared_operands(monkeypatch):
    import abft
    from abft.operands import PreparedGemm

    backend.require_device(torch.device("cuda"))
    a = torch.ones(8, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(16, 16, device="cuda", dtype=torch.bfloat16)
    prepared = PreparedGemm.prepare(a, b, "bf16", "none")
    original_a, original_b = prepared.a.clone(), prepared.b.clone()
    pointers = (prepared.a.data_ptr(), prepared.b.data_ptr())
    baseline = prepared.raw_native()
    native = torch.ops.sdc_cutlass.abft_augmented_mm
    calls = []

    def record_native(left, right, weights, row, col, bit):
        calls.append(((left.data_ptr(), right.data_ptr()), (row, col, bit)))
        assert weights.dtype == torch.bfloat16
        return native(left, right, weights, row, col, bit)

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("execute_prepared must reuse existing operands during clean execution and retry")

    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_augmented_mm", record_native)
    monkeypatch.setattr(PreparedGemm, "prepare", unexpected_prepare)
    clean, report = abft.execute_prepared(prepared, abft_config=_integration_config())
    _assert_augmented_report(report)
    assert not report.detected.item() and not report.nonfinite.item()
    assert report.fault_hit.item() == 0
    assert torch.count_nonzero(report.residual).item() == 0
    assert torch.equal(clean.view(torch.int16), baseline.view(torch.int16))

    recovered, fault_report = abft.execute_prepared(
        prepared, abft_config={**_integration_config(), "recovery": "recompute"},
        fault_config={"location": "accumulator", "row": 7, "col": 15, "bit": 31},
    )
    _assert_augmented_report(fault_report)
    assert fault_report.fault_hit.item() == 1 and fault_report.detected.item()
    assert fault_report.recovered.item() and not fault_report.recheck_failed.item()
    assert torch.count_nonzero(fault_report.residual).item() == 2
    assert torch.count_nonzero(fault_report.retry_residual).item() == 0
    assert torch.equal(recovered.view(torch.int16), clean.view(torch.int16))
    assert calls == [(pointers, (-1, -1, -1)), (pointers, (7, 15, 31)), (pointers, (-1, -1, -1))]
    assert pointers == (prepared.a.data_ptr(), prepared.b.data_ptr())
    torch.testing.assert_close(prepared.a, original_a, rtol=0, atol=0)
    torch.testing.assert_close(prepared.b, original_b, rtol=0, atol=0)


def test_augmented_linear_forward_gradients_and_update_match_unprotected(monkeypatch):
    from torch import nn

    from cutlass_gemm.linear import install

    backend.require_device(torch.device("cuda"))
    original = nn.Linear(128, 16, device="cuda")
    baseline_original = nn.Linear(128, 16, device="cuda")
    with torch.no_grad():
        original.weight.fill_(0.125)
        original.bias.zero_()
    baseline_original.load_state_dict(original.state_dict())
    baseline = install(baseline_original, precision="bf16", scaling="none")
    protected = install(original, precision="bf16", scaling="none",
                        abft_config=_integration_config(), report_max_records=3)
    x = torch.ones(8, 128, device="cuda", requires_grad=True)
    baseline_x = x.detach().clone().requires_grad_()
    baseline_output = baseline(baseline_x)
    baseline_output.float().sum().backward()
    before = {name: parameter.detach().clone() for name, parameter in protected.named_parameters()}
    native = torch.ops.sdc_cutlass.abft_augmented_mm
    calls = []

    def record_native(left, right, weights, row, col, bit):
        calls.append((tuple(left.shape), tuple(right.shape)))
        assert left.dtype == right.dtype == weights.dtype == torch.bfloat16
        assert (row, col, bit) == (-1, -1, -1)
        return native(left, right, weights, row, col, bit)

    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_augmented_mm", record_native)
    output = protected(x)
    output.float().sum().backward()
    assert calls == [((8, 128), (128, 16)), ((8, 16), (16, 128)), ((16, 8), (8, 128))]
    assert torch.equal(output.view(torch.int16), baseline_output.view(torch.int16))
    torch.testing.assert_close(x.grad, baseline_x.grad, rtol=0, atol=0)
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad).item() > 0
    reports = list(protected.protection.reports)
    assert [report.context["role"] for report in reports] == ["forward", "dgrad", "wgrad"]
    for report in reports:
        _assert_augmented_report(report)
        assert not report.detected.item() and not report.nonfinite.item()
        assert report.fault_hit.item() == 0
        assert torch.count_nonzero(report.residual).item() == 0
    for name, parameter in protected.named_parameters():
        expected = dict(baseline.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
    # Dyadic learning rate keeps this comparison exact as well.
    torch.optim.SGD(protected.parameters(), lr=0.125).step()
    torch.optim.SGD(baseline.parameters(), lr=0.125).step()
    for name, parameter in protected.named_parameters():
        torch.testing.assert_close(parameter, dict(baseline.named_parameters())[name], rtol=0, atol=0)
        assert torch.isfinite(parameter).all()
        assert not torch.equal(parameter, before[name])
