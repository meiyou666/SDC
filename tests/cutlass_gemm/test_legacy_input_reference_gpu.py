"""Same shared operands generate legacy predictions across pipeline wraps/layouts."""

import pytest
from test_abft_gpu import _prepare
from test_fp8_input_reference_gpu import _weights
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual legacy mainloop input-reference execution requires SM120",
)


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
@pytest.mark.parametrize("layouts", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("k", [16, 64, 80, 192, 448])
def test_legacy_reference_preserves_gemm_and_matches_operands(precision, layouts, k):
    m, n = 136, 144
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        base, old, args, _, _ = _prepare(precision, m=m, n=n, k=k, layouts=layouts)
        weights = _weights(m)
        baseline = base(*args)
        old_d, old_observed, old_hit = old(*args, weights, -1, -1, -1)
        d, observed, hit, predicted, magnitude = torch.ops.sdc_cutlass.abft_reference_mm(
            *args, weights, -1, -1, -1,
        )
        a, b = (operand.double() for operand in args)
        expected, expected_magnitude = [], []
        for start in range(0, m, 128):
            w, block = weights[:, start:start + 128].double(), a[start:start + 128]
            expected.append((w @ block) @ b)
            expected_magnitude.append((w.abs() @ block.abs()) @ b.abs())
        expected = torch.stack(expected, dim=1)
        expected_magnitude = torch.stack(expected_magnitude, dim=1)
    stream.synchronize()
    torch.testing.assert_close(d, baseline, atol=0, rtol=0)
    torch.testing.assert_close(d, old_d, atol=0, rtol=0)
    torch.testing.assert_close(observed, old_observed, atol=0, rtol=0)
    assert hit.item() == old_hit.item() == 0
    for actual, reference in [(predicted, expected), (magnitude, expected_magnitude)]:
        assert actual.shape == (2, 2, 256) and actual.dtype == torch.float32
        torch.testing.assert_close(actual[:, :, :n].double(), reference, atol=1e-3, rtol=1e-5)
        assert torch.count_nonzero(actual[:, :, n:]).item() == 0
    torch.testing.assert_close(observed, predicted, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_legacy_accumulator_fault_preserves_prediction(precision):
    _, old, args, _, _ = _prepare(precision, m=136, n=144, k=80, unit=True)
    weights = _weights(136)
    operation = torch.ops.sdc_cutlass.abft_reference_mm
    clean, obs, hit, pred, mag = operation(*args, weights, -1, -1, -1)
    fault, fault_obs, fault_hit, fault_pred, fault_mag = operation(*args, weights, 135, 143, 31)
    old_fault, old_obs, old_hit = old(*args, weights, 135, 143, 31)
    expected = clean.clone()
    expected[135, 143] = -expected[135, 143]
    torch.testing.assert_close(fault, expected, atol=0, rtol=0)
    torch.testing.assert_close(fault, old_fault, atol=0, rtol=0)
    torch.testing.assert_close(fault_obs, old_obs, atol=0, rtol=0)
    torch.testing.assert_close(pred, fault_pred, atol=0, rtol=0)
    torch.testing.assert_close(mag, fault_mag, atol=0, rtol=0)
    assert hit.item() == 0 and fault_hit.item() == old_hit.item() == 1
    delta = torch.zeros_like(obs)
    delta[:, 1, 143] = -2 * weights[:, 135]
    torch.testing.assert_close(fault_obs - obs, delta, atol=0, rtol=0)


@pytest.mark.parametrize("precision,k", [("bf16", 80), ("fp32", 33)])
def test_reference_visits_residue_k_tile_before_full_tiles(precision, k):
    from cutlass_gemm import backend

    backend.require_device(torch.device("cuda"))
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    m, n = 128, 128
    a = torch.arange(1, k + 1, device="cuda", dtype=torch.float32).expand(m, -1).contiguous().to(dtype)
    b = torch.ones(k, n, device="cuda", dtype=dtype)
    weights = torch.zeros(2, m, device="cuda")
    weights[0, 0], weights[1, 1] = 1, -2
    output, observed, _, predicted, magnitude = torch.ops.sdc_cutlass.abft_reference_mm(
        a, b, weights, -1, -1, -1,
    )
    total = k * (k + 1) / 2
    torch.testing.assert_close(output, torch.full_like(output, total), rtol=0, atol=0)
    expected = torch.tensor([total, -2 * total], device="cuda")[:, None].expand(-1, n)
    torch.testing.assert_close(predicted[:, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(observed[:, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(magnitude[:, 0], expected.abs(), rtol=0, atol=0)
