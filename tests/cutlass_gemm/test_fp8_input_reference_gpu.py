"""SM120 mainloop input-reference evidence from the same prepared FP8 codes."""

import pytest
from test_abft_gpu import _prepare
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual FP8 mainloop input-reference execution requires SM120",
)


def _weights(m):
    # Dyadic, signed, unequal vectors indexed by GLOBAL row; tile 1 must not
    # accidentally repeat tile 0's weights. Exactly representable products
    # also make mainloop reduction errors distinguishable from quantization.
    rows = torch.arange(m, device="cuda", dtype=torch.int64)
    return torch.stack(((rows * 7 % 29 - 14).float() / 8,
                        (rows * 11 % 37 - 18).float() / 16))


def _references(args, weights):
    codes_a, codes_b, alpha = args
    # _prepare used global_scale=1. These are the actual native FP8 operands,
    # not a second quantization of the source tensors or the rounded output.
    a, b = codes_a.double(), codes_b.double().T
    expected, magnitudes = [], []
    for start in range(0, a.shape[0], 128):
        h = weights[:, start:start + 128].double()
        block = a[start:start + 128]
        expected.append(((h @ block) @ b) * alpha.double())
        magnitudes.append(((h.abs() @ block.abs()) @ b.abs()) * alpha.double().abs())
    return torch.stack(expected, dim=1), torch.stack(magnitudes, dim=1)


def _assert_reference(actual, expected, n):
    assert actual.dtype == torch.float32 and actual.device == expected.device
    assert actual.shape == (2, expected.shape[1], ((n + 63) // 64) * 64)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[:, :, :n].double(), expected, rtol=1e-5, atol=1e-3)
    assert torch.count_nonzero(actual[:, :, n:]).item() == 0


@pytest.mark.parametrize("shape", [
    (136, 144, 64), (136, 144, 128), (136, 144, 288),
    (136, 144, 512), (136, 144, 1024),
    # 512 CTA tiles exceed RTX 5090's SM count and require persistent reuse.
    (2048, 2048, 512),
])
def test_fp8_input_reference_matches_prepared_codes_and_preserves_output(shape):
    m, n, k = shape
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        base, epilogue_only, args, _, _ = _prepare("fp8", m=m, n=n, k=k)
        weights = _weights(m)
        assert args[2].item() == 0.25
        old_output = base(*args)
        old_d, old_observed, old_hit = epilogue_only(*args, weights, -1, -1, -1)
        output, observed, hit, predicted, magnitude = torch.ops.sdc_cutlass.abft_fp8_reference_mm(
            *args, weights, -1, -1, -1,
        )
        expected, expected_magnitude = _references(args, weights)
    stream.synchronize()
    assert output.dtype == torch.float32 and output.stride() == old_output.stride()
    assert torch.isfinite(output).all() and torch.isfinite(observed).all()
    torch.testing.assert_close(output, old_output, rtol=0, atol=0)
    torch.testing.assert_close(output, old_d, rtol=0, atol=0)
    torch.testing.assert_close(observed, old_observed, rtol=0, atol=0)
    assert hit.item() == old_hit.item() == 0
    _assert_reference(predicted, expected, n)
    _assert_reference(magnitude, expected_magnitude, n)
    # Compare the independent clean prediction with what the epilogue observed.
    torch.testing.assert_close(observed, predicted, rtol=1e-5, atol=1e-3)
    assert (magnitude >= 0).all()


def test_fp8_epilogue_fault_changes_output_and_observation_but_not_input_reference():
    m, n, k = 136, 144, 288
    _, epilogue_only, args, _, _ = _prepare("fp8", m=m, n=n, k=k)
    weights = _weights(m)
    operation = torch.ops.sdc_cutlass.abft_fp8_reference_mm
    clean, observed, hit, predicted, magnitude = operation(*args, weights, -1, -1, -1)
    corrupted, fault_observed, fault_hit, fault_predicted, fault_magnitude = operation(
        *args, weights, m - 1, n - 1, 31,
    )
    old_corrupted, old_observed, old_hit = epilogue_only(*args, weights, m - 1, n - 1, 31)
    expected = clean.clone()
    expected[m - 1, n - 1] = -expected[m - 1, n - 1]
    assert clean[m - 1, n - 1] != 0
    assert hit.item() == 0 and fault_hit.item() == old_hit.item() == 1
    torch.testing.assert_close(corrupted, expected, rtol=0, atol=0)
    torch.testing.assert_close(corrupted, old_corrupted, rtol=0, atol=0)
    torch.testing.assert_close(fault_observed, old_observed, rtol=0, atol=0)
    torch.testing.assert_close(fault_predicted, predicted, rtol=0, atol=0)
    torch.testing.assert_close(fault_magnitude, magnitude, rtol=0, atol=0)
    expected_delta = torch.zeros_like(observed)
    expected_delta[:, (m - 1) // 128, n - 1] = weights[:, m - 1] * (-2 * clean[m - 1, n - 1])
    torch.testing.assert_close(fault_observed - observed, expected_delta, rtol=0, atol=1e-3)
    assert torch.count_nonzero(expected_delta).item() == 2
    assert torch.isfinite(fault_observed).all() and torch.isfinite(corrupted).all()
    expected_reference, expected_magnitude = _references(args, weights)
    _assert_reference(fault_predicted, expected_reference, n)
    _assert_reference(fault_magnitude, expected_magnitude, n)
