"""Real NVFP4 input-reference checks with heterogeneous block scale factors."""

import pytest
import torch

from abft.operands import PreparedGemm
from cutlass_gemm import backend
from cutlass_gemm.quantization import dequantize


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual NVFP4 mainloop input-reference execution requires SM120",
)


def _prepare(m, n, k):
    backend.require_device(torch.device("cuda"))

    def values(rows, phase):
        row = torch.arange(rows, device="cuda")[:, None]
        col = torch.arange(k, device="cuda")[None, :]
        exponent = ((row * 3 + col // 16 + phase) % 11 - 5).float()
        amplitude = torch.pow(2.0, exponent)
        pattern = ((row + col * 7 + phase) % 13 - 6).float()
        return (pattern * amplitude).contiguous()

    # Quantize only once. All native calls and references reuse these buffers.
    prepared = PreparedGemm.prepare(values(m, 0), values(n, 3).T, "nvfp4", "block16", "amax")
    assert prepared.qa.scales.unique().numel() > 4
    assert prepared.qb.scales.unique().numel() > 4
    rows = torch.arange(m, device="cuda")
    weights = torch.stack(((rows * 7 % 29 - 14).float() / 8,
                           (rows * 11 % 37 - 18).float() / 16))
    return prepared, weights


def _reference(prepared, weights):
    # Dequantization uses the exact codes, per-block scales and global scales
    # consumed by the native GEMM. No reference is derived from BF16 output.
    a, b = dequantize(prepared.qa).double(), dequantize(prepared.qb).double().T
    predicted, magnitude = [], []
    for start in range(0, prepared.m, 128):
        h, block = weights[:, start:start + 128].double(), a[start:start + 128]
        predicted.append((h @ block) @ b)
        magnitude.append((h.abs() @ block.abs()) @ b.abs())
    return torch.stack(predicted, 1), torch.stack(magnitude, 1)


def _check_reference(actual, expected, magnitude, n):
    assert actual.dtype == torch.float32 and actual.device == expected.device
    assert actual.shape == (2, expected.shape[1], ((n + 127) // 128) * 128)
    assert torch.isfinite(actual).all()
    # Cancellation can make a relative-to-prediction bound misleading. Bound
    # FP32 accumulation and scale arithmetic by the absolute-product reference.
    error = (actual[:, :, :n].double() - expected).abs()
    assert (error <= 5e-5 * magnitude + 1e-3).all()
    assert torch.count_nonzero(actual[:, :, n:]).item() == 0


@pytest.mark.parametrize("shape", [
    (136, 144, 32), (136, 144, 128), (136, 144, 288),
    (136, 144, 512), (136, 144, 1024), (2048, 2048, 512),
])
def test_nvfp4_varying_scales_input_reference_preserves_native_outputs(shape):
    prepared, weights = _prepare(*shape)
    raw = prepared.raw_native()
    old_output, old_observed, old_hit = prepared.collected_native(weights)
    output, observed, hit, predicted, magnitude = prepared.collected_reference_native(weights)
    expected, expected_magnitude = _reference(prepared, weights)
    assert output.dtype == torch.bfloat16 and output.stride() == raw.stride()
    torch.testing.assert_close(output, raw, rtol=0, atol=0)
    torch.testing.assert_close(output, old_output, rtol=0, atol=0)
    torch.testing.assert_close(observed, old_observed, rtol=0, atol=0)
    assert hit.item() == old_hit.item() == 0
    assert torch.isfinite(output).all() and torch.isfinite(observed).all()
    _check_reference(predicted, expected, expected_magnitude, prepared.n)
    _check_reference(magnitude, expected_magnitude, expected_magnitude, prepared.n)
    _check_reference(observed, expected, expected_magnitude, prepared.n)
    assert (magnitude >= 0).all()


def test_nvfp4_epilogue_sign_fault_leaves_varying_scale_input_reference_clean():
    prepared, weights = _prepare(136, 144, 288)
    clean, observed, hit, predicted, magnitude = prepared.collected_reference_native(weights)
    fault = (135, 143, 31)
    output, fault_observed, fault_hit, fault_predicted, fault_magnitude = prepared.collected_reference_native(
        weights, fault,
    )
    old_output, old_observed, old_hit = prepared.collected_native(weights, fault)
    assert hit.item() == 0 and fault_hit.item() == old_hit.item() == 1
    expected_output = clean.clone()
    assert expected_output[135, 143] != 0
    expected_output[135, 143] = -expected_output[135, 143]
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(output, old_output, rtol=0, atol=0)
    torch.testing.assert_close(fault_observed, old_observed, rtol=0, atol=0)
    torch.testing.assert_close(fault_predicted, predicted, rtol=0, atol=0)
    torch.testing.assert_close(fault_magnitude, magnitude, rtol=0, atol=0)
    # Only the last column of the last M tile can change; don't mistake the
    # post-cast BF16 cell for an exact reference to its FP32 projection delta.
    assert torch.all(fault_observed[:, 1, 143] != observed[:, 1, 143])
    unchanged = fault_observed.clone()
    unchanged[:, 1, 143] = observed[:, 1, 143]
    torch.testing.assert_close(unchanged, observed, rtol=0, atol=0)
    expected, expected_magnitude = _reference(prepared, weights)
    _check_reference(fault_predicted, expected, expected_magnitude, prepared.n)
    _check_reference(fault_magnitude, expected_magnitude, expected_magnitude, prepared.n)
