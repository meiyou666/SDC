"""Real input-fused references through the engine and Linear training bridge."""

import pytest
import torch
from torch import nn

import abft
from abft.operands import PreparedGemm
from cutlass_gemm import backend
from cutlass_gemm.linear import install


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual input-fused engine and training execution requires SM120",
)
SCALING = {"fp32": "none", "bf16": "none", "fp8": "tensorwise", "nvfp4": "block16"}
CONFIG = {"method": "classic", "reference_mode": "input_fused", "diagnostics": True,
          "rtol": 1e-5, "atol": 1e-3}


def _assert_clean_report(report):
    assert report.context["reference_mode"] == "input_fused"
    assert report.context["protection_scope"] == "computation_after_input_load"
    assert not report.detected.item() and not report.nonfinite.item()
    assert report.fault_hit.item() == 0
    assert torch.isfinite(report.residual).all() and torch.isfinite(report.threshold).all()
    assert (report.residual.abs() <= report.threshold).all()


@pytest.mark.parametrize("precision", SCALING)
def test_execute_prepared_native_input_reference_and_recovery(precision, monkeypatch):
    backend.require_device(torch.device("cuda"))
    a = torch.ones(136, 288, device="cuda")
    b = torch.ones(288, 144, device="cuda")
    prepared = PreparedGemm.prepare(a, b, precision, SCALING[precision], 1.0)
    # The raw, protected and recovery calls must consume the same prepared codes.
    original_a, original_b = prepared.a.clone(), prepared.b.clone()
    pointers = (prepared.a.data_ptr(), prepared.b.data_ptr())
    baseline = prepared.raw_native()

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("execute_prepared must not prepare or requantize operands")

    monkeypatch.setattr(PreparedGemm, "prepare", unexpected_prepare)
    clean, report = abft.execute_prepared(prepared, abft_config=CONFIG)
    _assert_clean_report(report)
    assert clean.dtype == baseline.dtype and clean.stride() == baseline.stride()
    torch.testing.assert_close(clean, baseline, rtol=0, atol=0)
    recovered, fault_report = abft.execute_prepared(
        prepared, abft_config={**CONFIG, "recovery": "recompute"},
        fault_config={"location": "accumulator", "row": 135, "col": 143, "bit": 31},
    )
    assert fault_report.context["reference_mode"] == "input_fused"
    assert fault_report.fault_hit.item() == 1 and fault_report.detected.item()
    assert fault_report.recovered.item() and not fault_report.recheck_failed.item()
    assert torch.isfinite(fault_report.retry_residual).all()
    assert (fault_report.retry_residual.abs() <= fault_report.threshold).all()
    assert recovered.dtype == clean.dtype and recovered.stride() == clean.stride()
    torch.testing.assert_close(recovered, clean, rtol=0, atol=0)
    assert pointers == (prepared.a.data_ptr(), prepared.b.data_ptr())
    torch.testing.assert_close(prepared.a, original_a, rtol=0, atol=0)
    torch.testing.assert_close(prepared.b, original_b, rtol=0, atol=0)


@pytest.mark.parametrize("precision", SCALING)
def test_linear_forward_dgrad_wgrad_use_real_input_fused_native(precision, monkeypatch):
    backend.require_device(torch.device("cuda"))
    original = nn.Linear(128, 256, device="cuda")
    with torch.no_grad():
        original.weight.fill_(0.125)
        original.bias.zero_()
    layer = install(original, precision=precision, scaling=SCALING[precision], global_scale=1.0,
                    abft_config=CONFIG, report_max_records=3)
    calls, native_calls = [], []
    real_execute = abft.execute
    op_name = {"fp32": "abft_reference_mm", "bf16": "abft_reference_mm",
               "fp8": "abft_fp8_reference_mm", "nvfp4": "abft_nvfp4_reference_mm"}[precision]
    real_native = getattr(torch.ops.sdc_cutlass, op_name)

    def observed_native(*args, **kwargs):
        native_calls.append(tuple(args[0].shape))
        return real_native(*args, **kwargs)

    def observed_execute(*args, **kwargs):
        output, report = real_execute(*args, **kwargs)
        calls.append(report.context["role"])
        _assert_clean_report(report)
        return output, report

    monkeypatch.setattr(torch.ops.sdc_cutlass, op_name, observed_native)
    monkeypatch.setattr(abft, "execute", observed_execute)
    x = torch.ones(2, 64, 128, device="cuda", requires_grad=True)
    before = {name: p.detach().clone() for name, p in layer.named_parameters()}
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.001)
    output = layer(x)
    assert output.shape == (2, 64, 256) and torch.isfinite(output).all()
    output.float().sum().backward()
    assert calls == ["forward", "dgrad", "wgrad"]
    assert len(native_calls) == 3
    assert len(layer.protection.reports) == 3
    assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)
    optimizer.step()
    for name, parameter in layer.named_parameters():
        assert torch.isfinite(parameter).all()
        assert not torch.equal(parameter, before[name])
