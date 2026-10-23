"""Native orchestration contracts and actual GPU stream/recovery integration."""
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from abft import AbftConfig, execute_prepared, native
from abft.engine import AbftError
from abft.operands import PreparedGemm
from cutlass_gemm import backend


GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual native ABFT integration requires the configured SM120 GPU",
)


def _config(**changes):
    values = dict(method="classic", reference_mode="augmented", threshold_scale="output",
                  implementation="native", atol=0.0, rtol=0.0, on_error="report")
    return AbftConfig(**(values | changes))


def _forbidden(*args, **kwargs):
    pytest.fail("Unexpected host tensor read, synchronization, or fallback")


def _stub_native(monkeypatch, detections=(False,)):
    """Fake device boundaries to test orchestration independently of CUDA math."""
    prepared = PreparedGemm.prepare(torch.ones(3, 16), torch.ones(16, 16), "bf16", "none")
    weights = torch.ones(2, 3, dtype=torch.bfloat16)
    outputs = [torch.full((3, 16), float(i + 1), dtype=torch.bfloat16)
               for i in range(len(detections))]
    partials = [torch.zeros(3, 2, 1, 128) for _ in detections]
    tile_flags = [torch.zeros(1, 1, dtype=torch.int32) for _ in detections]
    references = [torch.full((2, 2, 16), float(i + 1)) for i in range(len(detections))]
    flags = [torch.tensor([detected, False, False, False, False]) for detected in detections]
    calls, finalizer_calls = [], []

    def compact(a, b, passed_weights, row, col, bit):
        assert a is prepared.a and b is prepared.b and passed_weights is weights
        index = len(calls)
        calls.append((row, col, bit))
        return outputs[index], partials[index], tile_flags[index]

    def finalize(passed_partials, passed_flags, n, atol, rtol, reference, diagnostics):
        index = len(finalizer_calls)
        assert passed_partials is partials[index] and passed_flags is tile_flags[index]
        assert n == prepared.n
        finalizer_calls.append(reference)
        diagnostic_shape = (2, n) if diagnostics else (0,)
        return (references[index] if reference is None else reference,
                torch.full(diagnostic_shape, float(index)), torch.zeros(diagnostic_shape),
                flags[index], torch.tensor([int(calls[index][0] >= 0)], dtype=torch.int32),
                torch.full((2,), -1, dtype=torch.int64))

    monkeypatch.setattr(backend, "require_device", lambda device: None)
    monkeypatch.setattr(native, "classic_weights", lambda m, device: weights)
    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_augmented_compact", compact, raising=False)
    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_finalize", finalize, raising=False)
    return SimpleNamespace(prepared=prepared, calls=calls, finalizer_calls=finalizer_calls,
                           references=references, flags=flags, outputs=outputs)


@pytest.mark.parametrize("changes", [{"implementation": "typo"},
                                     {"method": "none", "reference_mode": "separate", "threshold_scale": "input"}])
def test_native_configuration_rejects_invalid_generic_contract(changes):
    with pytest.raises(ValueError, match="implementation"):
        _config(**changes)


@pytest.mark.parametrize("changes", [
    {"method": "other"}, {"parameters": {"extra": 1}},
    {"reference_mode": "separate", "threshold_scale": "input"},
])
def test_native_method_declares_and_validates_execution_contract(monkeypatch, changes):
    stub = _stub_native(monkeypatch)
    cfg = _config(**changes)  # Configuration parsing does not import a method or Torch.
    with pytest.raises(ValueError):
        execute_prepared(stub.prepared, cfg)
    assert not stub.calls


@pytest.mark.parametrize("method", ["classic", "sum"])
def test_native_input_fused_configuration_is_allowed(method):
    cfg = _config(method=method, reference_mode="input_fused", threshold_scale="input")
    assert cfg.method == method and cfg.implementation == "native"


@pytest.mark.parametrize("override", ["raw_fn", "reference_mm", "collector_fn", "reference_collector_fn"])
def test_native_does_not_silently_ignore_reference_overrides(monkeypatch, override):
    stub = _stub_native(monkeypatch)
    monkeypatch.setattr(native, "execute_augmented", _forbidden)
    with pytest.raises(ValueError, match="does not accept"):
        execute_prepared(stub.prepared, _config(), **{override: _forbidden})


@pytest.mark.parametrize("precision", ["fp32", "fp8", "nvfp4"])
def test_native_rejects_other_precisions_before_execution(monkeypatch, precision):
    stub = _stub_native(monkeypatch)
    stub.prepared.precision = precision
    monkeypatch.setattr(native, "execute_augmented", _forbidden)
    with pytest.raises(ValueError, match="requires BF16"):
        execute_prepared(stub.prepared, _config())


@pytest.mark.parametrize("location", ["a", "b"])
def test_native_rejects_input_memory_faults(monkeypatch, location):
    stub = _stub_native(monkeypatch)
    monkeypatch.setattr(native, "execute_augmented", _forbidden)
    with pytest.raises(ValueError, match="input memory faults"):
        execute_prepared(stub.prepared, _config(), {"location": location})


@pytest.mark.parametrize("context,selected", [
    ({"layer": "target", "role": "forward"}, True),
    ({"layer": "other", "role": "forward"}, False),
    ({"layer": "target", "role": "dgrad"}, False),
    ({"layer": "target", "role": "forward", "call_index": 2}, False),
    ({"layer": "target", "role": "forward", "retry": True}, False),
])
def test_native_fault_selector_preserves_context(monkeypatch, context, selected):
    stub = _stub_native(monkeypatch)
    fault = {"location": "accumulator", "row": 1, "col": 2, "bit": 31,
             "layer": "target", "role": "forward"}
    _, report = execute_prepared(stub.prepared, _config(), fault, context)
    assert stub.calls == [(1, 2, 31) if selected else (-1, -1, -1)]
    assert report.context["fault_selected"] is selected
    assert report.context["implementation"] == "native"
    assert report.context["reference_operand_reuse"] == "main_B_register_fragments"


@pytest.mark.parametrize("fault", [{"row": 3}, {"col": 16}, {"bit": 32}])
def test_native_rejects_out_of_bounds_selected_fault(monkeypatch, fault):
    stub = _stub_native(monkeypatch)
    with pytest.raises(ValueError, match="outside FP32 output"):
        execute_prepared(stub.prepared, _config(), {"location": "accumulator", **fault})
    assert not stub.calls


@pytest.mark.parametrize("diagnostics", [False, True])
def test_native_report_policy_performs_no_host_tensor_reads(monkeypatch, diagnostics):
    stub = _stub_native(monkeypatch)
    with monkeypatch.context() as guarded:
        for name in ("item", "__bool__", "cpu", "numpy", "tolist"):
            guarded.setattr(torch.Tensor, name, _forbidden)
        guarded.setattr(torch.cuda, "synchronize", _forbidden)
        _, report = execute_prepared(stub.prepared, _config(diagnostics=diagnostics))
    assert report.execution_path == "augmented_native"
    assert ("residual" in report.tensors) is diagnostics
    assert ("threshold" in report.tensors) is diagnostics
    assert "retry_residual" not in report.tensors
    assert report.detected.shape == torch.Size([])
    assert not report.to_record()["detected"]  # Serialization is an explicit read boundary.


@pytest.mark.parametrize("failed_recheck", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_native_recompute_reuses_first_reference_and_clean_operands(monkeypatch, failed_recheck, diagnostics):
    stub = _stub_native(monkeypatch, (True, failed_recheck))
    output, report = execute_prepared(
        stub.prepared, _config(recovery="recompute", diagnostics=diagnostics),
        {"location": "accumulator", "row": 1, "col": 2, "bit": 31},
    )
    assert stub.calls == [(1, 2, 31), (-1, -1, -1)]
    assert stub.finalizer_calls[0] is None
    assert stub.finalizer_calls[1] is stub.references[0]
    assert output is stub.outputs[0 if failed_recheck else 1]
    assert bool(report.detected) and not bool(report.located)
    assert bool(report.recovered) is not failed_recheck
    assert bool(report.recheck_failed) is failed_recheck
    assert report.fault_hit.tolist() == [1]
    assert ("retry_residual" in report.tensors) is diagnostics


@pytest.mark.parametrize("recovery,failed", [("none", True), ("recompute", True), ("recompute", False)])
def test_native_raise_policy_raises_only_unrecovered_anomaly(monkeypatch, recovery, failed):
    stub = _stub_native(monkeypatch, (True, failed))
    cfg = _config(recovery=recovery, on_error="raise")
    if failed:
        with pytest.raises(AbftError) as raised:
            execute_prepared(stub.prepared, cfg)
        report = raised.value.report
        assert report.detected and not report.recovered
    else:
        output, report = execute_prepared(stub.prepared, cfg)
        assert output is stub.outputs[1] and report.recovered


def test_native_retained_reports_do_not_share_mutable_state(monkeypatch):
    stub = _stub_native(monkeypatch, (True, False))
    _, first = execute_prepared(stub.prepared, _config(diagnostics=True))
    saved = {key: value.clone() for key, value in first.tensors.items()}
    _, second = execute_prepared(stub.prepared, _config(diagnostics=True))
    assert first.detected and not second.detected
    assert first.residual.data_ptr() != second.residual.data_ptr()
    for key, value in saved.items():
        torch.testing.assert_close(first.tensors[key], value, rtol=0, atol=0)


def test_native_profiles_three_normal_phases_without_python_merges(monkeypatch):
    stub = _stub_native(monkeypatch)
    seen = []

    @contextmanager
    def scope(name):
        seen.append(name)
        yield

    execute_prepared(stub.prepared, _config(), profile_scope=scope)
    assert seen == ["abft.reference", "abft.execute", "abft.finalize"]


def _gpu_prepared():
    backend.require_device(torch.device("cuda"))
    # C=1 and the two per-tile encodings are exactly representable in BF16.
    a = torch.zeros(129, 80, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(80, 144, device="cuda", dtype=torch.bfloat16)
    a[:, 0] = 1
    b[0] = 1
    return PreparedGemm.prepare(a, b, "bf16", "none")


@GPU
@pytest.mark.parametrize("mode", ["clean", "fault", "recover"])
def test_native_gpu_pipeline_output_detection_and_recovery(mode):
    prepared = _gpu_prepared()
    fault = None if mode == "clean" else {"location": "accumulator", "row": 128, "col": 143, "bit": 31}
    cfg = _config(diagnostics=True, recovery="recompute" if mode == "recover" else "none")
    baseline = prepared.raw_native()
    output, report = execute_prepared(prepared, cfg, fault)
    expected = baseline.clone()
    if mode == "fault":
        expected[128, 143] = -1
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert bool(report.detected) is (mode != "clean")
    assert bool(report.recovered) is (mode == "recover")
    assert not report.nonfinite and not report.located and not report.recheck_failed
    assert report.fault_hit.item() == (mode != "clean")
    assert report.residual.device == output.device


@GPU
def test_native_gpu_report_path_does_not_read_device_scalars(monkeypatch):
    prepared = _gpu_prepared()
    native.classic_weights(prepared.m, prepared.a.device)
    with monkeypatch.context() as guarded:
        for name in ("item", "__bool__", "cpu", "numpy", "tolist"):
            guarded.setattr(torch.Tensor, name, _forbidden)
        guarded.setattr(torch.cuda, "synchronize", _forbidden)
        output, report = execute_prepared(prepared, _config(diagnostics=True))
    assert output.is_cuda and all(value.is_cuda for value in report.tensors.values())
    assert not report.detected


@GPU
def test_classic_weight_cache_cross_stream_ordering_and_eviction(monkeypatch):
    monkeypatch.setattr(native, "_weights", OrderedDict())
    monkeypatch.setattr(native, "_MAX_WEIGHT_ENTRIES", 1)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(2_000_000)
        first = native.classic_weights(257, torch.device("cuda"))
    with torch.cuda.stream(consumer):
        second = native.classic_weights(257, torch.device("cuda"))
        assert first.data_ptr() == second.data_ptr()
        torch.cuda._sleep(2_000_000)
        snapshot = second.float().clone()
    del first, second
    with torch.cuda.stream(producer):
        native.classic_weights(258, torch.device("cuda"))  # Evict while the consumer may still run.
        churn = [torch.empty(2, 257, device="cuda", dtype=torch.bfloat16).fill_(-8) for _ in range(16)]
    consumer.synchronize()
    expected = torch.stack((torch.ones(257), torch.arange(1, 258).float())).bfloat16().float()
    torch.testing.assert_close(snapshot.cpu(), expected, rtol=0, atol=0)
    producer.synchronize()
    assert len(native._weights) == 1
    assert churn


@GPU
def test_classic_weight_cache_respects_byte_budget(monkeypatch):
    monkeypatch.setattr(native, "_weights", OrderedDict())
    monkeypatch.setattr(native, "_MAX_WEIGHT_BYTES", 1)
    first = native.classic_weights(129, torch.device("cuda"))
    second = native.classic_weights(129, torch.device("cuda"))
    assert first.data_ptr() != second.data_ptr() and not native._weights
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@GPU
def test_native_gpu_concurrent_calls_keep_reports_independent():
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    cfg = _config(diagnostics=True)
    with torch.cuda.stream(producer):
        first_output, first = execute_prepared(
            _gpu_prepared(), cfg, {"location": "accumulator", "row": 128, "col": 143, "bit": 31},
        )
    with torch.cuda.stream(consumer):
        second_output, second = execute_prepared(_gpu_prepared(), cfg)
    producer.synchronize()
    consumer.synchronize()
    assert first_output[128, 143].item() == -1 and second_output[128, 143].item() == 1
    assert first.detected and not second.detected
    assert first.residual.data_ptr() != second.residual.data_ptr()
    assert first.detected.untyped_storage().data_ptr() != second.detected.untyped_storage().data_ptr()
    assert first.residual[0, 143].item() == -2 and second.residual[0, 143].item() == 0


@GPU
@pytest.mark.parametrize("corrupt_retry", [False, True])
def test_native_gpu_retry_never_replaces_first_prediction_or_magnitude(monkeypatch, corrupt_retry):
    prepared = _gpu_prepared()
    compact = torch.ops.sdc_cutlass.abft_augmented_compact
    calls = []

    def wrapped(*args):
        output, partials, tile_flags = compact(*args)
        calls.append(args[-3:])
        if len(calls) == 2:
            if corrupt_retry:
                output = output + 200
                partials[0] += 200
                partials[1].copy_(partials[0])
                partials[2].fill_(1e20)
            else:
                partials[1].fill_(float("nan"))
                partials[2].fill_(float("inf"))
        return output, partials, tile_flags

    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_augmented_compact", wrapped)
    output, report = execute_prepared(
        prepared, _config(diagnostics=True, recovery="recompute", rtol=1e-4),
        {"location": "accumulator", "row": 128, "col": 143, "bit": 31},
    )
    assert calls == [(128, 143, 31), (-1, -1, -1)]
    assert report.detected and not report.located
    assert bool(report.recovered) is not corrupt_retry
    assert bool(report.recheck_failed) is corrupt_retry
    assert output[128, 143].item() == (-1 if corrupt_retry else 1)
    assert torch.isfinite(report.retry_residual).all()


@GPU
def test_native_gpu_linear_forward_dgrad_wgrad_match_unprotected(monkeypatch):
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
    cfg = _config(diagnostics=False, recovery="none")
    protected = install(original, precision="bf16", scaling="none",
                        abft_config=cfg, report_max_records=3)
    x = torch.ones(8, 128, device="cuda", requires_grad=True)
    baseline_x = x.detach().clone().requires_grad_()
    baseline_output = baseline(baseline_x)
    baseline_output.float().sum().backward()
    compact = torch.ops.sdc_cutlass.abft_augmented_compact
    calls = []

    def record_compact(left, right, weights, row, col, bit):
        calls.append((tuple(left.shape), tuple(right.shape)))
        assert left.dtype == right.dtype == weights.dtype == torch.bfloat16
        assert (row, col, bit) == (-1, -1, -1)
        return compact(left, right, weights, row, col, bit)

    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_augmented_compact", record_compact)
    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_augmented_mm", _forbidden)
    output = protected(x)
    output.float().sum().backward()
    assert calls == [((8, 128), (128, 16)), ((8, 16), (16, 128)), ((16, 8), (8, 128))]
    assert torch.equal(output.view(torch.int16), baseline_output.view(torch.int16))
    torch.testing.assert_close(x.grad, baseline_x.grad, rtol=0, atol=0)
    assert x.grad.dtype == torch.float32 and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad).item() > 0
    for name, parameter in protected.named_parameters():
        expected = dict(baseline.named_parameters())[name]
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad).item() > 0
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
    reports = list(protected.protection.reports)
    assert [report.context["role"] for report in reports] == ["forward", "dgrad", "wgrad"]
    assert [report.context["call_index"] for report in reports] == [1, 2, 3]
    for report in reports:
        assert report.execution_path == "augmented_native"
        assert report.context["implementation"] == "native"
        assert not report.detected and not report.nonfinite and not report.recovered
        assert report.fault_hit.item() == 0
        assert not {"residual", "threshold", "retry_residual"} & report.tensors.keys()
