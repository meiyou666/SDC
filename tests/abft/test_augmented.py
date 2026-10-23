"""CPU contracts for augmented BF16 evidence; these do not validate CUDA math."""
import pytest
import torch

from abft import AbftConfig, execute_prepared
from abft.engine import AbftError
from abft.methods import AbftMethod, Classic
from abft.operands import PreparedGemm


def config(**changes):
    return AbftConfig(method="classic", reference_mode="augmented", threshold_scale="output",
                      diagnostics=True, **changes)


def operands(m=3):
    return PreparedGemm.prepare(torch.ones(m, 16), torch.ones(16, 2), "bf16", "none")


def collect(prepared, weights, fault):
    """Fake pre-cast evidence with BF16 encoding and an optional accumulator flip."""
    assert weights.dtype == torch.bfloat16
    a, b, h = prepared.a.float(), prepared.b.float(), weights.float()
    accumulator = a @ b
    predicted = ((h @ a).to(torch.bfloat16).float() @ b).unsqueeze(1)
    row, col, bit = fault
    if row >= 0:
        mask = 1 << bit if bit < 31 else -(1 << bit)
        accumulator.view(torch.int32)[row, col] ^= mask
    observed = (h @ accumulator).unsqueeze(1)
    magnitude = (h.abs() @ accumulator.abs()).unsqueeze(1)
    return (accumulator.to(torch.bfloat16), observed,
            torch.tensor([int(row >= 0)], dtype=torch.int32), predicted, magnitude)


def forbidden(*args, **kwargs):
    pytest.fail("Unexpected separate reference or non-augmented collector")


def test_augmented_requires_explicit_output_threshold_scale():
    with pytest.raises(ValueError, match="threshold_scale=output"):
        AbftConfig(method="classic", reference_mode="augmented")
    assert config().threshold_scale == "output"


@pytest.mark.parametrize("mode", ["separate", "input_fused"])
def test_old_reference_modes_reject_output_threshold_scale(mode):
    with pytest.raises(ValueError, match="threshold_scale=input"):
        AbftConfig(method="classic", reference_mode=mode, threshold_scale="output")


def test_augmented_requires_an_enabled_method():
    with pytest.raises(ValueError, match="requires an ABFT method"):
        AbftConfig(reference_mode="augmented", threshold_scale="output")


def test_defaults_keep_existing_fp32_reference_contract():
    default = AbftConfig(method="classic")
    assert default.reference_mode == "separate" and default.threshold_scale == "input"
    fused = AbftConfig(method="classic", reference_mode="input_fused")
    state = Classic().prepare_fused(operands(), fused)
    assert state["weights"].dtype == torch.float32
    assert not state.get("augmented", False)


def test_encoding_weights_use_actual_bf16_row_numbers():
    prepared = operands(m=300)
    state = Classic().prepare_augmented(prepared, config())
    assert state["augmented"] is True
    assert state["weights"].dtype == torch.bfloat16
    assert state["weights"].device == prepared.a.device
    assert state["weights"].is_contiguous()
    torch.testing.assert_close(state["weights"][0], torch.ones(300, dtype=torch.bfloat16))
    torch.testing.assert_close(state["weights"][1], torch.arange(1, 301).to(torch.bfloat16))
    # Distinct FP32 row numbers need not remain distinct after BF16 encoding.
    assert state["weights"][1, 255] == state["weights"][1, 256]
    assert "expected" not in state and "magnitude" not in state


@pytest.mark.parametrize("precision", ["fp32", "fp8", "nvfp4"])
def test_augmented_rejects_other_precisions_before_native_execution(monkeypatch, precision):
    prepared = operands()
    prepared.precision = precision
    monkeypatch.setattr(prepared, "augmented_native", forbidden)
    with pytest.raises(ValueError, match="requires BF16"):
        execute_prepared(prepared, config(), reference_collector_fn=forbidden)
    with pytest.raises(ValueError, match="requires BF16"):
        Classic().prepare_augmented(prepared, config())


def test_prepared_augmented_dispatch_preserves_operands_and_fault(monkeypatch):
    from cutlass_gemm import backend

    prepared = operands()
    weights = Classic().prepare_augmented(prepared, config())["weights"]
    calls = []
    monkeypatch.setattr(backend, "require_device", lambda device: calls.append(device))
    marker = object()

    def native(a, b, passed_weights, row, col, bit):
        assert a is prepared.a and b is prepared.b and passed_weights is weights
        assert (row, col, bit) == (1, 0, 31)
        return marker

    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_augmented_mm", native, raising=False)
    assert prepared.augmented_native(weights, (1, 0, 31)) is marker
    assert calls == [prepared.a.device]


def test_prepared_augmented_rejects_fp32_before_device_check(monkeypatch):
    from cutlass_gemm import backend

    prepared = PreparedGemm.prepare(torch.ones(3, 16), torch.ones(16, 2), "fp32", "none")
    monkeypatch.setattr(backend, "require_device", forbidden)
    with pytest.raises(ValueError, match="requires BF16"):
        prepared.augmented_native(torch.ones(2, 3))


def test_augmented_engine_uses_native_evidence_and_records_precision(monkeypatch):
    prepared = operands()
    monkeypatch.setattr(prepared, "decode_block", forbidden)
    monkeypatch.setattr(prepared, "collected_native", forbidden)
    monkeypatch.setattr(prepared, "collected_reference_native", forbidden)
    monkeypatch.setattr(prepared, "augmented_native", lambda weights, fault: collect(prepared, weights, fault))
    output, report = execute_prepared(prepared, config(), reference_mm=forbidden, collector_fn=forbidden)
    torch.testing.assert_close(output, torch.full((3, 2), 16., dtype=torch.bfloat16))
    assert not report.detected and not report.nonfinite and not report.located
    assert report.location.tolist() == [-1, -1]
    assert report.execution_path == "instrumented"
    assert report.context["reference_mode"] == "augmented"
    assert report.context["reference_algorithm"] == "augmented_gemm"
    assert report.context["threshold_scale"] == "output"
    assert report.context["checksum_input_precision"] == "bf16"
    assert report.context["input_precision"] == "bf16"
    assert report.context["accumulator_precision"] == report.context["check_precision"] == "fp32"
    assert report.context["protection_scope"] == "computation_after_input_load"
    assert report.context["location_assumption"] == "disabled_for_low_precision_encoding"


@pytest.mark.parametrize("residual,detected", [(1., False), (3., True)])
def test_output_magnitude_controls_false_alarm_threshold(residual, detected):
    def collector(prepared, weights, fault):
        output, observed, hit, predicted, magnitude = collect(prepared, weights, fault)
        observed = observed.clone()
        observed[0, 0, 0] += residual
        # The fifth tensor is the native output-magnitude evidence, not |predicted|.
        magnitude.fill_(8.)
        return output, observed, hit, predicted, magnitude

    _, report = execute_prepared(operands(), config(atol=0., rtol=.25, on_error="report"),
                                 reference_collector_fn=collector)
    assert bool(report.detected) is detected
    torch.testing.assert_close(report.threshold, torch.full((2, 2), 2.))
    assert not report.located and report.location.tolist() == [-1, -1]


def test_accumulator_fault_detected_without_invalid_row_ratio_location():
    output, report = execute_prepared(
        operands(), config(atol=0., rtol=0., on_error="report"),
        {"location": "accumulator", "row": 1, "col": 0, "bit": 31},
        reference_collector_fn=collect,
    )
    assert output[1, 0] == -16.
    assert report.detected and not report.nonfinite
    assert not report.located and report.location.tolist() == [-1, -1]
    assert report.fault_hit.tolist() == [1]
    assert report.context["fault_selected"]
    torch.testing.assert_close(report.residual[:, 0], torch.tensor([-32., -64.]))


@pytest.mark.parametrize("location", ["a", "b"])
def test_augmented_rejects_input_memory_faults(location):
    with pytest.raises(ValueError, match="input memory faults"):
        execute_prepared(operands(), config(), {"location": location},
                         reference_collector_fn=forbidden)


@pytest.mark.parametrize("target", ["output", "observed", "predicted", "magnitude"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_evidence_always_detected_and_never_located(target, value):
    def collector(prepared, weights, fault):
        values = list(collect(prepared, weights, fault))
        index = {"output": 0, "observed": 1, "predicted": 3, "magnitude": 4}[target]
        values[index].reshape(-1)[0] = value
        return tuple(values)

    _, report = execute_prepared(operands(), config(on_error="report"),
                                 reference_collector_fn=collector)
    assert report.detected and report.nonfinite and not report.located
    assert report.location.tolist() == [-1, -1]


@pytest.mark.parametrize("corrupt_retry", [False, True])
def test_recompute_keeps_original_predicted_and_threshold_evidence(corrupt_retry):
    prepared = operands()
    clean_a, clean_b = prepared.a.clone(), prepared.b.clone()
    calls = []

    def collector(current, weights, fault):
        assert current is prepared
        calls.append(fault)
        output, observed, hit, predicted, magnitude = collect(current, weights, fault)
        if len(calls) == 2:
            # Neither a co-varying prediction nor an inflated retry tolerance
            # may authorize the replacement against new, untrusted evidence.
            predicted = predicted + 200.
            magnitude = torch.full_like(magnitude, 1e20)
            if corrupt_retry:
                output = output + 200.
                observed = predicted.clone()
        return output, observed, hit, predicted, magnitude

    output, report = execute_prepared(
        prepared, config(on_error="report", recovery="recompute"),
        {"location": "accumulator", "row": 1, "col": 0, "bit": 31},
        reference_collector_fn=collector,
    )
    assert calls == [(1, 0, 31), (-1, -1, -1)]
    assert report.detected and not report.located
    assert bool(report.recovered) is not corrupt_retry
    assert bool(report.recheck_failed) is corrupt_retry
    assert output[1, 0] == (-16. if corrupt_retry else 16.)
    torch.testing.assert_close(prepared.a, clean_a, rtol=0., atol=0.)
    torch.testing.assert_close(prepared.b, clean_b, rtol=0., atol=0.)
    expected_residual = torch.full((2, 2), 200.) if corrupt_retry else torch.zeros(2, 2)
    torch.testing.assert_close(report.retry_residual, expected_residual)


def test_unrecovered_augmented_fault_respects_raise_policy():
    with pytest.raises(AbftError) as error:
        execute_prepared(operands(), config(),
                         {"location": "accumulator", "row": 1, "bit": 31},
                         reference_collector_fn=collect)
    assert error.value.report.detected and not error.value.report.recovered
    assert not error.value.report.located


def test_plugin_without_augmented_implementation_is_explicitly_unsupported(monkeypatch):
    from abft import methods

    class SeparateOnly(AbftMethod):
        def prepare(self, *args):
            forbidden()

        def propagate(self, *args):
            forbidden()

        def verify(self, *args):
            forbidden()

    monkeypatch.setattr(methods, "_METHODS", {**methods._METHODS, "separate_only": SeparateOnly})
    cfg = AbftConfig(method="separate_only", reference_mode="augmented", threshold_scale="output")
    with pytest.raises(NotImplementedError, match="does not support augmented"):
        execute_prepared(operands(), cfg, reference_collector_fn=forbidden)
