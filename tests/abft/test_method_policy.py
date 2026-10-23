"""Declared CUDA policies, persistent method lifecycle, and Sum execution."""

from contextlib import contextmanager

import pytest
import torch

from abft import AbftConfig, execute_prepared, methods, native
from abft.engine import AbftError
from abft.methods import Classic, CompiledPolicy, Sum, get_method, register_method
from abft.operands import PreparedGemm
from cutlass_gemm import backend


GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Native CUDA policies require the verified SM120 project GPU",
)


def _prepared():
    return PreparedGemm.prepare(torch.ones(3, 16), torch.ones(16, 8), "fp32", "none")


def _collector(p, weights, fault):
    output = p.a @ p.b
    if fault[0] >= 0:
        mask = 1 << fault[2] if fault[2] < 31 else -(1 << 31)
        output.view(torch.int32)[fault[0], fault[1]] ^= mask
    return output, (weights @ output).unsqueeze(1), torch.tensor([int(fault[0] >= 0)], dtype=torch.int32)


@pytest.mark.parametrize("algorithm,channels", [(0, 1), (1, 2), (-1, 2), (2**32, 2), (True, 2)])
def test_compiled_policy_requires_installed_exact_id_and_channel_count(algorithm, channels):
    with pytest.raises(ValueError, match="installed"):
        CompiledPolicy(algorithm, channels)


def test_sum_is_registered_single_projection_and_has_no_locator():
    method = get_method("sum")
    p = _prepared()
    config = AbftConfig(method="sum", on_error="report", diagnostics=True)
    state = method.prepare(p, config, torch.mm)
    assert isinstance(method, Sum) and state["weights"].shape == (1, p.m)
    assert torch.equal(state["weights"], torch.ones(1, p.m))
    output, report = execute_prepared(
        p, config, {"location": "accumulator", "row": 1, "col": 2, "bit": 31},
        method_instance=method, collector_fn=_collector, reference_mm=torch.mm,
    )
    assert output[1, 2] == -16
    assert report.detected and not report.located
    assert report.location.tolist() == [-1, -1]
    assert report.residual.shape == (1, p.n)
    assert report.context["compiled_policy"]["id"] == 1
    assert report.context["compiled_policy"]["channels"] == 1


@pytest.mark.parametrize("name", ["classic", "sum"])
def test_method_rejects_parameters_before_collector(name):
    with pytest.raises(ValueError, match="does not accept"):
        execute_prepared(_prepared(), {"method": name, "parameters": {"unsupported": 1}},
                         collector_fn=lambda *args: pytest.fail("Unexpected collector"))


def test_python_subclass_cannot_silently_inherit_native_verification(monkeypatch):
    monkeypatch.setattr(methods, "_METHODS", dict(methods._METHODS))
    class CustomVerifier(Classic):
        def verify(self, *args):
            pytest.fail("Native silently selected a Python verifier")
    register_method("custom_verifier", CustomVerifier)
    config = AbftConfig(method="custom_verifier", implementation="native", reference_mode="input_fused")
    monkeypatch.setattr(backend, "require_device", lambda *args: pytest.fail("Unsupported policy reached CUDA"))
    with pytest.raises(ValueError, match="explicit compiled"):
        execute_prepared(_prepared(), config)


def test_method_instance_matches_registry_selection():
    with pytest.raises(ValueError, match="configured"):
        execute_prepared(_prepared(), {"method": "classic"}, method_instance=Sum())
    with pytest.raises(ValueError, match="enabled"):
        execute_prepared(_prepared(), method_instance=Classic())


def test_stateful_methods_require_retained_instance(monkeypatch):
    monkeypatch.setattr(methods, "_METHODS", dict(methods._METHODS))
    class Stateful(Classic):
        stateful = True
    register_method("stateful_required", Stateful)
    with pytest.raises(ValueError, match="retained method_instance or MethodSession"):
        execute_prepared(_prepared(), {"method": "stateful_required"},
                         reference_mm=lambda *args: pytest.fail("Unexpected preparation"))


@pytest.mark.parametrize("value", [1, "true", None])
def test_method_statistics_requirement_must_be_boolean(monkeypatch, value):
    monkeypatch.setattr(methods, "_METHODS", dict(methods._METHODS))
    class InvalidStatistics(Classic):
        needs_diagnostics = value
    register_method("invalid_statistics", InvalidStatistics)
    with pytest.raises(TypeError, match="needs_diagnostics must be a bool"):
        execute_prepared(_prepared(), {"method": "invalid_statistics"},
                         collector_fn=lambda *args: pytest.fail("Unexpected collector"))


@pytest.mark.parametrize("recheck_fails,on_error", [(False, "report"), (True, "report"), (True, "raise")])
def test_commit_once_after_retry_without_fault_truth_and_with_frozen_state(monkeypatch, recheck_fails, on_error):
    monkeypatch.setattr(methods, "_METHODS", dict(methods._METHODS))
    active_scopes = []

    @contextmanager
    def profile_scope(name):
        active_scopes.append(name)
        try:
            yield
        finally:
            active_scopes.pop()

    class Stateful(Classic):
        stateful = True

        def __init__(self):
            self.history = 0
            self.prepared_states = []
            self.verifications = []
            self.commits = []

        def prepare(self, operands, config, mm=None):
            state = super().prepare(operands, config, mm)
            state["history"] = self.history
            self.prepared_states.append(state)
            return state

        def verify(self, observed, output, state, config):
            assert state is self.prepared_states[-1]
            assert state["history"] == self.history
            self.verifications.append(state)
            return super().verify(observed, output, state, config)

        def commit(self, state, decision, context):
            assert active_scopes[-1] == "abft.state_update"
            assert len(self.verifications) == 2 and state is self.prepared_states[0]
            assert len(self.prepared_states) == 1
            assert not {"fault", "fault_hit", "fault_selected", "injection_point"} & decision.keys()
            assert not {"fault", "fault_hit", "fault_selected", "injection_point"} & context.keys()
            assert context["scale_profile"] == "fixed"
            self.commits.append((state, decision, context))
            self.history += 1

        def state_dict(self):
            return {"history": self.history}

        def load_state_dict(self, state):
            self.history = state["history"]

    register_method("stateful_transaction", Stateful)
    method = get_method("stateful_transaction")
    calls = []

    def collector(p, weights, fault):
        calls.append(fault)
        output, partials, hit = _collector(p, weights, fault)
        if len(calls) == 2 and recheck_fails:
            output = -output
            partials = (weights @ output).unsqueeze(1)
        return output, partials, hit

    kwargs = dict(method_instance=method, reference_mm=torch.mm, collector_fn=collector,
                  profile_scope=profile_scope)
    config = AbftConfig(method="stateful_transaction", recovery="recompute", on_error=on_error,
                        diagnostics=False)
    context = {"scale_profile": "fixed", "fault": "forbidden", "fault_hit": 123,
               "fault_selected": True, "injection_point": "forbidden"}
    if on_error == "raise":
        with pytest.raises(AbftError):
            execute_prepared(_prepared(), config, {"location": "accumulator", "bit": 31}, context, **kwargs)
    else:
        execute_prepared(_prepared(), config, {"location": "accumulator", "bit": 31}, context, **kwargs)
    assert calls == [(0, 0, 31), (-1, -1, -1)]
    assert len(method.commits) == 1 and method.state_dict() == {"history": 1}
    saved_decision = method.commits[0][1]
    assert "residual" in saved_decision and "threshold" in saved_decision
    assert bool(saved_decision["recovered"]) is not recheck_fails
    replacement = get_method("stateful_transaction")
    replacement.load_state_dict(method.state_dict())
    assert replacement.history == 1


def test_plugin_and_stateful_weights_are_not_put_in_builtin_cache(monkeypatch):
    class StatefulSum(Sum):
        stateful = True
        compiled_policy = CompiledPolicy(1, 1)

        def __init__(self):
            self.preparations = 0

        def prepare_fused(self, p, cfg):
            self.preparations += 1
            return {"weights": torch.full((1, p.m), float(self.preparations)), "m": p.m, "n": p.n}

    method = StatefulSum()
    config = AbftConfig(method="sum", reference_mode="input_fused", implementation="native")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: pytest.fail("Plugin accessed static cache"))
    first = native._builtin_weights(method, _prepared(), config, method.compiled_policy)
    second = native._builtin_weights(method, _prepared(), config, method.compiled_policy)
    assert first["weights"].data_ptr() != second["weights"].data_ptr()
    assert method.preparations == 2


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp8", "nvfp4"])
def test_prepared_collectors_append_real_policy_but_preserve_classic_arity(monkeypatch, precision):
    if precision in {"fp32", "bf16"}:
        p = PreparedGemm.prepare(torch.ones(3, 16), torch.ones(16, 8), precision, "none")
        plain, fused = "abft_mm", "abft_reference_mm"
    else:
        width = 16 if precision == "fp8" else 8
        p = PreparedGemm(torch.zeros(3, width, dtype=torch.uint8),
                         torch.zeros(8, width, dtype=torch.uint8), precision, 3, 8, 16,
                         alpha=torch.tensor(1.), sfa=torch.ones(1), sfb=torch.ones(1))
        plain = "abft_fp8_mm" if precision == "fp8" else "abft_nvfp4_mm"
        fused = "abft_fp8_reference_mm" if precision == "fp8" else "abft_nvfp4_reference_mm"
    calls = []
    marker = object()

    def collect(*args):
        calls.append(args)
        return marker

    monkeypatch.setattr(backend, "require_device", lambda *args: None)
    monkeypatch.setattr(torch.ops.sdc_cutlass, plain, collect, raising=False)
    monkeypatch.setattr(torch.ops.sdc_cutlass, fused, collect, raising=False)
    weights = torch.ones(1, 3)
    fault = (1, 2, 30)
    flag = torch.empty(1, dtype=torch.int32)
    assert p.collected_native(weights, fault) is marker
    assert calls[-1][-3:] == fault
    plain_arity = len(calls[-1])
    assert p.collected_native(weights, fault, algorithm=1, nonfinite=flag) is marker
    assert len(calls[-1]) == plain_arity + 2
    assert calls[-1][-2] == 1 and calls[-1][-1] is flag
    assert p.collected_reference_native(weights, fault) is marker
    assert calls[-1][-3:] == fault
    fused_arity = len(calls[-1])
    assert p.collected_reference_native(weights, fault, algorithm=1, nonfinite=flag) is marker
    assert len(calls[-1]) == fused_arity + 2
    assert calls[-1][-2] == 1 and calls[-1][-1] is flag


def test_augmented_sum_policy_is_forwarded_without_nonfinite_extra_argument(monkeypatch):
    p = PreparedGemm.prepare(torch.ones(3, 16), torch.ones(16, 8), "bf16", "none")
    calls = []
    monkeypatch.setattr(backend, "require_device", lambda *args: None)
    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_augmented_mm", lambda *args: calls.append(args), raising=False)
    weights = torch.ones(1, 3, dtype=torch.bfloat16)
    p.augmented_native(weights)
    assert len(calls[-1]) == 6
    p.augmented_native(weights, algorithm=1)
    assert len(calls[-1]) == 7 and calls[-1][-1] == 1


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp8", "nvfp4"])
def test_native_input_fused_selects_policy_and_actual_output_flag_without_python_math(monkeypatch, precision):
    p = _prepared()
    p.precision = precision
    cfg = AbftConfig(method="sum", reference_mode="input_fused", implementation="native", on_error="report")
    calls = []
    monkeypatch.setattr(backend, "require_device", lambda *args: None)
    monkeypatch.setattr(native, "_builtin_weights", lambda *args: {"weights": torch.ones(1, p.m), "m": p.m, "n": p.n})

    def collect(weights, fault, *, algorithm, nonfinite):
        assert algorithm == 1 and weights.shape == (1, p.m)
        assert nonfinite.dtype == torch.int32 and nonfinite.shape == (1,)
        nonfinite.fill_(1)  # Simulate C++ epilogue's post-cast flag, without scanning C.
        calls.append(nonfinite)
        partials = torch.zeros(1, 1, 128)
        return torch.zeros(p.m, p.n), partials, torch.zeros(1, dtype=torch.int32), partials, partials

    def finalize(observed, predicted, magnitude, hit, nonfinite, n, atol, rtol, reference, diagnostics, algorithm):
        assert nonfinite is calls[0] and algorithm == 1
        return (torch.zeros(2, 1, n), torch.empty(0), torch.empty(0),
                torch.tensor([True, True, False, False, False]), hit,
                torch.full((2,), -1, dtype=torch.int64))

    monkeypatch.setattr(p, "collected_reference_native", collect)
    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_finalize_reference", finalize, raising=False)
    monkeypatch.setattr(Sum, "verify", lambda *args: pytest.fail("Native called Python verify"))
    monkeypatch.setattr(Sum, "propagate", lambda *args: pytest.fail("Native called Python merge"))
    _, report = execute_prepared(p, cfg)
    assert report.nonfinite and report.detected
    assert report.execution_path == "input_fused_native"
    assert report.context["checksum_input_precision"] == "fp32"
    assert report.context["compiled_reference_format"] == "fp32_projection_fp32_accumulation"
    assert not report.located


@pytest.mark.parametrize("reference_mode", ["augmented", "input_fused"])
@pytest.mark.parametrize("algorithm,channels", [(0, 2), (1, 1)])
@pytest.mark.parametrize("needs_statistics", [False, True])
def test_native_adaptive_threshold_is_frozen_until_retry_and_commit(monkeypatch, reference_mode, algorithm, channels, needs_statistics):
    monkeypatch.setattr(methods, "_METHODS", dict(methods._METHODS))
    p = PreparedGemm.prepare(torch.ones(3, 16), torch.ones(16, 8), "bf16", "none")
    finalizer_calls, kernel_calls = [], []
    first_reference = torch.zeros(2, channels, p.n)
    residuals = [torch.full((channels, p.n), 2.0), torch.zeros(channels, p.n)]
    returned_threshold = torch.full((channels, p.n), 0.25)
    active_scopes = []

    @contextmanager
    def profile_scope(name):
        active_scopes.append(name)
        try:
            yield
        finally:
            active_scopes.pop()

    class Adaptive(Classic):
        stateful = True
        needs_diagnostics = needs_statistics
        compiled_policy = CompiledPolicy(algorithm, channels)

        def __init__(self):
            self.preparations = 0
            self.commits = 0
            self.history = torch.tensor([0.25], dtype=torch.float32)

        def _prepare(self, dtype):
            self.preparations += 1
            self.snapshot = self.history.clone()
            self.weights = torch.ones(channels, p.m, dtype=dtype)
            return {"weights": self.weights, "thresholds": self.snapshot, "m": p.m, "n": p.n}

        def prepare_fused(self, operands, config):
            return self._prepare(torch.float32)

        def prepare_augmented(self, operands, config):
            return self._prepare(torch.bfloat16)

        def verify(self, *args):
            pytest.fail("Compiled native policy invoked Python verify")

        def commit(self, state, decision, context):
            assert active_scopes[-1] == "abft.state_update"
            assert len(finalizer_calls) == 2 and self.preparations == 1
            assert state["weights"] is self.weights and state["thresholds"] is self.snapshot
            assert state["reference"] is first_reference
            if needs_statistics:
                assert decision["residual"] is residuals[0]
                assert decision["retry_residual"] is residuals[1]
                assert decision["threshold"] is returned_threshold
            else:
                assert "residual" not in decision
                assert decision["threshold"] is self.snapshot
            assert not {"fault", "fault_hit", "fault_selected"} & decision.keys()
            assert not {"fault", "fault_hit", "fault_selected"} & context.keys()
            self.history.add_(1)
            self.commits += 1

    name = "adaptive_native_test"
    register_method(name, Adaptive)
    method = get_method(name)
    monkeypatch.setattr(backend, "require_device", lambda *args: None)

    def compact(a, b, weights, row, col, bit, *policy_args):
        assert policy_args == ((algorithm,) if algorithm else ())
        assert weights is method.weights
        kernel_calls.append((row, col, bit))
        return torch.ones(p.m, p.n, dtype=torch.bfloat16), torch.zeros(3, channels, 1, 128), torch.zeros(1, 1, dtype=torch.int32)

    def collected(weights, fault, *, algorithm, nonfinite):
        assert weights is method.weights
        kernel_calls.append(fault)
        nonfinite.zero_()
        partial = torch.full((channels, 1, 128), float(len(kernel_calls)))
        return torch.ones(p.m, p.n), partial, torch.tensor([int(fault[0] >= 0)], dtype=torch.int32), partial, partial

    def finish(first, thresholds, passed_algorithm):
        assert passed_algorithm == algorithm and thresholds is method.snapshot
        assert torch.equal(thresholds, torch.tensor([0.25]))
        assert first is (None if not finalizer_calls else first_reference)
        finalizer_calls.append((first, thresholds))
        detected = len(finalizer_calls) == 1
        return (first_reference, residuals[len(finalizer_calls) - 1] if needs_statistics else torch.empty(0),
                returned_threshold if needs_statistics else torch.empty(0),
                torch.tensor([detected, False, False, False, False]),
                torch.tensor([int(detected)], dtype=torch.int32), torch.full((2,), -1, dtype=torch.int64))

    def finalize_aug(partials, flags, n, atol, rtol, reference, diagnostics, passed_algorithm, thresholds):
        assert diagnostics is needs_statistics
        return finish(reference, thresholds, passed_algorithm)

    def finalize_fused(observed, predicted, magnitude, hit, nonfinite, n, atol, rtol,
                       reference, diagnostics, passed_algorithm, thresholds):
        assert diagnostics is needs_statistics
        return finish(reference, thresholds, passed_algorithm)

    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_augmented_compact", compact, raising=False)
    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_finalize", finalize_aug, raising=False)
    monkeypatch.setattr(torch.ops.sdc_cutlass, "abft_finalize_reference", finalize_fused, raising=False)
    monkeypatch.setattr(p, "collected_reference_native", collected)
    cfg = AbftConfig(method=name, implementation="native", reference_mode=reference_mode,
                     threshold_scale="output" if reference_mode == "augmented" else "input",
                     recovery="recompute", on_error="report")
    _, report = execute_prepared(p, cfg, {"location": "accumulator", "bit": 31},
                                {"fault_hit": "forbidden", "fault_selected": True},
                                method_instance=method, profile_scope=profile_scope)
    assert method.commits == 1 and method.preparations == 1
    assert kernel_calls == [(0, 0, 31), (-1, -1, -1)]
    assert report.recovered and not report.recheck_failed
    assert report.context["threshold_source"] == "method_state"
    assert not {"residual", "threshold", "retry_residual"} & report.tensors.keys()
    assert torch.equal(method.snapshot, torch.tensor([0.25]))
    assert torch.equal(method.history, torch.tensor([1.25]))


@GPU
@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp8", "nvfp4"])
@pytest.mark.parametrize("method", ["classic", "sum"])
def test_native_input_fused_real_policy_clean_fault_and_recovery(precision, method):
    backend.require_device(torch.device("cuda"))
    a = torch.zeros(128, 128, device="cuda")
    b = torch.zeros(128, 128, device="cuda")
    scaling = {"fp32": "none", "bf16": "none", "fp8": "tensorwise", "nvfp4": "block16"}[precision]
    prepared = PreparedGemm.prepare(a, b, precision, scaling, global_scale=1.0)
    expected = prepared.raw_native()
    cfg = AbftConfig(method=method, implementation="native", reference_mode="input_fused",
                     recovery="recompute", on_error="report", diagnostics=True)
    clean, clean_report = execute_prepared(prepared, cfg)
    assert torch.equal(clean, expected) and not clean_report.detected
    output, report = execute_prepared(prepared, cfg, {"location": "accumulator", "row": 1, "col": 2, "bit": 30})
    assert report.detected and report.recovered and not report.recheck_failed
    assert report.fault_hit.item() == 1 and not report.located
    assert torch.equal(output, expected)
    assert report.residual.shape == (2 if method == "classic" else 1, prepared.n)


@GPU
def test_sum_augmented_is_bf16_same_precision_single_channel():
    a = torch.zeros(129, 80, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(80, 144, device="cuda", dtype=torch.bfloat16)
    a[:, 0], b[0] = 1, 1
    p = PreparedGemm.prepare(a, b, "bf16", "none")
    cfg = AbftConfig(method="sum", reference_mode="augmented", threshold_scale="output",
                     implementation="native", diagnostics=True, on_error="report")
    output, report = execute_prepared(p, cfg)
    assert torch.equal(output, p.raw_native())
    assert report.residual.shape == (1, 144) and not report.detected
    assert report.context["checksum_input_precision"] == "bf16"
    assert report.context["augmented_rows_per_data_tile"] == 1
