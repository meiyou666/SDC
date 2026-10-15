"""CPU contract tests; no substitute GPU-performance or kernel evidence."""
from types import SimpleNamespace

import pytest
import torch

from abft import AbftConfig, FaultConfig
from abft.engine import AbftError, execute
from abft.methods import Classic, register_method
from abft.operands import PreparedGemm


def collect(operands, weights, fault):
    a, b = operands.decode_block(0, operands.k)
    output = a @ b
    row, col, bit = fault
    if row >= 0:
        mask = 1 << bit if bit < 31 else -(1 << 31)
        output.view(torch.int32)[row, col] ^= mask
    return output, (weights @ output).unsqueeze(1), torch.tensor([int(row >= 0)])


def run(a, b, **kwargs):
    return execute(a, b, "fp32", "none", reference_mm=torch.mm,
                   collector_fn=collect, **kwargs)


@pytest.mark.parametrize("precision,bit", [("fp32", 31), ("bf16", 15)])
@pytest.mark.parametrize("location", ["a", "b"])
def test_input_bitflip_clones_without_polluting_clean(precision, bit, location):
    a, b = torch.ones(3, 16), torch.ones(16, 2)
    prepared = PreparedGemm.prepare(a, b, precision, "none")
    active = prepared.injected(FaultConfig(location=location, row=0, col=0, bit=bit))
    assert getattr(active, location)[0, 0] == -1
    assert getattr(prepared, location)[0, 0] == 1
    assert torch.all(a == 1) and torch.all(b == 1)
    other = "b" if location == "a" else "a"
    assert getattr(active, other) is getattr(prepared, other)


def nvfp4():
    # Two blocks have different scales, and B is stored in transposed rows.
    a = torch.full((3, 16), 0x22, dtype=torch.uint8)
    b = torch.full((2, 16), 0x22, dtype=torch.uint8)
    def meta(rows):
        return SimpleNamespace(scales=torch.tensor([[2., 4.]] * rows).to(torch.float8_e4m3fn).view(torch.uint8),
                               multiplier=torch.tensor(2.))
    return PreparedGemm(a, b, "nvfp4", 3, 2, 32, meta(3), meta(2))


@pytest.mark.parametrize("col,expected", [(0, 0x23), (1, 0x32)])
def test_nvfp4_nibble_flip(col, expected):
    clean = nvfp4()
    changed = clean.injected(FaultConfig(location="a", col=col, bit=0))
    assert changed.a[0, 0] == expected
    assert clean.a[0, 0] == 0x22
    changed_b = clean.injected(FaultConfig(location="b", row=col, col=1, bit=0))
    assert changed_b.b[1, 0] == expected
    assert clean.b[1, 0] == 0x22


def test_nvfp4_reference_uses_actual_block_scales_and_b_orientation():
    prepared = nvfp4()
    state = Classic().prepare(prepared, AbftConfig(reference_block_k=16), torch.mm)
    a, b = prepared.decode_block(0, 32)
    assert torch.equal(a[:, :16], torch.ones(3, 16))
    assert torch.equal(a[:, 16:], torch.full((3, 16), 2.))
    assert torch.equal(state["expected"], state["weights"] @ (a @ b))
    with pytest.raises(ValueError, match="block16"):
        prepared.decode_block(1, 17)


def test_reference_is_chunked_and_matches_prepared_bf16(monkeypatch):
    torch.manual_seed(3)
    p = PreparedGemm.prepare(torch.randn(4, 48), torch.randn(48, 3), "bf16", "none")
    calls = []
    decode = p.decode_block
    def record(start, end):
        calls.append((start, end))
        return decode(start, end)
    monkeypatch.setattr(p, "decode_block", record)
    state = Classic().prepare(p, AbftConfig(reference_block_k=16), torch.mm)
    assert calls == [(0, 16), (16, 32), (32, 48)]
    torch.testing.assert_close(state["expected"], state["weights"] @ (p.a.float() @ p.b.float()))


def test_accumulator_detection_and_location():
    _, report = run(torch.ones(3, 16), torch.ones(16, 2),
                    abft_config={"method": "classic", "on_error": "report"},
                    fault_config={"location": "accumulator", "row": 1, "col": 0, "bit": 31})
    assert report.detected and report.located
    assert report.location.tolist() == [1, 0]
    assert report.to_record()["fault_hit"] == [1]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_always_detected_never_located(value):
    method = Classic()
    p = PreparedGemm.prepare(torch.ones(3, 16), torch.ones(16, 2), "fp32", "none")
    cfg = AbftConfig()
    state = method.prepare(p, cfg, torch.mm)
    output = p.a @ p.b
    output[1, 0] = value
    result = method.verify(state["weights"] @ output, output, state, cfg)
    assert result["detected"] and result["nonfinite"]
    assert not result["located"]


def test_ambiguous_numeric_location_rejected():
    method = Classic()
    p = PreparedGemm.prepare(torch.ones(3, 16), torch.ones(16, 2), "fp32", "none")
    cfg = AbftConfig(atol=0.4, rtol=0.)
    state = method.prepare(p, cfg, torch.mm)
    observed = state["expected"].clone()
    observed[:, 0] += torch.tensor([1., 2.])
    result = method.verify(observed, p.a @ p.b, state, cfg)
    assert result["detected"] and not result["located"]


def test_recovery_reuses_clean_preparation_and_rechecks(monkeypatch):
    original = PreparedGemm.prepare.__func__
    preparations = []
    def prepare(cls, *args):
        p = original(cls, *args)
        preparations.append(p)
        return p
    monkeypatch.setattr(PreparedGemm, "prepare", classmethod(prepare))
    a, b = torch.ones(3, 16), torch.ones(16, 2)
    output, report = run(a, b, abft_config={"method": "classic", "recovery": "recompute"},
                         fault_config={"location": "a", "bit": 31})
    assert len(preparations) == 1
    assert report.detected and report.recovered and not report.recheck_failed
    assert torch.equal(output, a @ b)


def test_failed_recheck_raises_and_preserves_report():
    def broken(p, weights, fault):
        c, partials, hit = collect(p, weights, fault)
        return c, partials + 100, hit
    with pytest.raises(AbftError) as exc:
        execute(torch.ones(3, 16), torch.ones(16, 2), "fp32", "none",
                abft_config={"method": "classic", "recovery": "recompute"},
                reference_mm=torch.mm, collector_fn=broken)
    assert exc.value.report.recheck_failed and not exc.value.report.recovered


def test_none_uses_original_backend_without_preparation(monkeypatch):
    from abft import engine
    expected = torch.ones(1)
    monkeypatch.setattr(engine.backend, "gemm", lambda *args: expected)
    monkeypatch.setattr(PreparedGemm, "prepare", lambda *args: pytest.fail("Unexpected preparation"))
    output, report = run(torch.ones(3, 16), torch.ones(16, 2))
    assert output is expected and report is None


def test_report_policy_has_no_item_sync(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "item", lambda *args: pytest.fail("Implicit sync"))
    _, report = run(torch.ones(3, 16), torch.ones(16, 2),
                    abft_config={"method": "classic", "on_error": "report"})
    assert report.method == "classic"


def test_method_plugin(monkeypatch):
    from abft import methods
    monkeypatch.setattr(methods, "_METHODS", dict(methods._METHODS))
    class Plugin(Classic):
        def verify(self, *args):
            result = super().verify(*args)
            result["plugin_used"] = torch.tensor(True)
            return result
    register_method("test_plugin", Plugin)
    _, report = run(torch.ones(3, 16), torch.ones(16, 2), abft_config="test_plugin")
    assert report.plugin_used
    with pytest.raises(ValueError, match="duplicate"):
        register_method("test_plugin", Plugin)


@pytest.mark.parametrize("context", [{"call_index": 2}, {"retry": True}, {"layer": "other"}])
def test_fault_selection(context):
    _, report = run(torch.ones(3, 16), torch.ones(16, 2),
                    abft_config="classic", context=context,
                    fault_config={"location": "a", "bit": 31, "layer": "target"})
    assert not report.context["fault_selected"] and not report.detected


def test_fp8_reference_decodes_stored_codes_and_multiplier():
    qa = SimpleNamespace(multiplier=torch.tensor(2.))
    qb = SimpleNamespace(multiplier=torch.tensor(4.))
    a = torch.full((3, 32), 2.).to(torch.float8_e4m3fn).view(torch.uint8)
    b = torch.full((2, 32), 4.).to(torch.float8_e4m3fn).view(torch.uint8)
    p = PreparedGemm(a, b, "fp8", 3, 2, 32, qa, qb)
    state = Classic().prepare(p, AbftConfig(reference_block_k=16), torch.mm)
    assert torch.equal(state["expected"], torch.tensor([[96., 96.], [192., 192.]]))
    changed = p.injected(FaultConfig(location="b", row=3, col=1, bit=7))
    _, decoded_b = changed.decode_block(0, 16)
    assert decoded_b[3, 1] == -1
    assert p.b[1, 3] == b[1, 3]


@pytest.mark.parametrize("diagnostics", [False, True])
def test_report_diagnostics_and_precision_metadata(diagnostics):
    _, report = run(torch.ones(3, 16), torch.ones(16, 2),
                    abft_config={"method": "classic", "diagnostics": diagnostics})
    assert ("residual" in report.tensors) == diagnostics
    assert ("threshold" in report.tensors) == diagnostics
    assert (report.context["m"], report.context["n"], report.context["k"]) == (3, 2, 16)
    assert report.context["accumulator_precision"] == "fp32"


@pytest.mark.parametrize("value", [{"atol": float("nan")}, {"rtol": -1},
                                    {"reference_block_k": 17}, {"diagnostics": 1}])
def test_invalid_config(value):
    with pytest.raises(ValueError):
        AbftConfig(**value)


def test_reference_packs_fp32_k_slices_for_native_contract():
    p = PreparedGemm.prepare(torch.ones(3, 48), torch.ones(48, 2), "fp32", "none")
    calls = []
    def packed_mm(a, b):
        assert a.is_contiguous() or a.T.is_contiguous()
        assert b.is_contiguous() or b.T.is_contiguous()
        calls.append((a.shape, b.shape))
        return a @ b
    Classic().prepare(p, AbftConfig(reference_block_k=16), packed_mm)
    assert len(calls) == 12


@pytest.mark.parametrize("precision,bit", [("fp32", 31), ("bf16", 15)])
def test_preparation_and_fault_preserve_native_column_layout(precision, bit):
    a, b = torch.ones(16, 3).T, torch.ones(2, 16).T
    p = PreparedGemm.prepare(a, b, precision, "none")
    assert p.a.stride() == a.stride()
    assert p.b.stride() == b.stride()
    active = p.injected(FaultConfig(location="b", row=3, col=1, bit=bit))
    assert active.b.stride() == p.b.stride()
    assert active.b[3, 1] == -1
    assert p.b[3, 1] == 1


def test_report_serialization_preserves_nonfinite_paths_as_strict_json():
    import json

    from abft.engine import AbftReport
    report = AbftReport({"nonfinite": torch.tensor(True),
                         "residual": torch.tensor([[float("nan"), float("inf"), -float("inf")]])},
                        {}, "classic", "instrumented")
    record = report.to_record()
    assert record["nonfinite"] is True
    assert record["residual"] == [[None, None, None]]
    assert record["nonfinite_metrics"] == {"residual[0][0]": "nan", "residual[0][1]": "inf",
                                           "residual[0][2]": "-inf"}
    json.dumps(record, allow_nan=False)
