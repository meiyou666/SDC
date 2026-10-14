"""CPU mock tests cover integration only, not low precision CUDA execution."""

from contextlib import contextmanager
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from low_precision import adapter
from training.config import ROOT


@pytest.fixture
def engine(monkeypatch):
    state = SimpleNamespace(active=False, calls=0, input_dtypes=[])

    @contextmanager
    def autocast(*, enabled, recipe):
        assert enabled and recipe == "recipe"
        assert not state.active
        state.active = True
        state.calls += 1
        try:
            yield
        finally:
            state.active = False

    class Linear(nn.Linear):
        def __init__(self, inp, out, *, bias, params_dtype, device, fuse_wgrad_accumulation):
            assert not fuse_wgrad_accumulation
            super().__init__(inp, out, bias=bias, device=device, dtype=params_dtype)
            if not bias:
                del self._parameters["bias"]
                self.bias = torch.empty(0, device=device, dtype=params_dtype)

        def forward(self, inp):
            assert state.active
            state.input_dtypes.append(inp.dtype)
            return super().forward(inp)

    te = SimpleNamespace(Linear=Linear, autocast=autocast)
    monkeypatch.setattr(adapter, "_load_engine", lambda precision: (te, "recipe"))
    return state


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@pytest.mark.parametrize("bias", [False, True])
def test_checkpoint_backward_and_update(engine, precision, bias):
    original = nn.Linear(32, 32, bias=bias)
    reference = copy.deepcopy(original)
    weight = original.weight
    layer = adapter.install(original, precision)
    assert layer.weight is weight
    assert layer.bias is original.bias
    layer.to("cpu")
    assert not any(t.is_meta for t in [*layer.parameters(), *layer.buffers()])
    x = torch.randn(2, 16, 32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = checkpoint(layer, x, use_reentrant=False)
        expected = reference(xr)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    assert engine.calls == 2 and not engine.active
    expected_dtype = torch.bfloat16 if precision == "fp4" else torch.float32
    assert engine.input_dtypes == [expected_dtype, expected_dtype]
    assert x.grad.dtype == torch.float32
    assert layer.weight.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad)
    assert layer.weight.grad.dtype == torch.float32
    previous = weight.detach().clone()
    torch.optim.SGD(layer.parameters(), lr=0.1).step()
    assert not torch.equal(previous, weight)


def test_aliases_tied_weights_and_state_dict(engine):
    shared = nn.Linear(32, 32).eval()
    tied = nn.Linear(32, 32)
    tied.weight = shared.weight
    model = nn.ModuleDict({"first": shared, "alias": shared, "tied": tied})
    parameters = list(model.parameters())
    adapter.install(model, "fp4")
    assert model["first"] is model["alias"]
    assert model["first"].weight is model["tied"].weight
    assert not model["first"].training
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model["first"].weight.zero_()
    model.load_state_dict(state)
    torch.testing.assert_close(model["first"].weight, state["first.weight"])


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_invalid_input_does_not_call_engine(engine, precision):
    layer = adapter.install(nn.Linear(32, 32), precision)
    with pytest.raises(ValueError, match="输入形状"):
        layer(torch.randn(3, 32))
    assert engine.calls == 0


def test_install_preflight_does_not_partially_replace(engine):
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 7))
    original = model[0]
    with pytest.raises(ValueError, match="线性层形状"):
        adapter.install(model, "fp4")
    assert model[0] is original
    with pytest.raises(ValueError, match="FP32"):
        adapter.install(nn.Linear(32, 32).bfloat16(), "fp8")


def test_custom_linear_is_not_overwritten(engine):
    class CustomLinear(nn.Linear):
        pass

    with pytest.raises(ValueError, match="自定义算子"):
        adapter.install(CustomLinear(32, 32), "fp8")


def test_metadata_and_cuda_requirement(monkeypatch):
    assert adapter.metadata("fp4")["transformer_engine_version"] == "2.21.0.dev0"
    assert adapter.metadata("fp8")["dgrad_operands"] == "E5M2/E4M3"
    with pytest.raises(ValueError, match="precision"):
        adapter.metadata("bf16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        adapter.install(nn.Linear(32, 32), "fp8")


def test_fp4_tma_requires_32_not_16(engine):
    with pytest.raises(ValueError, match="线性层形状"):
        adapter.install(nn.Linear(16, 32), "fp4")
    layer = adapter.install(nn.Linear(32, 32), "fp4")
    with pytest.raises(ValueError, match="32 的倍数"):
        layer(torch.randn(16, 32))
    assert engine.calls == 0


@pytest.mark.parametrize("capability", [(9, 0), (12, 1)])
def test_device_matches_compiled_architecture(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    with pytest.raises(RuntimeError, match="SM120"):
        adapter.install(nn.Linear(32, 32), "fp8")


def test_recipe_metadata_matches_environment_source():
    baseline = json.loads((ROOT / "environment/baseline.json").read_text())
    engine = baseline["training_profiles"]["nvidia"]["transformer_engine"]
    for precision in ("fp8", "fp4"):
        record = adapter.metadata(precision)
        assert record["transformer_engine_version"] == engine["version"]
        assert record["transformer_engine_source_commit"] == engine["commit"]
        assert "".join(map(str, record["target_compute_capability"])) == engine["cuda_archs"]
