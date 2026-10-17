"""CPU tests exercise the Python bridge with torch.mm, never CUTLASS."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config, Qwen3ForCausalLM

from cutlass_gemm import backend
from cutlass_gemm.linear import CutlassLinear, install


@pytest.fixture
def bridge(monkeypatch):
    calls = []

    def mm(a, b):
        assert a.is_contiguous() or a.T.is_contiguous()
        assert b.is_contiguous() or b.T.is_contiguous()
        calls.append((a.dtype, b.dtype))
        return torch.mm(a, b)

    monkeypatch.setattr(backend, "mm", mm)
    return calls


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("summed", [False, True])
def test_forward_backward_noncontiguous(bridge, bias, summed):
    torch.manual_seed(11)
    original = nn.Linear(8, 16, bias=bias)
    reference = copy.deepcopy(original)
    layer = CutlassLinear(original)
    x = torch.randn(2, 8, 3).transpose(1, 2).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    actual, expected = layer(x), reference(xr)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    if summed:
        actual.sum().backward()
        expected.sum().backward()
    else:
        actual.backward(gradient)
        expected.backward(gradient)
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad)
    if bias:
        torch.testing.assert_close(layer.bias.grad, reference.bias.grad)
    assert len(bridge) == 3


def test_bf16_autocast_preserves_master_gradients(bridge):
    layer = CutlassLinear(nn.Linear(8, 16))
    x = torch.randn(8, 8, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(x)
    out.sum().backward()
    assert out.dtype == torch.bfloat16
    assert bridge == [(torch.bfloat16, torch.bfloat16)] * 3
    assert layer.weight.dtype == layer.weight.grad.dtype == torch.float32
    assert layer.bias.grad.dtype == x.grad.dtype == torch.float32
    torch.testing.assert_close(layer.weight.grad, x.detach().bfloat16().sum(0).float().expand(16, -1))


def test_install_preserves_parameters_shared_modules_and_mode(bridge):
    shared = nn.Linear(8, 8).eval()
    tied = nn.Linear(8, 8)
    tied.weight = shared.weight
    model = nn.ModuleDict({"first": shared, "alias": shared, "tied": tied})
    parameters = list(model.parameters())
    install(model)
    assert model["first"] is model["alias"]
    assert model["first"].weight is model["tied"].weight
    assert not model["first"].training
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert isinstance(install(nn.Linear(8, 8)), CutlassLinear)


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpoint_recomputation(bridge, reentrant):
    torch.manual_seed(12)
    layer = CutlassLinear(nn.Linear(8, 8))
    reference = copy.deepcopy(layer)
    x = torch.randn(8, 8, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    checkpoint(layer, x, use_reentrant=reentrant).square().sum().backward()
    reference(xr).square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad)
    torch.testing.assert_close(layer.bias.grad, reference.bias.grad)
    assert len(bridge) >= 7


def test_qwen_all_linear_layers_update(bridge):
    torch.manual_seed(13)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=32, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config)
    names = {name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)}
    before = {name: layer.weight.detach().clone() for name, layer in model.named_modules() if name in names}
    install(model)
    assert "lm_head" in names
    assert any("self_attn.q_proj" in name for name in names)
    assert all(isinstance(model.get_submodule(name), CutlassLinear) for name in names)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tokens = torch.randint(0, 32, (2, 8))
    loss = model(input_ids=tokens, labels=tokens).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    for name in names:
        layer = model.get_submodule(name)
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
        assert not torch.equal(layer.weight, before[name])


def test_backend_rejects_cpu_without_loading(monkeypatch):
    def unexpected():
        pytest.fail("CPU calls must fail before loading the CUDA extension")

    monkeypatch.setattr(backend, "load_extension", unexpected)
    with pytest.raises(RuntimeError, match="CUDA"):
        backend.mm(torch.ones(8, 8), torch.ones(8, 8))
