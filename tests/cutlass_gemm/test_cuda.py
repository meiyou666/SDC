"""Actual CUTLASS checks; require the configured SM120 GPU."""

import copy

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from cutlass_gemm import backend
from cutlass_gemm.linear import CutlassLinear, install


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual CUTLASS execution requires SM120 (RTX 5090)",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_packed_layouts_against_fp64(dtype, transpose_a, transpose_b):
    torch.manual_seed(15)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = torch.randn(24, 32, device="cuda", dtype=dtype)
        b = torch.randn(32, 16, device="cuda", dtype=dtype)
        if transpose_a:
            a = a.T.contiguous().T
        if transpose_b:
            b = b.T.contiguous().T
        result = backend.mm(a, b)
        expected = (a.double() @ b.double()).to(dtype)
    stream.synchronize()
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected, rtol=0.008 if dtype == torch.bfloat16 else 1e-5,
                               atol=0.004 if dtype == torch.bfloat16 else 1e-5)


def test_invalid_dtype_and_alignment_raise():
    half = torch.ones(8, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="BF16 and FP32"):
        backend.mm(half, half)
    a = torch.ones(8, 9, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(9, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="can_implement"):
        backend.mm(a, b)
    with pytest.raises(RuntimeError, match="same dtype"):
        backend.mm(half.float(), half.bfloat16())


def test_bf16_linear_gradients():
    torch.manual_seed(16)
    original = nn.Linear(16, 24, device="cuda")
    reference = copy.deepcopy(original)
    layer = CutlassLinear(original)
    x = torch.randn(8, 16, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = layer(x)
        expected = reference(xr)
    actual.sum().backward()
    expected.sum().backward()
    for a, b in ((actual, expected), (x.grad, xr.grad),
                 (layer.weight.grad, reference.weight.grad), (layer.bias.grad, reference.bias.grad)):
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.02)
    assert layer.weight.grad.dtype == torch.float32


def test_qwen_checkpointed_training_step():
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=32, attention_dropout=0.0,
                         use_cache=False)
    config._attn_implementation = "eager"
    model = install(Qwen3ForCausalLM(config).cuda()).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    weights = {name: layer.weight.detach().clone() for name, layer in model.named_modules()
               if isinstance(layer, CutlassLinear)}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tokens = torch.randint(0, 32, (2, 8), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=tokens, labels=tokens).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    for name, before in weights.items():
        weight = model.get_submodule(name).weight
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
        assert not torch.equal(weight, before)
