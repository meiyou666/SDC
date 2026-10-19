"""Real official CUTLASS quantization, GEMMs and CUDA Trainer acceptance."""

import json
import math

import pytest
from safetensors.torch import load_file
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from cutlass_gemm import backend
from cutlass_gemm.linear import CutlassLinear, install
from cutlass_gemm.quantization import dequantize, quantize


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual low-precision CUTLASS execution requires SM120",
)
SCALING = {"fp8": "tensorwise", "nvfp4": "block16"}


def test_nvfp4_known_codes_midpoints_zero_and_scales():
    # E2M1 encodings and round-to-nearest-even ties, independent of decode().
    values = torch.tensor([
        [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
        [0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., -0., -.25, -.75, -1.25, -1.75, -2.5, -3.5, -6.],
    ], device="cuda")
    encoded = quantize(values, "nvfp4", 1.0)
    expected = torch.tensor([
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        [0x00, 0x22, 0x44, 0x66, 0x88, 0xAA, 0xCC, 0xFE],
    ], dtype=torch.uint8, device="cuda")
    assert torch.equal(encoded.codes, expected)
    assert torch.equal(encoded.scales, torch.full((2, 1), 0x38, dtype=torch.uint8, device="cuda"))
    # Reusing a compiled converter must consume the new tensor and its values.
    reversed_sign = quantize(-values, "nvfp4", 1.0)
    assert torch.equal(reversed_sign.codes, expected ^ 0x88)
    decoded = dequantize(encoded)
    torch.testing.assert_close(decoded[0], values[0], rtol=0, atol=0)
    assert torch.signbit(decoded[0, 8]) and not torch.signbit(decoded[0, 0])
    zeros = quantize(torch.zeros(4, 64, device="cuda"), "nvfp4")
    assert zeros.multiplier.item() == 1.0
    assert not zeros.codes.any() and not zeros.scales.any()
    assert torch.equal(dequantize(zeros), torch.zeros(4, 64, device="cuda"))
    # Four constant blocks have exactly representable ideal scales. With max
    # input 12, amax policy gives st=224 and scales [224,448,112,0].
    blocks = torch.tensor([6., 12., 3., 0.], device="cuda").repeat_interleave(16).reshape(1, 64)
    scaled = quantize(blocks, "nvfp4")
    assert scaled.multiplier.item() == 224.0
    assert torch.equal(scaled.scales, torch.tensor([[0x76, 0x7E, 0x6E, 0]], dtype=torch.uint8, device="cuda"))
    torch.testing.assert_close(dequantize(scaled), blocks, rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["fp8", "nvfp4"])
def test_native_gemm_quantized_reference_on_nondefault_stream(precision):
    backend.require_device(torch.device("cuda"))
    # Rows cross the 128-row SF tile and K has 16 block scales, exercising
    # both row and scale-factor interleaving beyond the first layout tile.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.manual_seed(83)
        a = torch.randn(160, 256, device="cuda")
        b_rows = torch.randn(192, 256, device="cuda")
        qa, qb = quantize(a, precision), quantize(b_rows, precision)
        alpha = (qa.multiplier * qb.multiplier).reciprocal()
        if precision == "fp8":
            output = torch.ops.sdc_cutlass.fp8_mm(
                qa.codes.view(torch.float8_e4m3fn), qb.codes.view(torch.float8_e4m3fn), alpha,
            )
        else:
            ia, ib, size_a, size_b = backend._scale_layout(160, 192, 256, a.device)
            sa = backend._pack_scales(qa.scales, ia, size_a)
            sb = backend._pack_scales(qb.scales, ib, size_b)
            output = torch.ops.sdc_cutlass.nvfp4_mm(qa.codes, qb.codes, sa, sb, 256, alpha)
        reference = dequantize(qa).double() @ dequantize(qb).double().T
        routed = backend.gemm(a, b_rows.T, precision, SCALING[precision])
    stream.synchronize()
    assert torch.isfinite(output).all() and torch.isfinite(routed).all()
    assert output.dtype == (torch.float32 if precision == "fp8" else torch.bfloat16)
    torch.testing.assert_close(output.double(), reference, rtol=0.008 if precision == "nvfp4" else 2e-5,
                               atol=0.02 if precision == "nvfp4" else 2e-4)
    torch.testing.assert_close(routed, output, rtol=0, atol=0)


def _qwen():
    config = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=64, attention_dropout=0.0, use_cache=False,
        tie_word_embeddings=True, eos_token_id=1, pad_token_id=1,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).cuda().train()


@pytest.mark.parametrize("precision", ["fp8", "nvfp4"])
def test_tied_qwen_checkpointed_real_forward_backward(precision, monkeypatch):
    backend.require_device(torch.device("cuda"))
    torch.manual_seed(17)
    model = install(_qwen(), precision=precision, scaling=SCALING[precision])
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert all(isinstance(layer, CutlassLinear) for layer in model.modules() if isinstance(layer, nn.Linear))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.is_gradient_checkpointing
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    calls = []
    original_gemm = backend.gemm
    op_name = "fp8_mm" if precision == "fp8" else "nvfp4_mm"
    original_op = getattr(torch.ops.sdc_cutlass, op_name)
    native_calls = []

    def observed_op(*args, **kwargs):
        native_calls.append(tuple(args[0].shape))
        return original_op(*args, **kwargs)

    def observed_gemm(a, b, selected, scaling, global_scale="amax"):
        calls.append((tuple(a.shape), tuple(b.shape)))
        assert selected == precision and scaling == SCALING[precision]
        return original_gemm(a, b, selected, scaling, global_scale)

    monkeypatch.setattr(torch.ops.sdc_cutlass, op_name, observed_op)
    monkeypatch.setattr(backend, "gemm", observed_gemm)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tokens = torch.randint(0, 128, (1, 32), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=tokens, labels=tokens)
    assert torch.isfinite(output.loss) and torch.isfinite(output.logits).all()
    output.loss.backward()
    optimizer.step()
    for name, param in model.named_parameters():
        assert param.dtype == torch.float32
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert torch.isfinite(param).all() and not torch.equal(param, before[name]), name
    # up_proj: X[32,64] @ W.T[64,128], dY[32,128] @ W[128,64],
    # dY.T[128,32] @ X[32,64]. Every bridge call must reach the native op.
    for shapes in [((32, 64), (64, 128)), ((32, 128), (128, 64)), ((128, 32), (32, 64))]:
        assert shapes in calls
    assert len(native_calls) == len(calls) and len(calls) >= 3 * 8


@pytest.mark.parametrize("precision", ["fp8", "nvfp4"])
def test_configured_public_trainer_resume(precision, tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, TrainerCallback

    from training.config import Config, DataConfig, GemmConfig, ModelConfig, TrainConfig, validate
    from training.data import prepare_data
    from training.runner import ResumeTrainer, train

    model_dir = tmp_path / "model"
    words = ["[UNK]", "[EOS]", "the", "model", "learns", "from", "text"]
    tokenizer_backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, eos_token="[EOS]",
                           unk_token="[UNK]", pad_token="[EOS]").save_pretrained(model_dir)
    model = _qwen()
    model.save_pretrained(model_dir)
    del model
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps({"text": "the model learns from text " * 8}) for _ in range(4)),
            encoding="utf-8",
        )
    config = validate(Config(
        name="cutlass_cuda_resume", model=ModelConfig(name_or_path=str(model_dir), gradient_checkpointing=True),
        data=DataConfig(source="jsonl", prepared_dir=str(tmp_path / "data"), block_size=32,
                        train_file=str(tmp_path / "train.jsonl"), validation_file=str(tmp_path / "validation.jsonl")),
        training=TrainConfig(output_dir=str(tmp_path / "run"), device="cuda", precision="bf16", max_steps=2,
                             batch_size=1, eval_batch_size=1, gradient_accumulation_steps=1,
                             warmup_steps=0, logging_steps=1, eval_steps=1, save_steps=1, save_total_limit=2),
        gemm=GemmConfig(backend="cutlass", precision=precision, scaling=SCALING[precision]),
    ))
    prepare_data(config)

    class StopAtOne(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True
                control.should_save = True
            return control

    # No explicit transform or CLI precision: public config selects CUTLASS.
    partial, metrics = train(config, offline=True, callbacks=[StopAtOne()])
    assert partial.state.global_step == 1 and math.isfinite(metrics["validation_loss"])
    assert partial.model.lm_head.weight is partial.model.model.embed_tokens.weight
    assert isinstance(partial.model.lm_head, CutlassLinear)
    assert partial.model.lm_head.precision == precision
    checkpoint = tmp_path / "run" / "checkpoint-1"
    weights = load_file(checkpoint / "model.safetensors")
    optimizer_state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    # This test created the file; RNG state includes Python and NumPy objects.
    rng_state = torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=False)
    assert optimizer_state["state"] and "cuda" in rng_state
    del partial
    observed = set()
    original_load = ResumeTrainer._load_from_checkpoint
    original_optimizer = ResumeTrainer._load_optimizer_and_scheduler
    original_rng = ResumeTrainer._load_rng_state

    def checked_load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        state = self.model.state_dict()
        for key, value in weights.items():
            torch.testing.assert_close(state[key].cpu(), value, rtol=0, atol=0)
        assert self.model.lm_head.weight is self.model.model.embed_tokens.weight
        observed.add("model")
        return result

    def checked_optimizer(self, *args, **kwargs):
        result = original_optimizer(self, *args, **kwargs)
        restored = self.optimizer.state_dict()["state"]
        assert set(restored) == set(optimizer_state["state"])
        for key, values in optimizer_state["state"].items():
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(restored[key][name].cpu(), value, rtol=0, atol=0)
                else:
                    assert restored[key][name] == value
        observed.add("optimizer")
        return result

    def checked_rng(self, *args, **kwargs):
        result = original_rng(self, *args, **kwargs)
        torch.testing.assert_close(torch.get_rng_state(), rng_state["cpu"], rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng_state["cuda"], rtol=0, atol=0)
        observed.add("rng")
        return result

    monkeypatch.setattr(ResumeTrainer, "_load_from_checkpoint", checked_load)
    monkeypatch.setattr(ResumeTrainer, "_load_optimizer_and_scheduler", checked_optimizer)
    monkeypatch.setattr(ResumeTrainer, "_load_rng_state", checked_rng)
    resumed, metrics = train(config, resume=str(checkpoint), offline=True)
    assert observed == {"model", "optimizer", "rng"}
    assert resumed.state.global_step == metrics["global_step"] == 2
    assert math.isfinite(metrics["validation_loss"])
    assert (tmp_path / "run" / "checkpoint-2" / "optimizer.pt").is_file()
    assert resumed.model.lm_head.weight is resumed.model.model.embed_tokens.weight
    losses = [entry["loss"] for entry in resumed.state.log_history if "loss" in entry]
    assert losses and all(math.isfinite(loss) for loss in losses)
    assert all(param.dtype == torch.float32 and torch.isfinite(param).all() for param in resumed.model.parameters())
