"""Real TE GEMMs and checkpointed Qwen training on SM120; no math substitutes."""

import importlib
import json
import math

import pytest
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Actual Transformer Engine FP8/FP4 training requires SM120 (RTX 5090)",
)


def test_locked_cudnn_with_torch_bf16_backward():
    from training.config import ROOT

    baseline = json.loads((ROOT / "environment/baseline.json").read_text())
    major, minor, patch = map(int, baseline["training_profiles"]["nvidia"]["cudnn"].split(".")[:3])
    assert torch.backends.cudnn.version() == major * 10000 + minor * 100 + patch
    x = torch.ones(2, 8, 16, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight = torch.ones(8, 8, 3, 3, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with torch.backends.cudnn.flags(enabled=True, deterministic=True, benchmark=False):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            out = nn.functional.conv2d(x, weight)
            out.float().sum().backward()
            torch.cuda.synchronize()
    assert any("cudnn_convolution" in event.key for event in profile.key_averages())
    torch.testing.assert_close(out, torch.full_like(out, 72), rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 392), rtol=0, atol=0)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def _qwen():
    config = Qwen3Config(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=64, max_position_embeddings=64, attention_dropout=0.0,
        use_cache=False, tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).cuda().train()


def _step(model, tokens):
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=tokens, labels=tokens)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, name
        assert param.grad is not None, name
        assert param.grad.dtype == torch.float32, name
        assert torch.isfinite(param.grad).all(), name
    optimizer.step()
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), name
        assert not torch.equal(param, before[name]), name


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_checkpointed_qwen_real_gemms_and_safetensors(precision, monkeypatch, tmp_path):
    # Import only after the hardware gate. A broken TE installation on SM120
    # must fail instead of turning the acceptance test into an import skip.
    from transformer_engine.common.recipe import Float8CurrentScaling, Format, NVFP4BlockScaling
    from transformer_engine.pytorch import Linear
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
    from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage

    from low_precision.adapter import install

    linear_module = importlib.import_module("transformer_engine.pytorch.module.linear")
    original_gemm = linear_module.general_gemm
    calls = []
    expected_type = Float8TensorStorage if precision == "fp8" else NVFP4TensorStorage

    def record_gemm(a, b, *args, **kwargs):
        # TE commit 1770259 uses TN for fprop, NN for dgrad, NT for wgrad.
        # Call the original kernel: observing operands never replaces the math.
        calls.append((kwargs.get("layout", "TN"), kwargs.get("grad", False),
                      type(a).__name__, type(b).__name__))
        assert kwargs["out_dtype"] == torch.bfloat16
        for operand in (a, b):
            assert isinstance(operand, expected_type), type(operand)
        if precision == "fp8":
            # TE's operand order is weight/input, weight/grad, input/grad.
            assert a._fp8_dtype == DType.kFloat8E4M3
            expected_b = DType.kFloat8E5M2 if kwargs.get("grad", False) else DType.kFloat8E4M3
            assert b._fp8_dtype == expected_b
        else:
            assert a._fp4_dtype == b._fp4_dtype == DType.kFloat4E2M1
        return original_gemm(a, b, *args, **kwargs)

    monkeypatch.setattr(linear_module, "general_gemm", record_gemm)
    torch.manual_seed(43)
    original = _qwen()
    linear_names = [name for name, module in original.named_modules() if isinstance(module, nn.Linear)]
    model = install(original, precision=precision)
    assert not any(isinstance(module, nn.Linear) for module in model.modules())
    te_layers = [module for module in model.modules() if isinstance(module, Linear)]
    assert len(te_layers) == len(linear_names)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.is_gradient_checkpointing
    tokens = torch.randint(0, 128, (2, 32), device="cuda")
    _step(model, tokens)
    for layout, grad in (("TN", False), ("NN", True), ("NT", True)):
        assert sum(call[:2] == (layout, grad) for call in calls) >= len(te_layers), calls
    for layer in te_layers:
        recipe = layer.fp8_meta["recipe"]
        if precision == "fp8":
            assert isinstance(recipe, Float8CurrentScaling)
            assert recipe.fp8_format == Format.HYBRID
        else:
            assert isinstance(recipe, NVFP4BlockScaling)
            assert recipe == NVFP4BlockScaling()

    # Stateless recipes have empty uint8 extra state in this TE revision.
    # Keep these keys in safetensors; the fixed adapter reconstructs the recipe.
    state = model.state_dict()
    extra_keys = [key for key in state if key.endswith("_extra_state")]
    assert len(extra_keys) == len(te_layers)
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert all(state[key].dtype == torch.uint8 and state[key].numel() == 0 for key in extra_keys)
    checkpoint = tmp_path / "model.safetensors"
    save_file({key: value.detach().cpu().contiguous().clone() for key, value in state.items()}, checkpoint)
    loaded = load_file(checkpoint)
    assert set(loaded) == set(state)
    for key in state:
        torch.testing.assert_close(loaded[key], state[key].cpu(), rtol=0, atol=0)
    restored = install(_qwen(), precision=precision)
    restored.load_state_dict(loaded, strict=True)
    for name, param in restored.named_parameters():
        torch.testing.assert_close(param, dict(model.named_parameters())[name], rtol=0, atol=0)
    restored.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    calls.clear()
    _step(restored, tokens)
    assert {call[:2] for call in calls} == {("TN", False), ("NN", True), ("NT", True)}


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_public_trainer_checkpoint_resume(precision, tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, TrainerCallback

    from low_precision.adapter import install, metadata
    from low_precision.train import PrecisionRecord
    from training.config import Config, DataConfig, ModelConfig, TrainConfig, validate
    from training.data import prepare_data
    from training.runner import ResumeTrainer, train

    model_dir = tmp_path / "model"
    words = ["[UNK]", "[EOS]", "the", "model", "learns", "from", "text"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="[EOS]", unk_token="[UNK]", pad_token="[EOS]",
    )
    tokenizer.save_pretrained(model_dir)
    model = _qwen()
    model.config.tie_word_embeddings = True
    model.tie_weights()
    model.config.eos_token_id = model.config.pad_token_id = 1
    model.save_pretrained(model_dir)
    del model
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps({"text": "the model learns from text " * 8}) for _ in range(4)),
            encoding="utf-8",
        )
    config = validate(Config(
        name="low_precision_cuda_resume",
        model=ModelConfig(name_or_path=str(model_dir), gradient_checkpointing=True),
        data=DataConfig(
            source="jsonl", prepared_dir=str(tmp_path / "data"), block_size=32,
            train_file=str(tmp_path / "train.jsonl"), validation_file=str(tmp_path / "validation.jsonl"),
        ),
        training=TrainConfig(
            output_dir=str(tmp_path / "run"), device="cuda", precision="bf16", max_steps=2,
            batch_size=1, eval_batch_size=1, gradient_accumulation_steps=1,
            warmup_steps=0, logging_steps=1, eval_steps=1, save_steps=1, save_total_limit=2,
        ),
    ))
    prepare_data(config)

    class StopAtOne(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True
                control.should_save = True
            return control

    transform = dict(model_transform=install, transform_id="te-gpu-test:v1",
                     transform_params={"precision": precision})
    partial, metrics = train(config, offline=True, callbacks=[PrecisionRecord(precision), StopAtOne()], **transform)
    assert partial.model.lm_head.weight is partial.model.model.embed_tokens.weight
    assert partial.state.global_step == 1
    assert math.isfinite(metrics["validation_loss"])
    checkpoint = tmp_path / "run" / "checkpoint-1"
    weights = load_file(checkpoint / "model.safetensors")
    extra_keys = [key for key in weights if key.endswith("_extra_state")]
    assert extra_keys and all(weights[key].dtype == torch.uint8 and weights[key].numel() == 0
                              for key in extra_keys)
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert all(record["low_precision"][key] == value for key, value in metadata(precision).items())
    optimizer_state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    # Files were generated by this test. RNG includes NumPy/Python objects.
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
    resumed, metrics = train(config, resume=str(checkpoint), offline=True,
                             callbacks=[PrecisionRecord(precision)], **transform)
    assert resumed.model.lm_head.weight is resumed.model.model.embed_tokens.weight
    assert observed == {"model", "optimizer", "rng"}
    assert resumed.state.global_step == metrics["global_step"] == 2
    assert math.isfinite(metrics["validation_loss"])
    assert (tmp_path / "run" / "checkpoint-2" / "optimizer.pt").is_file()
    losses = [entry["loss"] for entry in resumed.state.log_history if "loss" in entry]
    assert losses and all(math.isfinite(loss) for loss in losses)
    for key in extra_keys:
        layer = resumed.model.get_submodule(key.removesuffix("._extra_state"))
        assert layer.fp8_meta["recipe"].nvfp4() == (precision == "fp4")
        assert layer.fp8_meta["recipe"].float8_current_scaling() == (precision == "fp8")
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert all(record["low_precision"][key] == value for key, value in metadata(precision).items())
