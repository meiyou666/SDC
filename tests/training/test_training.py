from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, TrainerCallback

from training.config import Config, DataConfig, ModelConfig, TrainConfig, load_config, validate
from training.data import CausalCollator, load_tokenizer, pack_texts, prepare_data, read_prepared
from training.metrics import JsonMetrics, NonFiniteLossError, write_json
from training.runner import resolve_runtime_config, train


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    model_dir = tmp_path / "model"
    words = ["[UNK]", "[EOS]", "[PAD]", "the", "model", "learns", "from", "text", "training",
             "uses", "gradients", "and", "validation", "measures", "loss", ".", "data", "is", "fixed"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", unk_token="[UNK]", pad_token="[EOS]")
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
        attention_dropout=0.1, eos_token_id=1, pad_token_id=1,
    ))
    model.save_pretrained(model_dir)
    for split, text in (("train", "the model learns from text . training uses gradients ."),
                        ("validation", "validation measures loss . data is fixed .")):
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps({"text": text}) for _ in range(13)), encoding="utf-8",
        )
    return validate(Config(
        name="test_training",
        model=ModelConfig(name_or_path=str(model_dir), gradient_checkpointing=True),
        data=DataConfig(
            source="jsonl", prepared_dir=str(tmp_path / "data"), block_size=8,
            train_file=str(tmp_path / "train.jsonl"), validation_file=str(tmp_path / "validation.jsonl"),
        ),
        training=TrainConfig(
            output_dir=str(tmp_path / "run"), device="cpu", precision="fp32", max_steps=3,
            warmup_steps=0, gradient_accumulation_steps=2, eval_steps=1, save_steps=1,
            logging_steps=1, cpu_threads=1, full_determinism=True, save_total_limit=2,
        ),
    ))


def test_packing_preserves_tokens_and_eos_across_tokenizer_batches(config):
    tokenizer = load_tokenizer(config, offline=True)
    texts = ["the model learns .", "", "training uses gradients ."] * 90
    expected = []
    for text in texts:
        if text.strip():
            expected.extend(tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id])
    stats = {}
    blocks = list(pack_texts(texts, tokenizer, 13, stats))
    actual = [token for block in blocks for token in block["input_ids"]]
    assert actual == expected[:len(expected) // 13 * 13]
    assert stats["tokens"] == len(expected)
    assert stats["dropped_tokens"] == len(expected) % 13
    assert stats["documents"] == 270
    batch = CausalCollator()(blocks[:2])
    assert torch.equal(batch["labels"], batch["input_ids"])
    assert (batch["labels"] == tokenizer.eos_token_id).any()
    assert batch["attention_mask"].all()


def test_prepared_data_is_reusable_and_detects_source_changes(config):
    first = prepare_data(config)
    assert prepare_data(config) == first
    datasets, manifest = read_prepared(config)
    for split in ("train", "validation"):
        assert len(datasets[split]) == manifest["statistics"][split]["blocks"] > 0
    with open(config.data.train_file, "a") as handle:
        handle.write('\n{"text": "extra data"}\n')
    with pytest.raises(ValueError, match="与当前配置不一致"):
        read_prepared(config)


def test_prepare_recovers_empty_directory(config):
    Path(config.data.prepared_dir).mkdir()
    manifest = prepare_data(config)
    assert manifest["statistics"]["train"]["blocks"] > 0


@pytest.mark.parametrize("content", [None, "{", "{}"])
def test_incomplete_prepared_directory_has_actionable_error(config, content):
    directory = Path(config.data.prepared_dir)
    directory.mkdir()
    residual = directory / "partial.arrow"
    residual.write_bytes(b"unfinished")
    if content is not None:
        (directory / "manifest.json").write_text(content)
    with pytest.raises(ValueError, match="不完整"):
        prepare_data(config)
    assert residual.read_bytes() == b"unfinished"


def test_unprepared_and_incomplete_data_are_distinct(config):
    with pytest.raises(ValueError, match="请先执行 prepare"):
        read_prepared(config)
    Path(config.data.prepared_dir).mkdir()
    with pytest.raises(ValueError, match="不完整"):
        read_prepared(config)


@pytest.mark.parametrize("change", ["modify", "add", "delete"])
def test_prepared_file_changes_are_detected(config, change):
    """Every read re-hashes the snapshot, so edits cannot slip into training."""
    prepare_data(config)
    read_prepared(config)
    filename = Path(config.data.prepared_dir) / "train/state.json"
    if change == "modify":
        value = filename.read_bytes()
        filename.write_bytes(b" " + value[1:])
    elif change == "add":
        (filename.parent / "unexpected.arrow").write_bytes(b"extra")
    else:
        filename.unlink()
    with pytest.raises(ValueError, match="文件发生变化"):
        read_prepared(config)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("world_zero", [True, False])
def test_nonfinite_loss_is_explicit_on_every_rank(tmp_path, value, world_zero):
    callback = JsonMetrics(tmp_path, "test")
    state = SimpleNamespace(is_world_process_zero=world_zero, global_step=3)
    with pytest.raises(NonFiniteLossError, match="loss 非有限值"):
        callback.on_log(None, state, None, logs={"loss": value})
    if world_zero:
        record = json.loads((tmp_path / "metrics.jsonl").read_text())
        assert record["loss"] is None
        assert record["nonfinite_metrics"] == {"loss": str(value)}
    else:
        assert not (tmp_path / "metrics.jsonl").exists()


def test_nonfinite_nonloss_metrics_are_recorded_without_stopping(tmp_path):
    callback = JsonMetrics(tmp_path, "test")
    callback.on_log(None, SimpleNamespace(is_world_process_zero=True, global_step=3), None,
                    logs={"loss": 1.0, "grad_norm": float("inf")})
    result = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert result["loss"] == 1.0 and result["grad_norm"] is None
    assert result["nonfinite_metrics"] == {"grad_norm": "inf"}


def test_json_writer_preserves_nonfinite_metric_names(tmp_path):
    filename = tmp_path / "result.json"
    write_json(filename, {"loss": float("nan"), "nested": {"values": [1.0, float("inf")]}})
    result = json.loads(filename.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert result["loss"] is None
    assert result["nested"]["values"] == [1.0, None]
    assert result["nonfinite_metrics"] == {"loss": "nan", "nested.values[1]": "inf"}


def test_divergent_training_writes_failure_result(config):
    from transformers import Trainer

    prepare_data(config)
    original = Trainer.compute_loss

    def divergent_loss(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if isinstance(value, tuple):
            return value[0] * float("nan"), value[1]
        return value * float("nan")

    with patch.object(Trainer, "compute_loss", divergent_loss):
        with pytest.raises(NonFiniteLossError, match="loss 非有限值"):
            train(config, offline=True)
    result = json.loads((Path(config.training.output_dir) / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["reason"] == "nonfinite_loss"
    assert result["loss"] is None
    assert result["global_step"] == 1


class StopAtOne(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_training_stop = True
            control.should_save = True
        return control


def test_real_forward_backward_update_and_exact_resume(config, tmp_path):
    prepare_data(config)
    original = Qwen3ForCausalLM.from_pretrained(config.model.name_or_path)
    full, full_metrics = train(config, offline=True)
    assert full_metrics["global_step"] == 3
    assert math.isfinite(full_metrics["validation_loss"])
    assert any(not torch.equal(original.state_dict()[name], value) for name, value in full.model.state_dict().items())
    resumed_config = replace(config, training=replace(config.training, output_dir=str(tmp_path / "resumed")))
    partial, _ = train(resumed_config, offline=True, callbacks=[StopAtOne()])
    assert partial.state.global_step == 1
    resumed, resumed_metrics = train(resumed_config, resume="latest", offline=True)
    assert resumed_metrics["global_step"] == 3
    assert resumed_metrics["validation_loss"] == full_metrics["validation_loss"]
    for name, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
    assert full.lr_scheduler.state_dict() == resumed.lr_scheduler.state_dict()
    for key, state in full.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, resumed.optimizer.state_dict()["state"][key][name])
    assert (tmp_path / "resumed" / "result.json").is_file()
    record = json.loads((tmp_path / "resumed" / "run.json").read_text())
    assert record["training_budget"] == {"unit": "optimizer_updates", "max_steps": 3, "repeat_dataset": True}
    assert resumed_metrics["epoch"] == resumed.state.epoch
    with pytest.raises(ValueError, match="非空"):
        train(config, offline=True)
    with pytest.raises(ValueError, match="已经达到"):
        train(config, resume="latest", offline=True)


class CountingLinear(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x)


def test_model_transform_is_the_trained_model(config):
    prepare_data(config)
    original = Qwen3ForCausalLM.from_pretrained(config.model.name_or_path)
    wrapped = []

    def wrap_gate_proj(model):
        for layer in model.model.layers:
            layer.mlp.gate_proj = CountingLinear(layer.mlp.gate_proj)
            wrapped.append(layer.mlp.gate_proj)
        return model

    trainer, metrics = train(config, offline=True, model_transform=wrap_gate_proj)
    assert metrics["global_step"] == 3
    assert trainer.model.model.layers[0].mlp.gate_proj is wrapped[0]
    assert wrapped[0].calls > 0
    assert not torch.equal(original.model.layers[0].mlp.gate_proj.weight, wrapped[0].linear.weight)


def test_resume_rejects_changed_optimizer_settings(config):
    prepare_data(config)
    train(config, offline=True, callbacks=[StopAtOne()])
    altered = replace(config, training=replace(config.training, learning_rate=0.002))
    with pytest.raises(ValueError, match="config_signature"):
        train(altered, resume="latest", offline=True)


def test_config_rejects_shared_splits_and_invalid_precision(config):
    with pytest.raises(ValueError, match="必须不同"):
        validate(replace(config, data=replace(config.data, validation_file=config.data.train_file)))
    with pytest.raises(ValueError, match="fp32"):
        validate(replace(config, training=replace(config.training, precision="fp16")))
    with pytest.raises(ValueError, match="整数"):
        validate(replace(config, training=replace(config.training, gradient_accumulation_steps=0)))


def test_public_configs_are_loadable():
    from training.config import ROOT

    for filename in (ROOT / "configs/training").glob("*.yaml"):
        load_config(filename)


@pytest.mark.parametrize(("runtime", "device"), [("nvidia", "cuda"), ("ascend", "npu")])
def test_qwen_config_changes_only_device_and_output(runtime, device, monkeypatch):
    from training.config import ROOT

    shared = load_config(ROOT / "configs/training/qwen3_06b.yaml")
    monkeypatch.setenv("SDC_RUNTIME", runtime)
    resolved = resolve_runtime_config(shared)
    assert resolved.model == shared.model
    assert resolved.data == shared.data
    assert resolved.training == replace(
        shared.training, device=device,
        output_dir=shared.training.output_dir.replace("{backend}", device),
    )
