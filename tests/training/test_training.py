from dataclasses import replace
import json
import math

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, TrainerCallback

from training.config import Config, DataConfig, ModelConfig, TrainConfig, load_config, validate
from training.data import CausalCollator, load_tokenizer, pack_texts, prepare_data, read_prepared
from training.runner import train


@pytest.fixture
def config(tmp_path):
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
    with pytest.raises(ValueError, match="differs"):
        read_prepared(config)


def test_prepared_artifact_changes_are_detected(config):
    prepare_data(config)
    from pathlib import Path

    filename = Path(config.data.prepared_dir) / "train" / "state.json"
    filename.write_text(filename.read_text() + "\n")
    with pytest.raises(ValueError, match="files have changed"):
        read_prepared(config)


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
    with pytest.raises(ValueError, match="not empty"):
        train(config, offline=True)
    with pytest.raises(ValueError, match="already reached"):
        train(config, resume="latest", offline=True)


def test_resume_rejects_changed_optimizer_settings(config):
    prepare_data(config)
    train(config, offline=True, callbacks=[StopAtOne()])
    altered = replace(config, training=replace(config.training, learning_rate=0.002))
    with pytest.raises(ValueError, match="config_signature"):
        train(altered, resume="latest", offline=True)


def test_config_rejects_shared_splits_and_invalid_precision(config):
    with pytest.raises(ValueError, match="separate"):
        validate(replace(config, data=replace(config.data, validation_file=config.data.train_file)))
    with pytest.raises(ValueError, match="fp32"):
        validate(replace(config, training=replace(config.training, precision="fp16")))
    with pytest.raises(ValueError, match="integer"):
        validate(replace(config, training=replace(config.training, gradient_accumulation_steps=0)))


def test_public_configs_are_loadable():
    from training.config import ROOT

    for filename in (ROOT / "configs/training").glob("*.yaml"):
        load_config(filename)
