from dataclasses import replace
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, TrainerCallback
from transformers.modeling_utils import load_state_dict

from training.config import (
    Config,
    DataConfig,
    GemmConfig,
    ModelConfig,
    TrainConfig,
    load_config,
    run_signature,
    validate,
)
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


def test_offline_hub_models_load_from_cached_snapshot(config):
    import training.data as data
    import training.runner as runner

    local = config.model.name_or_path
    revision = "a" * 40
    task = replace(config, model=replace(config.model, name_or_path="test/model", revision=revision))
    with patch.object(data, "snapshot_download", return_value=local) as download:
        assert data.runtime_model_source(task, offline=True) == local
        download.assert_called_once_with(repo_id="test/model", revision=revision, local_files_only=True)
        download.reset_mock()
        assert data.runtime_model_source(task, offline=False) == "test/model"
        assert data.runtime_model_source(config, offline=True) == local
        download.assert_not_called()
        # Real local loading must work without passing the Hub id into either
        # Transformers loader, and must not change the logical model identity.
        assert data.load_tokenizer(task, offline=True).eos_token_id is not None
        model = runner._load_model(task, offline=True)
        assert next(model.parameters()).dtype == torch.float32
        assert data.model_source(task.model) == "test/model"


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


def test_interrupted_training_writes_failure_result(config):
    from transformers import Trainer

    prepare_data(config)

    def failing_loss(self, *args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    with patch.object(Trainer, "compute_loss", failing_loss):
        with pytest.raises(RuntimeError, match="out of memory"):
            train(config, offline=True)
    result = json.loads((Path(config.training.output_dir) / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["reason"] == "RuntimeError"
    assert result["message"] == "CUDA out of memory"


class StopAtOne(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_training_stop = True
            control.should_save = True
        return control


@pytest.mark.parametrize("wrapped", [False, True, "cutlass"])
@pytest.mark.parametrize("tied", [False, True])
def test_real_forward_backward_update_and_exact_resume(config, tmp_path, wrapped, tied, monkeypatch):
    if tied:
        model = Qwen3ForCausalLM.from_pretrained(config.model.name_or_path)
        model.config.tie_word_embeddings = True
        model.tie_weights()
        model.save_pretrained(config.model.name_or_path)
    prepare_data(config)
    original = Qwen3ForCausalLM.from_pretrained(config.model.name_or_path)
    transform = {
        "model_transform": wrap_gate_proj, "transform_id": "counting-linear:v1",
        "transform_params": {"projection_size": 8},
    } if wrapped else {}
    if wrapped == "cutlass":
        from cutlass_gemm import backend
        from cutlass_gemm.linear import install

        # Test the Trainer/autograd bridge independently of the GPU kernel.
        monkeypatch.setattr(backend, "mm", torch.mm)
        transform = {"model_transform": install, "transform_id": "cutlass-bridge:test"}
    full, full_metrics = train(config, offline=True, **transform)
    assert full_metrics["global_step"] == 3
    assert math.isfinite(full_metrics["validation_loss"])
    assert not torch.equal(original.model.layers[0].mlp.up_proj.weight, full.model.model.layers[0].mlp.up_proj.weight)
    resumed_config = replace(config, training=replace(config.training, output_dir=str(tmp_path / "resumed")))
    partial, _ = train(resumed_config, offline=True, callbacks=[StopAtOne()], **transform)
    assert partial.state.global_step == 1
    resumed, resumed_metrics = train(resumed_config, resume="latest", offline=True, **transform)
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
    assert record["resume_from"] == str(tmp_path / "resumed" / "checkpoint-1")
    assert "train_loss" not in resumed_metrics
    assert resumed_metrics["epoch"] == resumed.state.epoch
    with pytest.raises(ValueError, match="非空"):
        train(config, offline=True, **transform)
    with pytest.raises(ValueError, match="已经达到"):
        train(config, resume="latest", offline=True, **transform)


@pytest.fixture
def stateful_abft(monkeypatch):
    """Small lifecycle fixture, not a reproduction of an adaptive paper method."""
    import abft
    from abft.engine import execute
    import abft.methods as methods

    class History(methods.Classic):
        stateful = True
        compiled_policy = None

        def __init__(self):
            self.count = 0
            self.history = torch.empty(0)
            self.rng = torch.Generator().manual_seed(719)

        def prepare(self, operands, config, mm=None):
            state = super().prepare(operands, config, mm)
            generator = torch.Generator()
            generator.set_state(self.rng.get_state())
            state["sample"] = torch.rand(1, generator=generator)
            state["rng_next"] = generator.get_state()
            return state

        def commit(self, state, decision, context):
            assert "fault_hit" not in decision and "fault" not in context
            self.count += 1
            self.history = torch.cat((self.history, state["sample"]))
            self.rng.set_state(state["rng_next"])

        def state_dict(self):
            return {"count": self.count, "history": self.history, "rng": self.rng.get_state()}

        def load_state_dict(self, state):
            self.count, self.history = state["count"], state["history"]
            self.rng.set_state(state["rng"])

    monkeypatch.setattr(methods, "_METHODS", {**methods._METHODS, "test_history": History})

    def collect(operands, weights, fault):
        assert fault[0] == -1
        a, b = operands.decode_block(0, operands.k)
        result = a @ b
        return result, (weights @ result).unsqueeze(1), torch.zeros(1, dtype=torch.int32)

    def cpu_execute(*args, **kwargs):
        return execute(*args, **kwargs, reference_mm=torch.mm, collector_fn=collect)

    monkeypatch.setattr(abft, "execute", cpu_execute)
    return {"method": "test_history", "atol": 1000., "rtol": 0.}


def test_stateful_abft_training_restores_before_first_call_and_matches_full_run(config, tmp_path, stateful_abft):
    from cutlass_gemm.linear import install, protection_sessions

    prepare_data(config)
    transform = {"model_transform": install, "transform_id": "stateful-abft:test",
                 "transform_params": {"precision": "fp32", "abft_config": stateful_abft}}
    full, _ = train(config, offline=True, **transform)
    resumed_config = replace(config, training=replace(config.training, output_dir=str(tmp_path / "resumed-history")))
    partial, _ = train(resumed_config, offline=True, callbacks=[StopAtOne()], **transform)
    checkpoint = tmp_path / "resumed-history" / "checkpoint-1"
    snapshot = torch.load(checkpoint / "abft_state.pt", weights_only=True)
    saved_session = snapshot["controllers"][0]["session"]
    assert saved_session["call_index"] > 0 and saved_session["instances"]
    # The runner's final evaluation advances the stopped model after saving;
    # compare resume against the checkpoint, not that post-save evaluation.
    assert protection_sessions(partial.model)[0]["session"].call_index > saved_session["call_index"]

    class CheckRestored(TrainerCallback):
        checked = False

        def on_init_end(self, args, state, control, model, **kwargs):
            restored = protection_sessions(model)[0]["session"].state_dict()
            assert restored["call_index"] == saved_session["call_index"]
            for before, after in zip(saved_session["instances"], restored["instances"], strict=True):
                assert before["scope"] == after["scope"]
                assert before["state"]["count"] == after["state"]["count"]
                assert torch.equal(before["state"]["rng"], after["state"]["rng"])
                assert torch.equal(before["state"]["history"], after["state"]["history"])
            self.checked = True

    check = CheckRestored()
    resumed, _ = train(resumed_config, resume="latest", offline=True, callbacks=[check], **transform)
    assert check.checked
    for name, tensor in full.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[name]), name
    full_state = protection_sessions(full.model)[0]["session"].state_dict()
    resumed_state = protection_sessions(resumed.model)[0]["session"].state_dict()
    assert full_state["call_index"] == resumed_state["call_index"]
    for before, after in zip(full_state["instances"], resumed_state["instances"], strict=True):
        assert before["scope"] == after["scope"]
        assert before["state"]["count"] == after["state"]["count"]
        assert torch.equal(before["state"]["rng"], after["state"]["rng"])
        assert torch.equal(before["state"]["history"], after["state"]["history"])
    assert not any("history" in name or "session" in name for name in resumed.model.state_dict())
    assert json.loads((tmp_path / "resumed-history" / "run.json").read_text())["abft_session"]["status"] == "restored"


def test_algorithm_state_missing_file_and_layer_schema_validation(tmp_path, stateful_abft):
    from cutlass_gemm.linear import install
    from training.runner import AlgorithmState

    stateful = AlgorithmState(install(torch.nn.Sequential(torch.nn.Linear(2, 2)), abft_config=stateful_abft))
    with pytest.raises(ValueError, match="requires abft_state.pt"):
        stateful.restore(tmp_path, device="cpu")
    stateless = AlgorithmState(install(torch.nn.Linear(2, 2), abft_config="classic"))
    stateless.restore(tmp_path, device="cpu")
    assert stateless.status == "legacy_stateless_no_history"
    assert stateless.sessions[0]["session"].call_index == 0
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    stateful.on_save(SimpleNamespace(output_dir=str(tmp_path)),
                     SimpleNamespace(is_world_process_zero=True, global_step=1), None)
    stateful.restore(checkpoint, device="cpu")
    payload = torch.load(checkpoint / "abft_state.pt", weights_only=True)
    payload["controllers"][0]["layers"] = ["wrong.layer"]
    torch.save(payload, checkpoint / "abft_state.pt")
    with pytest.raises(ValueError, match="layer layout"):
        stateful.restore(checkpoint, device="cpu")
    payload["controllers"][0]["layers"] = ["0"]
    payload["schema_version"] = True
    torch.save(payload, checkpoint / "abft_state.pt")
    with pytest.raises(ValueError, match="schema"):
        stateful.restore(checkpoint, device="cpu")


class CountingLinear(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x)


def wrap_gate_proj(model, projection_size=8):
    for layer in model.model.layers:
        layer.mlp.gate_proj = CountingLinear(layer.mlp.gate_proj)
        layer.mlp.gate_proj.register_buffer("projection", torch.randn(projection_size))
    return model


def test_model_transform_is_the_trained_model(config):
    prepare_data(config)
    original = Qwen3ForCausalLM.from_pretrained(config.model.name_or_path)
    wrapped = []

    def wrap_gate_proj(model):
        for layer in model.model.layers:
            layer.mlp.gate_proj = CountingLinear(layer.mlp.gate_proj)
            wrapped.append(layer.mlp.gate_proj)
        return model

    trainer, metrics = train(config, offline=True, model_transform=wrap_gate_proj, transform_id="counting-linear:v1")
    assert metrics["global_step"] == 3
    assert trainer.model.model.layers[0].mlp.gate_proj is wrapped[0]
    assert wrapped[0].calls > 0
    assert not torch.equal(original.model.layers[0].mlp.gate_proj.weight, wrapped[0].linear.weight)
    record = json.loads((Path(config.training.output_dir) / "run.json").read_text())
    assert record["model_transform"] == {"id": "counting-linear:v1", "parameters": {}}


@pytest.fixture
def transformed_run(config):
    prepare_data(config)
    train(config, offline=True, callbacks=[StopAtOne()], model_transform=wrap_gate_proj,
          transform_id="counting-linear:v1", transform_params={"projection_size": 8})
    return config


@pytest.mark.parametrize("change", ["omitted", "id", "parameters"])
def test_resume_rejects_changed_transform_before_touching_run(transformed_run, change):
    config = transformed_run
    filename = Path(config.training.output_dir) / "run.json"
    original = filename.read_bytes()
    assert json.loads(original)["model_transform"] == {
        "id": "counting-linear:v1", "parameters": {"projection_size": 8},
    }
    kwargs = {} if change == "omitted" else {
        "model_transform": wrap_gate_proj,
        "transform_id": "counting-linear:v2" if change == "id" else "counting-linear:v1",
        "transform_params": {"projection_size": 16 if change == "parameters" else 8},
    }
    with pytest.raises(ValueError, match="续训设置不一致：model_transform"):
        train(config, resume="latest", offline=True, **kwargs)
    assert filename.read_bytes() == original


def test_resume_rejects_transform_added_to_plain_run(config):
    prepare_data(config)
    train(config, offline=True, callbacks=[StopAtOne()])
    with pytest.raises(ValueError, match="续训设置不一致：model_transform"):
        train(config, resume="latest", offline=True, model_transform=wrap_gate_proj,
              transform_id="counting-linear:v1")


def test_resume_rejects_missing_transform_record(transformed_run):
    filename = Path(transformed_run.training.output_dir) / "run.json"
    record = json.loads(filename.read_text())
    record.pop("model_transform")
    filename.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="缺少 model_transform"):
        train(transformed_run, resume="latest", offline=True)


def test_resume_rejects_weight_keys_even_when_transform_id_matches(transformed_run):
    filename = Path(transformed_run.training.output_dir) / "run.json"
    original = filename.read_bytes()
    with pytest.raises(ValueError, match="检查点权重与当前模型不一致"):
        train(transformed_run, resume="latest", offline=True,
              model_transform=lambda model, **kwargs: model,
              transform_id="counting-linear:v1", transform_params={"projection_size": 8})
    assert filename.read_bytes() == original


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_resume_stops_on_corrupt_checkpoint_keys(transformed_run, change):
    output = Path(transformed_run.training.output_dir)
    original = (output / "run.json").read_bytes()
    checkpoint = output / "checkpoint-1"
    weights = load_state_dict(str(checkpoint / "model.safetensors"))
    if change == "missing":
        weights.pop("model.layers.0.mlp.gate_proj.linear.weight")
    else:
        weights["unexpected.weight"] = torch.ones(1)
    torch.save(weights, checkpoint / "pytorch_model.bin")
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="检查点权重与当前模型不一致"):
        train(transformed_run, resume="latest", offline=True, model_transform=wrap_gate_proj,
              transform_id="counting-linear:v1", transform_params={"projection_size": 8})
    assert (output / "run.json").read_bytes() == original


@pytest.mark.parametrize("kwargs, message", [
    ({"model_transform": wrap_gate_proj}, "transform_id"),
    ({"transform_id": "wrapper:v1"}, "model_transform"),
    ({"model_transform": wrap_gate_proj, "transform_id": "wrapper:v1",
      "transform_params": {"threshold": float("nan")}}, "JSON"),
])
def test_transform_settings_fail_before_loading_data(config, kwargs, message):
    with pytest.raises(ValueError, match=message):
        train(config, offline=True, **kwargs)
    assert not Path(config.training.output_dir).exists()


@pytest.mark.parametrize("full_determinism", [False, True])
def test_transform_initialization_uses_training_seed(config, tmp_path, full_determinism):
    prepare_data(config)
    draws = []
    weights = []

    def randomized(model):
        draws.append((random.random(), float(np.random.random()), torch.randn(8).tolist()))
        return wrap_gate_proj(model)

    for index, ambient_seed in enumerate((101, 202)):
        task = replace(config, training=replace(
            config.training, output_dir=str(tmp_path / f"seed-{index}"), full_determinism=full_determinism,
        ))
        random.seed(ambient_seed)
        np.random.seed(ambient_seed)
        torch.manual_seed(ambient_seed)
        trainer, _ = train(task, offline=True, callbacks=[StopAtOne()],
                           model_transform=randomized, transform_id="randomized:v1")
        weights.append({name: tensor.clone() for name, tensor in trainer.model.state_dict().items()})
    assert draws[0] == draws[1]
    assert all(torch.equal(weights[0][name], tensor) for name, tensor in weights[1].items())


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


@pytest.mark.parametrize(("precision", "scaling", "outer"), [
    ("fp32", "none", "fp32"), ("bf16", "none", "bf16"),
    ("fp8", "tensorwise", "bf16"), ("nvfp4", "block16", "bf16"),
])
def test_cutlass_config_precision_contract(config, precision, scaling, outer):
    task = replace(config, gemm=GemmConfig("cutlass", precision, scaling),
                   training=replace(config.training, device="cuda", precision=outer))
    assert validate(task) == task
    with pytest.raises(ValueError, match="training.precision"):
        validate(replace(task, training=replace(task.training, precision="fp16")))
    with pytest.raises(ValueError, match="CUDA"):
        validate(replace(task, training=replace(task.training, device="npu")))


@pytest.mark.parametrize("gemm", [
    GemmConfig("transformer_engine"), GemmConfig("cutlass", "fp8", "none"),
    GemmConfig("cutlass", "fp4", "block16"), GemmConfig("torch", "fp8", "tensorwise"),
    GemmConfig(abft="unimplemented"),
    GemmConfig(global_scale=0), GemmConfig(global_scale=float("inf")),
    GemmConfig(global_scale=True), GemmConfig(global_scale="auto"),
])
def test_config_rejects_unsupported_gemm(config, gemm):
    with pytest.raises(ValueError):
        validate(replace(config, gemm=gemm))


def test_runtime_options_do_not_change_resume_signature(config):
    operational = replace(config, training=replace(
        config.training, offline=True, resume="latest", progress_bar=not config.training.progress_bar,
    ))
    assert run_signature(config) == run_signature(operational)
    assert run_signature(config) != run_signature(replace(config, gemm=GemmConfig(precision="bf16")))
    scaled = replace(config, gemm=replace(config.gemm, global_scale=2.0))
    assert validate(scaled) == scaled
    assert run_signature(config) != run_signature(scaled)


@pytest.mark.parametrize("progress_bar", [True, False])
def test_native_progress_and_stage_output_preserve_training_results(config, capsys, progress_bar):
    from transformers.trainer_callback import PrinterCallback, ProgressCallback

    prepare_data(config)
    task = replace(config, training=replace(
        config.training, max_steps=1, gradient_accumulation_steps=1, offline=True,
        progress_bar=progress_bar, save_strategy="no", eval_strategy="no",
    ))
    trainer, metrics = train(task)
    expected = ProgressCallback if progress_bar else PrinterCallback
    assert any(isinstance(callback, expected) for callback in trainer.callback_handler.callbacks)
    assert trainer.args.disable_tqdm is not progress_bar
    captured = capsys.readouterr()
    stages = [
        "开始读取并校验", "数据读取完成", "开始加载分词器", "分词器加载完成",
        "开始加载模型", "模型加载完成", "GEMM：backend=torch", "开始训练",
        "开始最终验证", "最终验证完成", "结果目录",
    ]
    positions = [captured.out.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "'loss':" in captured.out and "'learning_rate':" in captured.out
    output = Path(task.training.output_dir)
    assert json.loads((output / "result.json").read_text())["validation_loss"] == metrics["validation_loss"]
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert any("loss" in record and "learning_rate" in record for record in records)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_torch_gemm_precision_inherits_or_matches_training(config, precision):
    task = replace(config, gemm=GemmConfig(),
                   training=replace(config.training, device="npu", precision=precision))
    assert validate(task).gemm.precision == precision
    explicit = replace(task, gemm=GemmConfig(precision=precision))
    assert validate(explicit) == explicit
    mismatch = "bf16" if precision == "fp32" else "fp32"
    with pytest.raises(ValueError, match="gemm.precision.*training.precision"):
        validate(replace(task, gemm=GemmConfig(precision=mismatch)))


@pytest.mark.parametrize(("section", "key", "value"), [
    ("training", "optimizer", "sgd"), ("training", "lr_scheduler_type", "linear"),
    ("training", "tf32", True), ("training", "save_strategy", "no"),
    ("training", "eval_strategy", "no"),
    ("model", "gradient_checkpointing_use_reentrant", True),
])
def test_training_algorithm_choices_change_resume_signature(config, section, key, value):
    task = replace(config, **{section: replace(getattr(config, section), **{key: value})})
    assert validate(task) == task
    assert run_signature(task) != run_signature(config)


def test_training_algorithm_choices_reach_trainer(config, monkeypatch):
    import training.runner as runner

    task = replace(config,
                   model=replace(config.model, gradient_checkpointing_use_reentrant=True),
                   training=replace(config.training, optimizer="sgd", lr_scheduler_type="linear",
                                    tf32=True, save_strategy="no", eval_strategy="no"))
    captured = {}

    def arguments(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(device=torch.device("cpu"), world_size=1, n_gpu=0)

    monkeypatch.setattr(runner, "TrainingArguments", arguments)
    runner.training_arguments(task)
    assert captured["optim"] == "sgd"
    assert captured["lr_scheduler_type"] == "linear"
    assert captured["tf32"] is True
    assert captured["save_strategy"] == captured["eval_strategy"] == "no"
    assert captured["gradient_checkpointing_kwargs"] == {"use_reentrant": True}


@pytest.mark.parametrize(("key", "value"), [
    ("optimizer", ""), ("lr_scheduler_type", None), ("tf32", "true"),
    ("progress_bar", "true"),
    ("save_strategy", "epoch"), ("eval_strategy", "epoch"),
])
def test_training_algorithm_choices_reject_invalid_values(config, key, value):
    with pytest.raises(ValueError, match=key):
        validate(replace(config, training=replace(config.training, **{key: value})))


def test_config_alone_controls_offline_and_resume(config):
    prepare_data(config)
    task = replace(config, training=replace(config.training, offline=True))
    train(task, callbacks=[StopAtOne()])
    task = replace(task, training=replace(task.training, resume="latest"))
    _, metrics = train(task)
    assert metrics["global_step"] == 3
    resolved = json.loads((Path(task.training.output_dir) / "config.resolved.json").read_text())
    assert resolved["training"]["offline"] is True
    assert resolved["training"]["resume"] == "latest"


@pytest.mark.parametrize("strategy", ["no", "steps"])
def test_config_controls_final_checkpoint(config, strategy):
    prepare_data(config)
    task = replace(config, training=replace(
        config.training, offline=True, save_strategy=strategy,
        save_steps=100, eval_strategy="no",
    ))
    _, metrics = train(task)
    assert metrics["status"] == "completed" and metrics["global_step"] == 3
    checkpoints = sorted(Path(task.training.output_dir).glob("checkpoint-*"))
    if strategy == "no":
        assert checkpoints == []
    else:
        assert [path.name for path in checkpoints] == ["checkpoint-3"]
        assert (checkpoints[0] / "optimizer.pt").is_file()


def test_config_cutlass_selects_transform_before_data_loading(config, monkeypatch):
    from cutlass_gemm import backend
    from cutlass_gemm.linear import install
    import training.runner as runner

    task = replace(config, gemm=GemmConfig("cutlass", "fp8", "tensorwise", global_scale=2.0),
                   training=replace(config.training, device="cuda", precision="bf16"))
    monkeypatch.setattr(backend, "build_key", lambda: "test-build")
    captured = []
    original = runner.transform_record

    def capture(transform, identity, parameters):
        captured.append((transform, identity, parameters))
        return original(transform, identity, parameters)

    monkeypatch.setattr(runner, "transform_record", capture)
    with pytest.raises(ValueError, match="请先执行 prepare"):
        train(task)
    assert captured == [(install, "cutlass:test-build", {
        "precision": "fp8", "scaling": "tensorwise", "global_scale": 2.0,
    })]
    with pytest.raises(ValueError, match="model_transform"):
        train(task, model_transform=wrap_gate_proj, transform_id="custom")


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
        output_dir=shared.training.output_dir.replace("{device}", device),
    )


def test_abft_fault_config_and_signatures(config):
    from abft.config import AbftConfig, FaultConfig

    base = replace(config, gemm=GemmConfig("cutlass", "fp32"),
                   training=replace(config.training, device="cuda"))
    protected = validate(replace(base, gemm=replace(base.gemm,
                         abft={"method": "classic", "on_error": "report"},
                         fault={"location": "accumulator", "role": "wgrad"},
                         report_max_records=4)))
    assert isinstance(protected.gemm.abft, AbftConfig)
    assert isinstance(protected.gemm.fault, FaultConfig)
    assert run_signature(validate(base)) != run_signature(protected)
    assert protected.to_dict()["gemm"]["fault"]["role"] == "wgrad"
    with pytest.raises(ValueError, match="resume"):
        validate(replace(protected, training=replace(protected.training, resume="latest")))
    with pytest.raises(ValueError):
        validate(replace(base, gemm=replace(base.gemm, report_max_records=-1)))
    with pytest.raises(ValueError):
        validate(replace(base, gemm=replace(base.gemm, abft={"typo": True})))


def test_runner_passes_protection_configuration(config, monkeypatch, tmp_path):
    from cutlass_gemm import backend
    import training.runner as runner

    task = replace(config, gemm=GemmConfig("cutlass", "fp32", abft={"method": "classic"},
                   fault={"location": "accumulator", "bit": 20}, report_max_records=3,
                   report_path=str(tmp_path / "report.jsonl")),
                   training=replace(config.training, device="cuda"))
    captured = []
    monkeypatch.setattr(backend, "build_key", lambda: "test-build")

    def capture(transform, identifier, parameters):
        captured.append(parameters)
        raise RuntimeError("captured before loading data")

    monkeypatch.setattr(runner, "transform_record", capture)
    with pytest.raises(RuntimeError, match="captured"):
        train(task)
    assert captured[0]["abft_config"]["method"] == "classic"
    assert captured[0]["fault_config"]["bit"] == 20
    assert captured[0]["report_max_records"] == 3
    assert captured[0]["report_path"] == str(tmp_path / "report.jsonl")
