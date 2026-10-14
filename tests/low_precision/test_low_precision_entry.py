"""Training entry and run records; these checks do not execute GPU kernels."""

import json
from types import SimpleNamespace

import pytest

from low_precision import train as entry
from low_precision.adapter import TE_VERSION
from training.config import Config, DataConfig, ModelConfig, TrainConfig


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_entry_selects_recipe_and_reuses_training(monkeypatch, precision):
    config = Config(
        "test", ModelConfig("model"), DataConfig("data/prepared"),
        TrainConfig("results/unused", device="cuda"),
    )
    monkeypatch.setattr(entry, "load_config", lambda _: config)
    monkeypatch.setattr(entry, "require_device", lambda _: None)
    monkeypatch.setattr(entry, "build_key", lambda _: "test-build")
    calls = []
    monkeypatch.setattr(entry, "train", lambda config, **kwargs: calls.append((config, kwargs)))
    monkeypatch.setattr("sys.argv", [
        "low_precision.train", "--precision", precision, "--output-dir", "results/selected",
        "--resume", "latest", "--offline",
    ])
    entry.main()
    actual, options = calls[0]
    assert actual.data is config.data
    assert actual.training.output_dir == "results/selected"
    assert actual.training.precision == "bf16"
    assert options["model_transform"] is entry.install
    assert options["transform_params"] == {"precision": precision}
    assert options["transform_id"] == f"transformer-engine:{precision}:test-build"
    assert options["resume"] == "latest" and options["offline"]
    assert options["callbacks"][0].precision == precision


@pytest.mark.parametrize("device,precision", [("npu", "bf16"), ("cuda", "fp32")])
def test_entry_rejects_inconsistent_outer_precision(monkeypatch, device, precision):
    config = Config(
        "test", ModelConfig("model"), DataConfig("data/prepared"),
        TrainConfig("results/unused", device=device, precision=precision),
    )
    monkeypatch.setattr(entry, "load_config", lambda _: config)
    monkeypatch.setattr("sys.argv", [
        "low_precision.train", "--precision", "fp8", "--output-dir", "results/selected",
    ])
    with pytest.raises(SystemExit, match="2"):
        entry.main()


def test_precision_record_preserves_resume_fields(tmp_path, monkeypatch):
    record = {"config_signature": "config", "model_transform": {"id": "adapter"}}
    filename = tmp_path / "run.json"
    filename.write_text(json.dumps(record))
    monkeypatch.setattr(entry.importlib.metadata, "version", lambda _: TE_VERSION)
    callback = entry.PrecisionRecord("fp4")
    args = SimpleNamespace(output_dir=str(tmp_path))
    callback.on_train_begin(args, SimpleNamespace(is_world_process_zero=True), None)
    actual = json.loads(filename.read_text())
    assert actual["config_signature"] == record["config_signature"]
    assert actual["model_transform"] == record["model_transform"]
    assert actual["low_precision"]["precision"] == "fp4"
    assert actual["low_precision"]["recipe"] == "NVFP4BlockScaling"
    assert actual["low_precision"]["transformer_engine"] == TE_VERSION


def test_build_key_changes_with_recipe_and_component_version(monkeypatch):
    monkeypatch.setattr(entry.importlib.metadata, "version", lambda _: TE_VERSION)
    fp8 = entry.build_key("fp8")
    assert fp8 != entry.build_key("fp4")
    monkeypatch.setattr(entry.importlib.metadata, "version", lambda _: "different-build")
    assert fp8 != entry.build_key("fp8")
