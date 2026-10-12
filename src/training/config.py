"""Read the owner-managed training configuration."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    revision: str | None = None
    attention: str = "eager"
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class DataConfig:
    prepared_dir: str
    source: str = "huggingface"
    name: str | None = None
    subset: str | None = None
    revision: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    text_field: str = "text"
    train_file: str | None = None
    validation_file: str | None = None
    block_size: int = 512
    max_train_documents: int | None = None
    max_validation_documents: int | None = None


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
    device: str = "npu"
    precision: str = "bf16"
    seed: int = 2026
    max_steps: int = 1000
    batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    cpu_threads: int = 4
    full_determinism: bool = False


@dataclass(frozen=True)
class Config:
    name: str
    model: ModelConfig
    data: DataConfig
    training: TrainConfig

    def to_dict(self):
        return asdict(self)


def path(value):
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (ROOT / result).resolve()


def model_source(config):
    local = path(config.name_or_path)
    return str(local) if local.is_dir() else config.name_or_path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(filename):
    result = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _positive(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def validate(config):
    m, d, t = config.model, config.data, config.training
    if not isinstance(config.name, str) or not config.name.strip():
        raise ValueError("name must be a nonempty string.")
    if not isinstance(m.name_or_path, str) or not m.name_or_path.strip():
        raise ValueError("model.name_or_path is required.")
    if not path(m.name_or_path).is_dir() and not re.fullmatch(r"[0-9a-f]{40}", m.revision or ""):
        raise ValueError("Pin model.revision to the Hugging Face commit SHA.")
    if m.attention not in {"eager", "sdpa"}:
        raise ValueError("model.attention must be eager or sdpa.")
    if d.source not in {"huggingface", "jsonl"}:
        raise ValueError("data.source must be huggingface or jsonl.")
    if d.source == "huggingface":
        if not d.name or not re.fullmatch(r"[0-9a-f]{40}", d.revision or ""):
            raise ValueError("Set data.name and an immutable data.revision.")
        if d.train_split == d.validation_split:
            raise ValueError("Use separate training and validation splits.")
    else:
        if not d.train_file or not d.validation_file:
            raise ValueError("Set data.train_file and data.validation_file.")
        if path(d.train_file) == path(d.validation_file):
            raise ValueError("Use separate training and validation files.")
    _positive(d.block_size, "data.block_size", 2)
    for key in ("max_train_documents", "max_validation_documents"):
        if getattr(d, key) is not None:
            _positive(getattr(d, key), f"data.{key}")
    if t.device not in {"cpu", "cuda", "npu", "accelerator"}:
        raise ValueError("training.device must be accelerator, cuda, npu or cpu (unit tests).")
    if t.precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("training.precision must be fp32, bf16 or fp16.")
    if t.device == "cpu" and t.precision != "fp32":
        raise ValueError("Use fp32 for the shared CPU check.")
    for key in ("max_steps", "batch_size", "eval_batch_size", "gradient_accumulation_steps",
                "logging_steps", "eval_steps", "save_steps", "save_total_limit", "cpu_threads"):
        _positive(getattr(t, key), f"training.{key}")
    for key in ("seed", "warmup_steps", "dataloader_num_workers"):
        _positive(getattr(t, key), f"training.{key}", 0)
    if t.warmup_steps >= t.max_steps:
        raise ValueError("warmup_steps must be smaller than max_steps.")
    if not 0 < t.learning_rate or not 0 < t.max_grad_norm or not 0 <= t.weight_decay:
        raise ValueError("Use positive learning_rate/max_grad_norm and nonnegative weight_decay.")
    for name, value in (("model.gradient_checkpointing", m.gradient_checkpointing),
                        ("training.full_determinism", t.full_determinism)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a YAML boolean.")
    for name, value in (("prepared_dir", d.prepared_dir), ("output_dir", t.output_dir)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path.")
    prepared, output = path(d.prepared_dir), path(t.output_dir)
    if prepared == output or prepared in output.parents or output in prepared.parents:
        raise ValueError("Keep prepared data and training output in separate directories.")
    return config


def load_config(filename):
    raw = yaml.safe_load(Path(filename).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"name", "model", "data", "training"}:
        raise ValueError("The config must contain name, model, data and training.")
    sections = {}
    for key, cls in (("model", ModelConfig), ("data", DataConfig), ("training", TrainConfig)):
        values = raw[key]
        if not isinstance(values, dict):
            raise ValueError(f"{key} must be a mapping.")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {key} settings: {', '.join(sorted(unknown))}")
        try:
            sections[key] = cls(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid {key} settings: {exc}") from exc
    return validate(Config(name=raw["name"], **sections))


def data_signature(config):
    values = asdict(config.data)
    values.pop("prepared_dir")
    values["tokenizer"] = {"source": model_source(config.model), "revision": config.model.revision}
    if config.data.source == "jsonl":
        values["input_hashes"] = {
            key: file_digest(path(getattr(config.data, key)))
            for key in ("train_file", "validation_file")
        }
    local = path(config.model.name_or_path)
    if local.is_dir():
        values["tokenizer_hashes"] = {
            p.name: file_digest(p) for p in sorted(local.iterdir())
            if p.is_file() and p.suffix in {".json", ".txt", ".model"}
        }
    return digest(values)


def run_signature(config):
    values = config.to_dict()
    values["training"].pop("output_dir")
    values["data"].pop("prepared_dir")
    values.pop("name")
    return digest(values)
