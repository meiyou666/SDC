"""Training configuration: schema, validation and reproducibility digests.

``data_signature`` covers everything that determines the tokenised dataset;
prepared data is only reused when it matches. ``run_signature`` covers
everything that determines the optimisation trajectory; resuming requires it to
be identical.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml


#: Repository root, used to resolve the relative paths written in the config.
ROOT = Path(__file__).resolve().parents[2]

DEVICES = {"cpu", "cuda", "npu", "accelerator"}
PRECISIONS = {"fp32", "bf16", "fp16"}
ATTENTIONS = {"eager", "sdpa"}
SOURCES = {"huggingface", "jsonl"}

#: Settings that must be positive integers, with their minimum value.
_POSITIVE_TRAINING = (
    "max_steps", "batch_size", "eval_batch_size", "gradient_accumulation_steps",
    "logging_steps", "eval_steps", "save_steps", "save_total_limit", "cpu_threads",
)
_NONNEGATIVE_TRAINING = ("seed", "warmup_steps", "dataloader_num_workers")


@dataclass(frozen=True)
class ModelConfig:
    """Model identity. ``revision`` pins a Hugging Face commit SHA."""

    name_or_path: str
    revision: str | None = None
    attention: str = "eager"
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class DataConfig:
    """Corpus identity and the packing applied to it.

    ``source: huggingface`` reads ``name``/``subset`` at the pinned ``revision``.
    ``source: jsonl`` reads local files whose lines are ``{"text": "..."}``.
    Texts are concatenated and cut into ``block_size`` token sequences.
    """

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
    """Optimisation settings. ``max_steps`` counts optimizer updates.

    ``device: accelerator`` resolves to ``cuda`` or ``npu`` from the container's
    ``SDC_RUNTIME``, so the same file runs unchanged on both training images.
    """

    output_dir: str
    device: str = "accelerator"
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
    """A complete, validated training task."""

    name: str
    model: ModelConfig
    data: DataConfig
    training: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path; relative paths are read against the repository root."""
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (ROOT / result).resolve()


def model_source(config: ModelConfig) -> str:
    """Return a local model directory when it exists, otherwise the Hub id."""
    local = resolve_path(config.name_or_path)
    return str(local) if local.is_dir() else config.name_or_path


def digest(value: Any) -> str:
    """SHA256 of a JSON-serialisable value, stable across key order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(filename: str | Path) -> str:
    """SHA256 of a file, read in 1 MiB blocks."""
    result = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _positive(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是不小于 {minimum} 的整数。")


def validate(config: Config) -> Config:
    """Reject configurations that would fail late or silently change results."""
    m, d, t = config.model, config.data, config.training
    if not isinstance(config.name, str) or not config.name.strip():
        raise ValueError("name 必须是非空字符串。")
    if not isinstance(m.name_or_path, str) or not m.name_or_path.strip():
        raise ValueError("必须设置 model.name_or_path。")
    if not resolve_path(m.name_or_path).is_dir() and not re.fullmatch(r"[0-9a-f]{40}", m.revision or ""):
        raise ValueError("model.revision 必须固定为 Hugging Face 的 commit SHA。")
    if m.attention not in ATTENTIONS:
        raise ValueError(f"model.attention 必须是 {' 或 '.join(sorted(ATTENTIONS))}。")
    if d.source not in SOURCES:
        raise ValueError(f"data.source 必须是 {' 或 '.join(sorted(SOURCES))}。")
    if d.source == "huggingface":
        if not d.name or not re.fullmatch(r"[0-9a-f]{40}", d.revision or ""):
            raise ValueError("必须设置 data.name 和固定的 data.revision。")
        if d.train_split == d.validation_split:
            raise ValueError("训练集和验证集必须使用不同的 split。")
    else:
        if not d.train_file or not d.validation_file:
            raise ValueError("必须设置 data.train_file 和 data.validation_file。")
        if resolve_path(d.train_file) == resolve_path(d.validation_file):
            raise ValueError("训练文件和验证文件必须不同。")
    _positive(d.block_size, "data.block_size", 2)
    for key in ("max_train_documents", "max_validation_documents"):
        if getattr(d, key) is not None:
            _positive(getattr(d, key), f"data.{key}")
    if t.device not in DEVICES:
        raise ValueError("training.device 必须是 accelerator、cuda、npu 或 cpu（仅用于单元测试）。")
    if t.precision not in PRECISIONS:
        raise ValueError(f"training.precision 必须是 {' 或 '.join(sorted(PRECISIONS))}。")
    if t.device == "cpu" and t.precision != "fp32":
        raise ValueError("CPU 检查请使用 fp32。")
    for key in _POSITIVE_TRAINING:
        _positive(getattr(t, key), f"training.{key}")
    for key in _NONNEGATIVE_TRAINING:
        _positive(getattr(t, key), f"training.{key}", 0)
    if t.warmup_steps >= t.max_steps:
        raise ValueError("warmup_steps 必须小于 max_steps。")
    if not 0 < t.learning_rate or not 0 < t.max_grad_norm or not 0 <= t.weight_decay:
        raise ValueError("learning_rate、max_grad_norm 必须为正，weight_decay 必须非负。")
    for name, value in (("model.gradient_checkpointing", m.gradient_checkpointing),
                        ("training.full_determinism", t.full_determinism)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} 必须是 YAML 布尔值。")
    for name, value in (("prepared_dir", d.prepared_dir), ("output_dir", t.output_dir)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 必须是非空路径。")
    prepared, output = resolve_path(d.prepared_dir), resolve_path(t.output_dir)
    if prepared == output or prepared in output.parents or output in prepared.parents:
        raise ValueError("prepared_dir 与 output_dir 必须是互不包含的目录。")
    return config


def load_config(filename: str | Path) -> Config:
    """Read and validate a training YAML file."""
    raw = yaml.safe_load(Path(filename).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"name", "model", "data", "training"}:
        raise ValueError("配置必须包含 name、model、data 和 training 四个部分。")
    sections: dict[str, Any] = {}
    for key, cls in (("model", ModelConfig), ("data", DataConfig), ("training", TrainConfig)):
        values = raw[key]
        if not isinstance(values, dict):
            raise ValueError(f"{key} 必须是映射。")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"{key} 中有未知设置：{', '.join(sorted(unknown))}")
        try:
            sections[key] = cls(**values)
        except TypeError as exc:
            raise ValueError(f"{key} 设置无效：{exc}") from exc
    return validate(Config(name=raw["name"], **sections))


def data_signature(config: Config) -> str:
    """Digest of everything that determines the tokenised dataset.

    ``prepared_dir`` is excluded so the same data can be prepared anywhere, and
    the tokenizer is included because it decides the token ids.
    """
    values = asdict(config.data)
    values.pop("prepared_dir")
    values["tokenizer"] = {"source": model_source(config.model), "revision": config.model.revision}
    if config.data.source == "jsonl":
        values["input_hashes"] = {
            key: file_digest(resolve_path(getattr(config.data, key)))
            for key in ("train_file", "validation_file")
        }
    local = resolve_path(config.model.name_or_path)
    if local.is_dir():
        values["tokenizer_hashes"] = {
            p.name: file_digest(p) for p in sorted(local.iterdir())
            if p.is_file() and p.suffix in {".json", ".txt", ".model"}
        }
    return digest(values)


def run_signature(config: Config) -> str:
    """Digest of everything that determines the optimisation trajectory.

    Output locations and the task name are excluded; every other setting is
    included, so resuming into a changed configuration is refused.
    """
    values = config.to_dict()
    values["training"].pop("output_dir")
    values["data"].pop("prepared_dir")
    values.pop("name")
    return digest(values)
