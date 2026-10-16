"""Turn real text into fixed-length next-token training examples.

``prepare_data`` writes a snapshot of the tokenised corpus to ``prepared_dir``:
``train/`` and ``validation/`` Arrow datasets with one ``input_ids`` column of
``block_size`` tokens, plus ``manifest.json`` holding the data signature, EOS
id, packing statistics and a SHA256 per file. ``read_prepared`` checks the
signature and re-hashes the files on every read.
"""

from collections.abc import Iterable, Iterator
from dataclasses import asdict
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from datasets import Dataset, DatasetDict, Features, List, Value, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer

from .config import Config, data_signature, digest, file_digest, model_source, resolve_path


#: Texts handed to the tokenizer per call. Only affects speed, not the output.
TOKENIZER_BATCH = 128

#: Model files downloaded by ``prepare``; excludes optimizer and ONNX exports.
MODEL_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]

INCOMPLETE = (
    "prepared_dir 不完整：{directory}。请改用新的 prepared_dir，"
    "或先删除这个残留目录再重新准备数据。"
)


def load_tokenizer(config: Config, offline: bool = False):
    """Load the pinned tokenizer and guarantee an EOS id for packing."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_source(config.model), revision=config.model.revision,
        trust_remote_code=False, local_files_only=offline,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("分词器必须定义 eos_token_id。")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def documents(config: Config, split: str) -> Iterator[str]:
    """Stream the raw texts of one split, at most ``max_*_documents`` of them.

    One row is one record of the upstream dataset, which for WikiText is a line
    rather than an article.
    """
    data = config.data
    if data.source == "huggingface":
        rows: Iterable[Any] = load_dataset(
            data.name, data.subset, revision=data.revision,
            split=data.train_split if split == "train" else data.validation_split,
            streaming=True,
        )
    else:
        filename = resolve_path(data.train_file if split == "train" else data.validation_file)

        def json_rows() -> Iterator[Any]:
            with filename.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"JSON 格式错误：{filename}:{line_number}") from exc

        rows = json_rows()
    limit = data.max_train_documents if split == "train" else data.max_validation_documents
    for row in islice(rows, limit):
        text = row.get(data.text_field)
        if not isinstance(text, str):
            raise ValueError(f"{split} 的每条记录都需要字符串字段 {data.text_field!r}。")
        yield text


def pack_texts(
    texts: Iterable[str], tokenizer, block_size: int, stats: dict[str, int],
) -> Iterator[dict[str, list[int]]]:
    """Concatenate texts into ``block_size`` blocks, one EOS after each text.

    Tokens carry across both documents and tokenizer batches; only the trailing
    partial block is discarded. ``stats`` is filled in place with the document,
    token and block counts recorded in the manifest.
    """
    stats.update(documents=0, nonempty_documents=0, tokens=0, blocks=0, dropped_tokens=0)
    pending: list[int] = []
    iterator = iter(texts)
    while batch := list(islice(iterator, TOKENIZER_BATCH)):
        stats["documents"] += len(batch)
        batch = [text for text in batch if text.strip()]
        stats["nonempty_documents"] += len(batch)
        if not batch:
            continue
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False, return_attention_mask=False)
        for ids in encoded["input_ids"]:
            ids = ids + [tokenizer.eos_token_id]
            stats["tokens"] += len(ids)
            pending.extend(ids)
            complete = len(pending) // block_size * block_size
            for start in range(0, complete, block_size):
                stats["blocks"] += 1
                yield {"input_ids": pending[start:start + block_size]}
            pending = pending[complete:]
    stats["dropped_tokens"] = len(pending)


def _artifact_hashes(directory: Path) -> dict[str, str]:
    """SHA256 of every prepared file except the manifest that records them."""
    return {
        str(p.relative_to(directory)): file_digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }


def prepare_data(config: Config) -> dict[str, Any]:
    """Tokenise and pack the corpus into ``prepared_dir``; reuse it if present.

    The snapshot is built in a temporary directory and renamed into place.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("请在启动训练前，单进程准备一次数据。")
    output = resolve_path(config.data.prepared_dir)
    signature = data_signature(config)
    if output.exists():
        if output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        else:
            _, manifest = read_prepared(config)
            return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        tokenizer = load_tokenizer(config)
        split_data, statistics = {}, {}
        for split in ("train", "validation"):
            stats: dict[str, int] = {}

            def generate(split=split, stats=stats):
                yield from pack_texts(documents(config, split), tokenizer, config.data.block_size, stats)

            split_data[split] = Dataset.from_generator(
                generate,
                features=Features({"input_ids": List(Value("int32"), length=config.data.block_size)}),
                cache_dir=str(staging / "build-cache" / split),
            )
            if not len(split_data[split]):
                raise ValueError(f"没有完整的 {split} 序列：请增加文本量或减小 block_size。")
            statistics[split] = stats
        packed = staging / "packed"
        DatasetDict(split_data).save_to_disk(str(packed))
        manifest = {
            "schema_version": 1, "task": "causal_language_modeling", "signature": signature,
            "model": asdict(config.model), "data": asdict(config.data),
            "eos_token_id": tokenizer.eos_token_id, "statistics": statistics,
            "artifacts": _artifact_hashes(packed),
        }
        (packed / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        packed.rename(output)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def read_prepared(config: Config) -> tuple[DatasetDict, dict[str, Any]]:
    """Load prepared data after checking its signature and file hashes."""
    directory = resolve_path(config.data.prepared_dir)
    filename = directory / "manifest.json"
    if not directory.exists():
        raise ValueError("缺少已准备的数据，请先执行 prepare 命令。")
    incomplete = INCOMPLETE.format(directory=directory)
    if not filename.is_file():
        raise ValueError(incomplete)
    try:
        manifest = json.loads(filename.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(incomplete) from exc
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("artifacts"), dict)
        or not manifest["artifacts"]
        or "signature" not in manifest
    ):
        raise ValueError(incomplete)
    if manifest["signature"] != data_signature(config):
        raise ValueError("已准备的数据与当前配置不一致，请改用新的 prepared_dir 重新准备。")
    if manifest["artifacts"] != _artifact_hashes(directory):
        raise ValueError("已准备的数据文件发生变化，请在新目录重新准备数据。")
    return load_from_disk(str(directory)), manifest


def download_model(config: Config) -> str:
    """Fetch the pinned model weights into the shared Hugging Face cache."""
    source = model_source(config.model)
    if Path(source).is_dir():
        return source
    return snapshot_download(
        repo_id=source, revision=config.model.revision, allow_patterns=MODEL_PATTERNS,
    )


class CausalCollator:
    """Build a next-token batch.

    Blocks are already ``block_size`` tokens, so no padding is applied and the
    EOS positions stay in the loss even when ``pad_token`` equals ``eos_token``.
    """

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        ids = torch.tensor([example["input_ids"] for example in features], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def prepared_digest(manifest: dict[str, Any]) -> str:
    """Single digest identifying the prepared data, recorded in ``run.json``."""
    return digest({"signature": manifest["signature"], "artifacts": manifest["artifacts"]})
