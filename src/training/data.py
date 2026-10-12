"""Prepare fixed-length next-token examples from real text."""

from dataclasses import asdict
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import tempfile

from datasets import Dataset, DatasetDict, Features, List, Value, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer

from .config import data_signature, digest, file_digest, model_source, path


def load_tokenizer(config, offline=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_source(config.model), revision=config.model.revision,
        trust_remote_code=False, local_files_only=offline,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def documents(config, split):
    data = config.data
    if data.source == "huggingface":
        rows = load_dataset(
            data.name, data.subset, revision=data.revision,
            split=data.train_split if split == "train" else data.validation_split,
            streaming=True,
        )
    else:
        filename = path(data.train_file if split == "train" else data.validation_file)

        def json_rows():
            with filename.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Invalid JSON: {filename}:{line_number}") from exc

        rows = json_rows()
    limit = data.max_train_documents if split == "train" else data.max_validation_documents
    for row in islice(rows, limit):
        text = row.get(data.text_field)
        if not isinstance(text, str):
            raise ValueError(f"Each {split} record needs a string field {data.text_field!r}.")
        yield text


def pack_texts(texts, tokenizer, block_size, stats):
    """Carry tokens across batches/documents; discard only the final short block."""
    stats.update(documents=0, nonempty_documents=0, tokens=0, blocks=0, dropped_tokens=0)
    pending = []
    iterator = iter(texts)
    while batch := list(islice(iterator, 128)):
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


def _artifact_hashes(directory):
    return {
        str(p.relative_to(directory)): file_digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }


def _artifact_state(directory):
    state = {}
    for filename in sorted(directory.rglob("*")):
        if filename.is_file() and filename.name != "manifest.json":
            stat = filename.stat()
            state[str(filename.relative_to(directory))] = [
                stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev,
            ]
    return state


def _verify_artifacts(directory, manifest, force=False):
    """Reuse a successful hash check while the manifest and file metadata match."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cache = cache_root / "sdc-verification" / (digest(str(directory.resolve())) + ".json")
    state = _artifact_state(directory)
    expected = {"manifest": prepared_digest(manifest), "files": state}
    if not force:
        try:
            if json.loads(cache.read_text()) == expected:
                return
        except (OSError, ValueError):
            pass
    if manifest["artifacts"] != _artifact_hashes(directory):
        raise ValueError("Prepared dataset files have changed. Prepare the data in a new directory.")
    if _artifact_state(directory) != state:
        raise ValueError("Prepared dataset files changed during verification. Stop writes and retry.")
    temporary = None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=cache.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(expected, stream)
        temporary.replace(cache)
    except OSError:
        # Read-only caches do not prevent using data that passed SHA256 verification.
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def prepare_data(config, verify_data=False):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Prepare data once before launching distributed training.")
    output = path(config.data.prepared_dir)
    signature = data_signature(config)
    if output.exists():
        if output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        else:
            _, manifest = read_prepared(config, verify_data=verify_data)
            return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        tokenizer = load_tokenizer(config)
        split_data, statistics = {}, {}
        for split in ("train", "validation"):
            stats = {}

            def generate(split=split, stats=stats):
                yield from pack_texts(documents(config, split), tokenizer, config.data.block_size, stats)

            split_data[split] = Dataset.from_generator(
                generate,
                features=Features({"input_ids": List(Value("int32"), length=config.data.block_size)}),
                cache_dir=str(staging / "build-cache" / split),
            )
            if not len(split_data[split]):
                raise ValueError(f"No complete {split} sequences; supply more text or reduce block_size.")
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


def read_prepared(config, verify_data=False):
    directory = path(config.data.prepared_dir)
    filename = directory / "manifest.json"
    if not directory.exists():
        raise ValueError("Prepared data is missing. Run the prepare command first.")
    incomplete = f"Prepared directory is incomplete: {directory}. Use a new prepared_dir or remove the incomplete directory before preparing again."
    if not filename.is_file():
        raise ValueError(incomplete)
    try:
        manifest = json.loads(filename.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(incomplete) from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), dict) or not manifest["artifacts"] or "signature" not in manifest:
        raise ValueError(incomplete)
    if manifest.get("signature") != data_signature(config):
        raise ValueError("Prepared data differs from this config. Use a new prepared_dir and prepare it.")
    _verify_artifacts(directory, manifest, force=verify_data)
    dataset = load_from_disk(str(directory))
    return dataset, manifest


def download_model(config):
    source = model_source(config.model)
    if Path(source).is_dir():
        return source
    return snapshot_download(
        repo_id=source, revision=config.model.revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )


class CausalCollator:
    """No padding: retain the EOS targets even when pad_token equals eos_token."""

    def __call__(self, features):
        ids = torch.tensor([example["input_ids"] for example in features], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def prepared_digest(manifest):
    return digest({"signature": manifest["signature"], "artifacts": manifest["artifacts"]})
