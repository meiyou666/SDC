from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .core import FaultSpec


AUTHOR_PROJECTION_WEIGHTS: Tuple[Tuple[str, int], ...] = (
    ("v_proj", 1),
    ("k_proj", 1),
    ("q_proj", 7),
    ("o_proj", 7),
    ("up_proj", 37),
    ("gate_proj", 37),
    ("down_proj", 37),
)


@dataclass(frozen=True)
class ModelShape:
    num_layers: int
    projection_widths: Mapping[str, int]

    @classmethod
    def from_config(cls, config: Any) -> "ModelShape":
        hidden_size = int(config.hidden_size)
        attention_heads = int(config.num_attention_heads)
        key_value_heads = int(config.num_key_value_heads)
        head_dim = int(
            getattr(config, "head_dim", hidden_size // attention_heads)
        )
        intermediate_size = int(config.intermediate_size)
        num_layers = int(config.num_hidden_layers)

        if hidden_size <= 0 or attention_heads <= 0:
            raise ValueError("Invalid hidden size or attention head count")
        if hidden_size % attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention heads")
        if min(key_value_heads, head_dim, intermediate_size, num_layers) <= 0:
            raise ValueError("Model dimensions must be positive")

        widths = {
            "q_proj": attention_heads * head_dim,
            "k_proj": key_value_heads * head_dim,
            "v_proj": key_value_heads * head_dim,
            "o_proj": hidden_size,
            "gate_proj": intermediate_size,
            "up_proj": intermediate_size,
            "down_proj": hidden_size,
        }
        return cls(num_layers=num_layers, projection_widths=widths)


@dataclass(frozen=True)
class ManifestEntry:
    dataset_index: int
    fault: FaultSpec
    campaign_seed: int

    @property
    def spec_id(self) -> str:
        return self.fault.stable_hash

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = asdict(self.fault)
        result.update(
            {
                "dataset_index": self.dataset_index,
                "campaign_seed": self.campaign_seed,
                "spec_id": self.spec_id,
                "batch_index": 0,
                "sequence_index": 0,
                "feature_index": self.fault.flat_index,
                "fault_model": "single_random_fp16_bit_flip",
            }
        )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManifestEntry":
        fault = FaultSpec(
            sample_index=int(value["sample_index"]),
            trial_index=int(value["trial_index"]),
            token_step=int(value["token_step"]),
            layer_index=int(value["layer_index"]),
            projection=str(value["projection"]),
            flat_index=int(value["flat_index"]),
            bit_index=int(value["bit_index"]),
        )
        entry = cls(
            dataset_index=int(value["dataset_index"]),
            fault=fault,
            campaign_seed=int(value["campaign_seed"]),
        )
        supplied_id = value.get("spec_id")
        if supplied_id is not None and supplied_id != entry.spec_id:
            raise ValueError("Manifest spec_id does not match its fault fields")
        return entry


def build_fault_manifest(
    dataset_indices: Sequence[int],
    fault_steps: Sequence[int],
    trials_per_sample: int,
    shape: ModelShape,
    seed: int = 196,
) -> Tuple[ManifestEntry, ...]:
    if len(dataset_indices) != len(fault_steps):
        raise ValueError("dataset_indices and fault_steps must have same length")
    if not dataset_indices:
        raise ValueError("At least one sample is required")
    if trials_per_sample <= 0:
        raise ValueError("trials_per_sample must be positive")

    names = [name for name, _ in AUTHOR_PROJECTION_WEIGHTS]
    weights = [weight for _, weight in AUTHOR_PROJECTION_WEIGHTS]
    missing = set(names).difference(shape.projection_widths)
    if missing:
        raise ValueError(f"Missing projection widths: {sorted(missing)}")

    rng = random.Random(seed)
    entries: List[ManifestEntry] = []

    for sample_index, (dataset_index, token_step) in enumerate(
        zip(dataset_indices, fault_steps)
    ):
        if dataset_index < 0:
            raise ValueError("dataset indices must be non-negative")
        if token_step < 1:
            raise ValueError("fault steps must be >= 1")

        for trial_index in range(trials_per_sample):
            layer_index = rng.randrange(shape.num_layers)
            projection = rng.choices(names, weights=weights, k=1)[0]
            bit_index = rng.randrange(16)
            width = int(shape.projection_widths[projection])
            flat_index = rng.randrange(width)

            entries.append(
                ManifestEntry(
                    dataset_index=dataset_index,
                    campaign_seed=seed,
                    fault=FaultSpec(
                        sample_index=sample_index,
                        trial_index=trial_index,
                        token_step=token_step,
                        layer_index=layer_index,
                        projection=projection,
                        flat_index=flat_index,
                        bit_index=bit_index,
                    ),
                )
            )

    validate_manifest(
        entries,
        sample_count=len(dataset_indices),
        trials_per_sample=trials_per_sample,
        shape=shape,
    )
    return tuple(entries)


def validate_manifest(
    entries: Iterable[ManifestEntry],
    sample_count: int,
    trials_per_sample: int,
    shape: ModelShape,
    num_new_tokens: int | None = None,
) -> Tuple[ManifestEntry, ...]:
    materialized = tuple(entries)
    expected = sample_count * trials_per_sample
    if len(materialized) != expected:
        raise ValueError(
            f"Expected {expected} manifest entries, got {len(materialized)}"
        )

    ids = {entry.spec_id for entry in materialized}
    if len(ids) != len(materialized):
        raise ValueError("Manifest contains duplicate spec_id values")

    counts = {sample_index: 0 for sample_index in range(sample_count)}
    for entry in materialized:
        fault = entry.fault
        if fault.sample_index not in counts:
            raise ValueError("Manifest sample_index is out of range")
        counts[fault.sample_index] += 1
        if not 0 <= fault.layer_index < shape.num_layers:
            raise ValueError("Manifest layer_index is out of range")
        width = int(shape.projection_widths[fault.projection])
        if not 0 <= fault.flat_index < width:
            raise ValueError("Manifest feature index is out of range")
        if num_new_tokens is not None and fault.token_step >= num_new_tokens:
            raise ValueError(
                "Fault step must be less than fixed generated token count"
            )

    if set(counts.values()) != {trials_per_sample}:
        raise ValueError(f"Unexpected per-sample trial counts: {counts}")
    return materialized


def manifest_sha256(entries: Iterable[ManifestEntry]) -> str:
    payload = [entry.to_dict() for entry in entries]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def save_manifest(path: str | Path, entries: Iterable[ManifestEntry]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.to_dict() for entry in entries]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_manifest(path: str | Path) -> Tuple[ManifestEntry, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Manifest root must be a JSON array")
    return tuple(ManifestEntry.from_dict(item) for item in payload)
