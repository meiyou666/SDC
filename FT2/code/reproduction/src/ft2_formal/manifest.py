from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from .adapters import ModelAdapter
from .schema import FaultSpec, FaultType


def _derived_seed(
    campaign_seed: int,
    model_key: str,
    dataset_key: str,
    dataset_index: int,
    sample_position: int,
    fault_type: FaultType,
    trial_index: int,
) -> int:
    payload = "|".join(
        (
            "ft2-pcg64-v1",
            str(campaign_seed),
            model_key,
            dataset_key,
            str(dataset_index),
            str(sample_position),
            fault_type.value,
            str(trial_index),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def build_pair_manifest(
    *,
    adapter: ModelAdapter,
    dataset_key: str,
    dataset_indices: Sequence[int],
    target_steps: Sequence[int],
    trials_per_fault_type: int,
    generation_steps: int,
    campaign_seed: int = 196,
    max_input_tokens: int = 1024,
) -> Tuple[FaultSpec, ...]:
    if len(dataset_indices) != len(target_steps) or not dataset_indices:
        raise ValueError("dataset_indices and target_steps must be non-empty peers")
    if trials_per_fault_type <= 0 or generation_steps <= 0:
        raise ValueError("Trial and generation counts must be positive")
    weights = adapter.projection_weights
    names = tuple(weights)
    probabilities = np.asarray([weights[name] for name in names], dtype=np.float64)
    probabilities /= probabilities.sum()
    sites = adapter.sites_by_key
    result: list[FaultSpec] = []

    for sample_position, (dataset_index, target_step) in enumerate(
        zip(dataset_indices, target_steps)
    ):
        dataset_index = int(dataset_index)
        target_step = int(target_step)
        if dataset_index < 0:
            raise ValueError("dataset_index must be non-negative")
        if not 0 <= target_step < generation_steps:
            raise ValueError(
                f"Author target step {target_step} is outside {generation_steps} steps"
            )
        for fault_type in FaultType:
            for trial_index in range(trials_per_fault_type):
                seed = _derived_seed(
                    campaign_seed,
                    adapter.model_key,
                    dataset_key,
                    dataset_index,
                    sample_position,
                    fault_type,
                    trial_index,
                )
                rng = np.random.Generator(np.random.PCG64(seed))
                layer_index = int(rng.integers(0, adapter.num_layers))
                projection = str(rng.choice(names, p=probabilities))
                site = sites[f"layer.{layer_index}.{projection}"]
                expected_sequence_length = max_input_tokens if target_step == 0 else 1
                sequence_index = int(rng.integers(0, expected_sequence_length))
                feature_index = int(rng.integers(0, site.out_features))
                if fault_type is FaultType.FP16_1BIT:
                    bits = (int(rng.integers(0, 16)),)
                elif fault_type is FaultType.FP16_2BIT:
                    bits = tuple(
                        sorted(
                            int(value)
                            for value in rng.choice(16, size=2, replace=False)
                        )
                    )
                else:
                    bits = (int(rng.integers(10, 15)),)
                result.append(
                    FaultSpec(
                        model_key=adapter.model_key,
                        dataset_key=dataset_key,
                        dataset_index=dataset_index,
                        sample_position=sample_position,
                        trial_index=trial_index,
                        fault_type=fault_type,
                        target_step=target_step,
                        layer_index=layer_index,
                        projection=projection,
                        site_key=site.key,
                        sequence_index=sequence_index,
                        feature_index=feature_index,
                        expected_sequence_length=expected_sequence_length,
                        expected_out_features=site.out_features,
                        bit_positions=bits,
                        campaign_seed=campaign_seed,
                    )
                )
    expected = (
        len(dataset_indices) * len(FaultType) * trials_per_fault_type
    )
    if len(result) != expected:
        raise AssertionError("Manifest cardinality is incorrect")
    ids = {spec.spec_id for spec in result}
    if len(ids) != len(result):
        raise AssertionError("Manifest contains duplicate spec IDs")
    return tuple(result)


def manifest_sha256(specs: Sequence[FaultSpec]) -> str:
    payload = [spec.to_dict() for spec in specs]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def save_manifest(
    path: str | Path,
    specs: Sequence[FaultSpec],
    metadata: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "metadata": dict(metadata),
        "manifest_sha256": manifest_sha256(specs),
        "specs": [spec.to_dict() for spec in specs],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], Tuple[FaultSpec, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = tuple(FaultSpec.from_dict(value) for value in payload["specs"])
    if payload.get("manifest_sha256") != manifest_sha256(specs):
        raise ValueError("Manifest hash verification failed")
    return dict(payload["metadata"]), specs
