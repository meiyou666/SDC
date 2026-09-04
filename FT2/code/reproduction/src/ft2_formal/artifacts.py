from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


ARTIFACT_HASH_FIELD = "artifact_sha256"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def artifact_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop(ARTIFACT_HASH_FIELD, None)
    payload[ARTIFACT_HASH_FIELD] = json_sha256(payload)
    return payload


def verify_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    supplied = payload.pop(ARTIFACT_HASH_FIELD, None)
    expected = json_sha256(payload)
    if supplied != expected:
        raise ValueError(
            f"Artifact hash mismatch: supplied={supplied!r}, expected={expected!r}"
        )
    payload[ARTIFACT_HASH_FIELD] = expected
    return payload


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_artifact(
    path: str | Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = artifact_payload(value)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, text)
    return payload


def load_artifact(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return verify_artifact(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_ids_sha256(value: torch.Tensor) -> str:
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError("Expected token IDs with shape [1, sequence]")
    integers = [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return json_sha256(integers)


def stable_run_id(
    *,
    campaign_fingerprint: str,
    spec_id: str,
    mode_id: str,
) -> str:
    payload = "|".join(
        ("ft2-run-v1", campaign_fingerprint, spec_id, mode_id)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
