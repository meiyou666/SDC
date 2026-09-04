#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reproduction" / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer

from ft2_formal.artifacts import (
    atomic_write_artifact,
    json_sha256,
    load_artifact,
    sha256_file,
    utc_now,
)
from ft2_formal.campaign import (
    CONFIG_PATH,
    MAIN_MODES,
    _input_metadata,
    _output_payload,
    pair_id,
)
from ft2_formal.decoding import fixed_greedy_generate, prepare_prompt
from ft2_formal.engine import FT2HookEngine
from ft2_formal.manifest import load_manifest
from ft2_formal.reduced_campaign import FORMAL_OUTPUT_ROOT, ReducedFormalCampaign
from ft2_formal.schema import BoundsSource, FaultSpec
from ft2_formal.tasks import (
    DATASETS,
    MODELS,
    load_task_dataset,
    prompt_and_references,
    score_output,
)


MODEL_KEY = "qwen2_math_7b"
DATASET_KEY = "gsm8k"
PAIR_ID = pair_id(MODEL_KEY, DATASET_KEY)
EXPECTED_PROMPTS = 10
EXPECTED_SPECS = 90
EXPECTED_CONTROLS = 30
EXPECTED_RUNS = 270
GENERATION_STEPS = 180
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "reproduction" / "results" / "qwen_gsm_eos_sensitivity_v1"
)
EVALUABLE_OUTCOMES = {"SDC", "MASKED_IDENTICAL", "MASKED_SEMANTIC"}
METRIC_CONTRACT = {
    "schema": "qwen-gsm-eos-sensitivity-v1",
    "primary": {
        "id": "full_fixed_180",
        "scope": "all_180_generated_tokens",
        "stop_on_eos": False,
        "evaluator": "last_numeric_answer_match",
        "interpretation": "frozen_primary_protocol",
    },
    "sensitivity": {
        "id": "pre_first_eos",
        "scope": "generated_tokens_strictly_before_first_frozen_eos",
        "evaluator": "last_numeric_answer_match",
        "interpretation": "post_hoc_sensitivity_only",
    },
    "reachability": {
        "source": "mode_matched_fault_free_control",
        "step_semantics": (
            "fault target_step s affects logits that select generated_token_ids[s]"
        ),
        "rule": "reachable iff clean_first_eos_index is null or target_step <= index",
    },
    "clean_failure_policy": (
        "mode-matched clean incorrect makes associated fault-mode cells "
        "NON_EVALUABLE_CLEAN_FAILURE; never silently drop or count as masked"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent Qwen2-Math/GSM8K EOS sensitivity extension"
    )
    parser.add_argument(
        "--formal-root", type=Path, default=FORMAL_OUTPUT_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--audit-only", action="store_true")
    stage.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _add_eos_ids(target: set[int], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid EOS token id")
    if isinstance(value, int):
        target.add(int(value))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _add_eos_ids(target, item)
        return
    raise TypeError(f"Unsupported EOS token id value: {value!r}")


def discover_frozen_eos_ids() -> tuple[tuple[int, ...], dict[str, Any]]:
    snapshot = MODELS[MODEL_KEY].snapshot
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True
    )
    eos_ids: set[int] = set()
    _add_eos_ids(eos_ids, tokenizer.eos_token_id)
    sources: dict[str, Any] = {
        "tokenizer_runtime_eos_token_id": tokenizer.eos_token_id,
        "snapshot_files": {},
    }
    for name in (
        "generation_config.json",
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = snapshot / name
        if not path.exists():
            continue
        record: dict[str, Any] = {
            "path": f"model_snapshot/{name}",
            "sha256": sha256_file(path),
        }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        value = payload.get("eos_token_id")
        record["eos_token_id"] = value
        if value is not None:
            _add_eos_ids(eos_ids, value)
        sources["snapshot_files"][name] = record
    if not eos_ids:
        raise RuntimeError("No EOS token ids were discovered")
    return tuple(sorted(eos_ids)), sources


def first_eos_index(
    token_ids: Sequence[int], eos_ids: Iterable[int]
) -> int | None:
    frozen = frozenset(int(item) for item in eos_ids)
    for index, token_id in enumerate(token_ids):
        if int(token_id) in frozen:
            return index
    return None


def tokens_before_eos(
    token_ids: Sequence[int], eos_index: int | None
) -> list[int]:
    values = [int(item) for item in token_ids]
    return values if eos_index is None else values[:eos_index]


def fault_reachable(
    target_step: int, clean_first_eos_index: int | None
) -> bool:
    return (
        clean_first_eos_index is None
        or int(target_step) <= clean_first_eos_index
    )


def clean_status(
    *, mode_id: str, baseline_correct: bool, mode_correct: bool
) -> str:
    if mode_id == "no_protection":
        return "CLEAN_OK" if mode_correct else "BASELINE_CLEAN_FAILURE"
    if baseline_correct:
        return (
            "CLEAN_OK"
            if mode_correct
            else "PROTECTION_CLEAN_REGRESSION"
        )
    return (
        "BASELINE_FAILURE_PROTECTED_CORRECT"
        if mode_correct
        else "BASELINE_CLEAN_FAILURE"
    )


def view_outcome(
    *,
    terminal_status: str,
    clean_status_value: str,
    reachable: bool | None,
    faulty_correct: bool,
    exact_tokens: bool,
) -> str:
    if terminal_status == "due":
        return "DUE"
    if terminal_status != "completed":
        return "INVALID_RUN"
    if clean_status_value != "CLEAN_OK":
        return "NON_EVALUABLE_CLEAN_FAILURE"
    if reachable is False:
        return "NON_EVALUABLE_UNREACHABLE_UNDER_EOS"
    if not faulty_correct:
        return "SDC"
    return "MASKED_IDENTICAL" if exact_tokens else "MASKED_SEMANTIC"


def control_id(
    extension_fingerprint: str, source_position: int, mode_id: str
) -> str:
    return json_sha256(
        {
            "schema": "qwen-gsm-eos-control-v1",
            "extension_fingerprint": extension_fingerprint,
            "source_position": int(source_position),
            "mode_id": mode_id,
        }
    )


def run_id(
    extension_fingerprint: str, spec_id: str, mode_id: str
) -> str:
    return json_sha256(
        {
            "schema": "qwen-gsm-eos-run-v1",
            "extension_fingerprint": extension_fingerprint,
            "spec_id": spec_id,
            "mode_id": mode_id,
        }
    )


def _input_records(
    formal_root: Path, selection: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    fixed = {
        "parent_campaign": formal_root / "campaign.json",
        "parent_main_status": formal_root / "main_status.json",
        "parent_selection": formal_root / "selection.json",
        "parent_manifest": (
            formal_root / "manifests" / f"{PAIR_ID}.json"
        ),
        "parent_offline_bounds": (
            formal_root / "offline_bounds" / f"{PAIR_ID}.json"
        ),
        "formal_config": CONFIG_PATH,
        "extension_script": Path(__file__).resolve(),
    }
    records = {}
    for role, path in fixed.items():
        resolved = path.resolve()
        try:
            stable_path = resolved.relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"Fingerprint input is outside repository: {resolved}"
            ) from exc
        records[role] = {
            "path": stable_path,
            "sha256": sha256_file(resolved),
        }
    entry = selection["pairs"][PAIR_ID]
    for index, item in enumerate(entry["screening_artifacts"]):
        path = Path(item["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        role = f"parent_screening_{index:02d}"
        records[role] = {
            "path": path.resolve().relative_to(
                REPO_ROOT.resolve()
            ).as_posix(),
            "sha256": sha256_file(path),
        }
    return records


def build_context(
    formal_root: Path, output_root: Path
) -> dict[str, Any]:
    formal_root = formal_root.resolve()
    output_root = output_root.resolve()
    runner = ReducedFormalCampaign(formal_root)
    campaign = load_artifact(formal_root / "campaign.json")
    status = load_artifact(formal_root / "main_status.json")
    selection = load_artifact(formal_root / "selection.json")
    manifest_path = formal_root / "manifests" / f"{PAIR_ID}.json"
    manifest_meta, specs = load_manifest(manifest_path)
    manifest_document = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest_digest = manifest_document["manifest_sha256"]
    bounds_artifact = load_artifact(
        formal_root / "offline_bounds" / f"{PAIR_ID}.json"
    )
    fingerprint = runner.campaign_fingerprint
    if campaign["campaign_fingerprint"] != fingerprint:
        raise RuntimeError("Parent campaign fingerprint mismatch")
    if status.get("status") != "ABORTED_SAFETY_GATE":
        raise RuntimeError("Parent campaign is not frozen at the safety gate")
    if status.get("campaign_fingerprint") != fingerprint:
        raise RuntimeError("Parent main-status fingerprint mismatch")
    if selection.get("campaign_fingerprint") != fingerprint:
        raise RuntimeError("Parent selection fingerprint mismatch")
    if bounds_artifact.get("campaign_fingerprint") != fingerprint:
        raise RuntimeError("Parent bounds fingerprint mismatch")
    if manifest_meta.get("campaign_fingerprint") != fingerprint:
        raise RuntimeError("Parent manifest fingerprint mismatch")
    entry = selection["pairs"][PAIR_ID]
    if len(entry["source_positions"]) != EXPECTED_PROMPTS:
        raise RuntimeError("Expected ten frozen GSM8K prompts")
    if len(specs) != EXPECTED_SPECS:
        raise RuntimeError("Expected ninety frozen GSM8K FaultSpecs")
    if len({spec.spec_id for spec in specs}) != EXPECTED_SPECS:
        raise RuntimeError("FaultSpec ids are not unique")
    if any(
        spec.model_key != MODEL_KEY
        or spec.dataset_key != DATASET_KEY
        or spec.target_step < 1
        for spec in specs
    ):
        raise RuntimeError("Frozen FaultSpecs violate the extension contract")

    eos_ids, eos_sources = discover_frozen_eos_ids()
    inputs = _input_records(formal_root, selection)
    prompt_records = []
    for source_position, dataset_index, item in zip(
        entry["source_positions"],
        entry["dataset_indices"],
        entry["screening_artifacts"],
    ):
        path = Path(item["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        screen = load_artifact(path)
        if screen["artifact_sha256"] != item["artifact_sha256"]:
            raise RuntimeError("Frozen screening dependency hash mismatch")
        prompt_records.append(
            {
                "source_position": int(source_position),
                "dataset_index": int(dataset_index),
                "screening_artifact_sha256": screen["artifact_sha256"],
                "prompt_sha256": screen["input"]["prompt_sha256"],
            }
        )

    immutable = {
        "schema_version": 1,
        "experiment_id": "qwen_gsm_eos_sensitivity_v1",
        "post_hoc_sensitivity": True,
        "parent_campaign_fingerprint": fingerprint,
        "parent_main_status_sha256": status["artifact_sha256"],
        "inputs": inputs,
        "parent_manifest_sha256": manifest_digest,
        "parent_bounds_artifact_sha256": bounds_artifact[
            "artifact_sha256"
        ],
        "parent_model_identity": campaign["immutable"]["models"][MODEL_KEY],
        "parent_dataset_identity": campaign["immutable"]["datasets"][
            DATASET_KEY
        ],
        "parent_source_files": campaign["immutable"]["source_files"],
        "prompt_records": prompt_records,
        "frozen_eos_ids": list(eos_ids),
        "eos_sources": eos_sources,
        "generation": {
            "num_new_tokens": GENERATION_STEPS,
            "greedy": True,
            "stop_on_eos": False,
            "batch_size": 1,
            "dtype": "float16",
            "max_input_tokens": 1024,
        },
        "metric_contract": METRIC_CONTRACT,
        "modes": [
            {
                "mode_id": item.mode_id,
                "bounds_source": item.bounds_source.value,
                "correction": item.correction.value,
                "scaling_factor": item.scaling_factor,
            }
            for item in MAIN_MODES
        ],
        "expected": {
            "prompts": EXPECTED_PROMPTS,
            "fault_specs": EXPECTED_SPECS,
            "clean_controls": EXPECTED_CONTROLS,
            "fault_runs": EXPECTED_RUNS,
        },
        "seed": 196,
    }
    extension_fingerprint = json_sha256(immutable)
    planned_controls = [
        control_id(extension_fingerprint, source_position, mode.mode_id)
        for source_position in entry["source_positions"]
        for mode in MAIN_MODES
    ]
    planned_runs = [
        run_id(extension_fingerprint, spec.spec_id, mode.mode_id)
        for spec in specs
        for mode in MAIN_MODES
    ]
    return {
        "runner": runner,
        "formal_root": formal_root,
        "output_root": output_root,
        "campaign": campaign,
        "status": status,
        "selection": selection,
        "entry": entry,
        "manifest_meta": manifest_meta,
        "manifest_sha256": manifest_digest,
        "specs": tuple(specs),
        "bounds_artifact": bounds_artifact,
        "eos_ids": eos_ids,
        "immutable": immutable,
        "extension_fingerprint": extension_fingerprint,
        "planned_controls": planned_controls,
        "planned_runs": planned_runs,
    }


def ensure_extension_lock(context: Mapping[str, Any]) -> dict[str, Any]:
    root = context["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "extension.lock.json"
    fingerprint = context["extension_fingerprint"]
    if path.exists():
        lock = load_artifact(path)
        if (
            lock.get("extension_fingerprint") != fingerprint
            or lock.get("immutable") != context["immutable"]
            or lock.get("planned_control_ids")
            != context["planned_controls"]
            or lock.get("planned_run_ids") != context["planned_runs"]
        ):
            raise RuntimeError(
                "Extension identity changed; refusing to mix results"
            )
        return lock
    payload = {
        "schema_version": 1,
        "extension_fingerprint": fingerprint,
        "immutable": context["immutable"],
        "planned_control_ids": context["planned_controls"],
        "planned_run_ids": context["planned_runs"],
        "created_at": utc_now(),
    }
    lock = atomic_write_artifact(path, payload)
    protocol = {
        "schema_version": 1,
        "extension_fingerprint": fingerprint,
        "post_hoc_sensitivity": True,
        "primary_protocol": METRIC_CONTRACT["primary"],
        "sensitivity_protocol": METRIC_CONTRACT["sensitivity"],
        "reachability": METRIC_CONTRACT["reachability"],
        "clean_failure_policy": METRIC_CONTRACT[
            "clean_failure_policy"
        ],
        "frozen_eos_ids": list(context["eos_ids"]),
        "expected": context["immutable"]["expected"],
        "created_at": utc_now(),
    }
    atomic_write_artifact(root / "extension_protocol.json", protocol)
    lineage = {
        "schema_version": 1,
        "extension_fingerprint": fingerprint,
        "parent_campaign_fingerprint": context["campaign"][
            "campaign_fingerprint"
        ],
        "parent_main_status_sha256": context["status"][
            "artifact_sha256"
        ],
        "parent_manifest_sha256": context["manifest_sha256"],
        "parent_bounds_artifact_sha256": context["bounds_artifact"][
            "artifact_sha256"
        ],
        "input_files": context["immutable"]["inputs"],
        "created_at": utc_now(),
    }
    atomic_write_artifact(root / "lineage.json", lineage)
    return lock


def _verify_input_hashes(context: Mapping[str, Any]) -> None:
    for role, record in context["immutable"]["inputs"].items():
        observed = sha256_file(REPO_ROOT / record["path"])
        if observed != record["sha256"]:
            raise RuntimeError(f"Frozen input changed during run: {role}")


def _control_path(
    root: Path, source_position: int, mode_id: str
) -> Path:
    return (
        root
        / "raw"
        / "controls"
        / f"{source_position:02d}__{mode_id}.json"
    )


def _derived_control_path(
    root: Path, source_position: int, mode_id: str
) -> Path:
    return (
        root
        / "derived"
        / "controls"
        / f"{source_position:02d}__{mode_id}.json"
    )


def _run_path(root: Path, spec_id: str, mode_id: str) -> Path:
    return root / "raw" / "runs" / f"{spec_id}__{mode_id}.json"


def _derived_run_path(
    root: Path, spec_id: str, mode_id: str
) -> Path:
    return root / "derived" / "runs" / f"{spec_id}__{mode_id}.json"


def _assert_extension(
    record: Mapping[str, Any], fingerprint: str
) -> None:
    if record.get("extension_fingerprint") != fingerprint:
        raise RuntimeError("Artifact belongs to a different extension")


def _view_payload(
    *,
    dataset_key: str,
    tokenizer: Any,
    token_ids: Sequence[int],
    eos_ids: Sequence[int],
    references: Sequence[str],
) -> dict[str, Any]:
    eos_index = first_eos_index(token_ids, eos_ids)
    prefix_ids = tokens_before_eos(token_ids, eos_index)
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    return {
        "full_fixed_180": {
            "scope": "all_180_generated_tokens",
            "score": score_output(
                dataset_key,
                tokenizer.decode(
                    [int(item) for item in token_ids],
                    skip_special_tokens=True,
                ),
                references,
            ),
        },
        "pre_first_eos": {
            "scope": "generated_tokens_strictly_before_first_frozen_eos",
            "first_eos_index": eos_index,
            "token_ids": prefix_ids,
            "text": prefix_text,
            "score": score_output(dataset_key, prefix_text, references),
        },
    }


def run_controls(
    context: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    engine: FT2HookEngine,
    bounds: Mapping[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    root = context["output_root"]
    fingerprint = context["extension_fingerprint"]
    entry = context["entry"]
    dataset = load_task_dataset(DATASET_KEY)
    split = DATASETS[DATASET_KEY].evaluation_split
    result: dict[tuple[int, str], dict[str, Any]] = {}
    completed = 0
    for source_position, dataset_index in zip(
        entry["source_positions"], entry["dataset_indices"]
    ):
        canonical = context["runner"]._load_screen_record(
            MODEL_KEY, DATASET_KEY, source_position
        )
        example = dataset[split][dataset_index]
        prompt, references = prompt_and_references(DATASET_KEY, example)
        prepared = prepare_prompt(
            tokenizer, prompt, device="cuda", max_input_tokens=1024
        )
        if prepared.prompt_sha256 != canonical["input"]["prompt_sha256"]:
            raise RuntimeError("Control prompt differs from frozen screening")
        for protection in MAIN_MODES:
            mode_id = protection.mode_id
            expected_id = control_id(
                fingerprint, source_position, mode_id
            )
            path = _control_path(root, source_position, mode_id)
            if path.exists():
                record = load_artifact(path)
                _assert_extension(record, fingerprint)
                if (
                    record.get("control_id") != expected_id
                    or record.get("status") != "completed"
                    or len(record["output"]["token_ids"])
                    != GENERATION_STEPS
                    or record["engine"]["injection_count"] != 0
                ):
                    raise RuntimeError(
                        f"Invalid existing control artifact: {path}"
                    )
            else:
                supplied = (
                    bounds
                    if protection.bounds_source is BoundsSource.OFFLINE
                    else None
                )
                engine.start_inference(
                    fault=None,
                    protection=protection,
                    offline_bounds=supplied,
                )
                started = utc_now()
                try:
                    output = fixed_greedy_generate(
                        model,
                        tokenizer,
                        prepared,
                        engine,
                        num_new_tokens=GENERATION_STEPS,
                    )
                except Exception as exc:
                    partial = engine.abort()
                    atomic_write_artifact(
                        path,
                        {
                            "schema_version": 1,
                            "extension_fingerprint": fingerprint,
                            "control_id": expected_id,
                            "status": "invalid",
                            "source_position": source_position,
                            "dataset_index": dataset_index,
                            "mode_id": mode_id,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                            "engine": partial.to_dict(),
                            "started_at": started,
                            "ended_at": utc_now(),
                        },
                    )
                    raise
                if len(output.token_ids) != GENERATION_STEPS:
                    raise AssertionError("Control output length changed")
                engine_record = output.engine_record
                if engine_record.injection_count != 0:
                    raise AssertionError("Clean control injected a fault")
                if (
                    protection.bounds_source is BoundsSource.NONE
                    and engine_record.correction_elements != 0
                ):
                    raise AssertionError(
                        "No-protection control performed correction"
                    )
                if (
                    protection.bounds_source is BoundsSource.FIRST_TOKEN
                    and len(engine_record.online_bounds)
                    != len(adapter.critical_sites)
                ):
                    raise AssertionError(
                        "First-token clean-control bounds are incomplete"
                    )
                views = _view_payload(
                    dataset_key=DATASET_KEY,
                    tokenizer=tokenizer,
                    token_ids=output.token_ids,
                    eos_ids=context["eos_ids"],
                    references=references,
                )
                record = atomic_write_artifact(
                    path,
                    {
                        "schema_version": 1,
                        "extension_fingerprint": fingerprint,
                        "control_id": expected_id,
                        "status": "completed",
                        "pair_id": PAIR_ID,
                        "model_key": MODEL_KEY,
                        "model_revision": MODELS[MODEL_KEY].revision,
                        "dataset_key": DATASET_KEY,
                        "dataset_revision": DATASETS[
                            DATASET_KEY
                        ].revision,
                        "source_position": int(source_position),
                        "dataset_index": int(dataset_index),
                        "mode_id": mode_id,
                        "protection": {
                            "bounds_source": protection.bounds_source.value,
                            "correction": protection.correction.value,
                            "scaling_factor": protection.scaling_factor,
                        },
                        "parent_screening_artifact_sha256": canonical[
                            "artifact_sha256"
                        ],
                        "parent_bounds_artifact_sha256": (
                            context["bounds_artifact"]["artifact_sha256"]
                            if protection.bounds_source
                            is BoundsSource.OFFLINE
                            else None
                        ),
                        "input": _input_metadata(
                            tokenizer, prompt, prepared
                        ),
                        "references": list(references),
                        "frozen_eos_ids": list(context["eos_ids"]),
                        "frozen_first_eos_index": views[
                            "pre_first_eos"
                        ]["first_eos_index"],
                        "output": _output_payload(output),
                        "views": views,
                        "engine": engine_record.to_dict(),
                        "started_at": started,
                        "ended_at": utc_now(),
                    },
                )
            result[(source_position, mode_id)] = record
            completed += 1
            print(
                f"[control] {completed}/{EXPECTED_CONTROLS} "
                f"source={source_position} mode={mode_id} "
                f"full={record['views']['full_fixed_180']['score']['task_correct']} "
                f"pre_eos={record['views']['pre_first_eos']['score']['task_correct']}",
                flush=True,
            )
    if completed != EXPECTED_CONTROLS:
        raise AssertionError("Wrong number of clean controls")
    return result


def derive_controls(
    context: Mapping[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    root = context["output_root"]
    fingerprint = context["extension_fingerprint"]
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for source_position in context["entry"]["source_positions"]:
        raw = {
            mode.mode_id: load_artifact(
                _control_path(root, source_position, mode.mode_id)
            )
            for mode in MAIN_MODES
        }
        for view_id in ("full_fixed_180", "pre_first_eos"):
            baseline = raw["no_protection"]["views"][view_id]["score"][
                "task_correct"
            ]
            for mode in MAIN_MODES:
                mode_id = mode.mode_id
                raw[mode_id].setdefault("_derived_statuses", {})[
                    view_id
                ] = clean_status(
                    mode_id=mode_id,
                    baseline_correct=bool(baseline),
                    mode_correct=bool(
                        raw[mode_id]["views"][view_id]["score"][
                            "task_correct"
                        ]
                    ),
                )
        for mode in MAIN_MODES:
            mode_id = mode.mode_id
            control = raw[mode_id]
            path = _derived_control_path(
                root, source_position, mode_id
            )
            payload = {
                "schema_version": 1,
                "extension_fingerprint": fingerprint,
                "control_id": control["control_id"],
                "raw_artifact_sha256": control["artifact_sha256"],
                "source_position": int(source_position),
                "dataset_index": control["dataset_index"],
                "mode_id": mode_id,
                "views": {
                    view_id: {
                        "score": control["views"][view_id]["score"],
                        "clean_status": control["_derived_statuses"][
                            view_id
                        ],
                    }
                    for view_id in (
                        "full_fixed_180",
                        "pre_first_eos",
                    )
                },
                "frozen_first_eos_index": control[
                    "frozen_first_eos_index"
                ],
                "correction_elements": control["engine"][
                    "correction_elements"
                ],
                "created_at": utc_now(),
            }
            if path.exists():
                derived = load_artifact(path)
                _assert_extension(derived, fingerprint)
                if (
                    derived["raw_artifact_sha256"]
                    != control["artifact_sha256"]
                    or derived["views"] != payload["views"]
                ):
                    raise RuntimeError(
                        f"Existing derived control is stale: {path}"
                    )
            else:
                derived = atomic_write_artifact(path, payload)
            result[(source_position, mode_id)] = derived
            print(
                f"[clean-status] source={source_position} mode={mode_id} "
                f"full={derived['views']['full_fixed_180']['clean_status']} "
                f"pre_eos={derived['views']['pre_first_eos']['clean_status']}",
                flush=True,
            )
    return result


def run_faults(
    context: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    engine: FT2HookEngine,
    bounds: Mapping[str, Any],
    controls: Mapping[tuple[int, str], Mapping[str, Any]],
) -> None:
    root = context["output_root"]
    fingerprint = context["extension_fingerprint"]
    entry = context["entry"]
    dataset = load_task_dataset(DATASET_KEY)
    split = DATASETS[DATASET_KEY].evaluation_split
    completed = 0
    for spec in context["specs"]:
        source_position = entry["source_positions"][spec.sample_position]
        canonical = context["runner"]._load_screen_record(
            MODEL_KEY, DATASET_KEY, source_position
        )
        example = dataset[split][spec.dataset_index]
        prompt, references = prompt_and_references(DATASET_KEY, example)
        prepared = prepare_prompt(
            tokenizer, prompt, device="cuda", max_input_tokens=1024
        )
        if prepared.prompt_sha256 != canonical["input"]["prompt_sha256"]:
            raise RuntimeError("Fault prompt differs from frozen screening")
        site = adapter.sites_by_key[spec.site_key]
        for protection in MAIN_MODES:
            mode_id = protection.mode_id
            expected_id = run_id(
                fingerprint, spec.spec_id, mode_id
            )
            path = _run_path(root, spec.spec_id, mode_id)
            if path.exists():
                record = load_artifact(path)
                _assert_extension(record, fingerprint)
                if (
                    record.get("run_id") != expected_id
                    or record.get("spec_id") != spec.spec_id
                    or record.get("mode_id") != mode_id
                    or record.get("terminal_status")
                    not in {"completed", "due"}
                ):
                    raise RuntimeError(
                        f"Invalid existing fault artifact: {path}"
                    )
                completed += 1
                continue
            control = controls[(source_position, mode_id)]
            started = utc_now()
            base = {
                "schema_version": 1,
                "extension_fingerprint": fingerprint,
                "parent_campaign_fingerprint": context["campaign"]["campaign_fingerprint"],
                "run_id": expected_id,
                "spec_id": spec.spec_id,
                "pair_id": PAIR_ID,
                "model_key": MODEL_KEY,
                "model_revision": MODELS[MODEL_KEY].revision,
                "dataset_key": DATASET_KEY,
                "dataset_revision": DATASETS[DATASET_KEY].revision,
                "dataset_split": split,
                "source_manifest_sha256": context["manifest_sha256"],
                "selection_rank": spec.sample_position,
                "source_position": int(source_position),
                "dataset_index": spec.dataset_index,
                "mode_id": mode_id,
                "protection": {
                    "bounds_source": protection.bounds_source.value,
                    "correction": protection.correction.value,
                    "scaling_factor": protection.scaling_factor,
                },
                "fault": spec.to_dict(),
                "target_site": {
                    "key": site.key,
                    "module_path": site.module_path,
                    "critical": site.critical,
                    "sampling_weight": site.sampling_weight,
                },
                "parent_screening_artifact_sha256": canonical["artifact_sha256"],
                "mode_clean_control_id": control["control_id"],
                "mode_clean_control_sha256": control["artifact_sha256"],
                "parent_bounds_artifact_sha256": (
                    context["bounds_artifact"]["artifact_sha256"]
                    if protection.bounds_source is BoundsSource.OFFLINE
                    else None
                ),
                "references": list(references),
                "started_at": started,
                "attempt": 1,
            }
            supplied = (
                bounds
                if protection.bounds_source is BoundsSource.OFFLINE
                else None
            )
            engine.start_inference(
                fault=spec,
                protection=protection,
                offline_bounds=supplied,
            )
            try:
                output = fixed_greedy_generate(
                    model,
                    tokenizer,
                    prepared,
                    engine,
                    num_new_tokens=GENERATION_STEPS,
                )
            except Exception as exc:
                partial = engine.abort()
                trace = partial.injection_trace
                valid_fault = (
                    partial.injection_count == 1
                    and trace is not None
                    and trace.bit_flip_verified
                    and trace.hamming_distance == len(spec.bit_positions)
                )
                scientific_due = (
                    valid_fault
                    and protection.bounds_source is BoundsSource.FIRST_TOKEN
                    and spec.target_step == 0
                    and isinstance(exc, ValueError)
                    and str(exc).startswith("Bounds values must be finite")
                )
                atomic_write_artifact(
                    path,
                    {
                        **base,
                        "terminal_status": "due" if scientific_due else "invalid",
                        "due": scientific_due,
                        "due_reason": (
                            "DUE_INVALID_FIRST_TOKEN_BOUNDS"
                            if scientific_due
                            else None
                        ),
                        "engine": partial.to_dict(),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        "ended_at": utc_now(),
                    },
                )
                if not scientific_due:
                    raise RuntimeError(
                        f"Framework-invalid extension run {expected_id}"
                    ) from exc
                completed += 1
                print(
                    f"[run] {completed}/{EXPECTED_RUNS} DUE "
                    f"{spec.fault_type.value} {mode_id}",
                    flush=True,
                )
                continue
            engine_record = output.engine_record
            trace = engine_record.injection_trace
            if (
                engine_record.injection_count != 1
                or trace is None
                or not trace.bit_flip_verified
                or trace.hamming_distance != len(spec.bit_positions)
            ):
                raise AssertionError("Fault injection trace is invalid")
            if len(output.token_ids) != GENERATION_STEPS:
                raise AssertionError("Fault output length changed")
            if (
                protection.bounds_source is BoundsSource.NONE
                and engine_record.correction_elements != 0
            ):
                raise AssertionError("No-protection run performed correction")
            views = _view_payload(
                dataset_key=DATASET_KEY,
                tokenizer=tokenizer,
                token_ids=output.token_ids,
                eos_ids=context["eos_ids"],
                references=references,
            )
            record = atomic_write_artifact(
                path,
                {
                    **base,
                    "terminal_status": "completed",
                    "due": False,
                    "frozen_eos_ids": list(context["eos_ids"]),
                    "frozen_first_eos_index": views["pre_first_eos"]["first_eos_index"],
                    "output": _output_payload(output),
                    "views": views,
                    "engine": engine_record.to_dict(),
                    "online_bounds_sha256": (
                        json_sha256(
                            {
                                key: value.to_dict()
                                for key, value in sorted(
                                    engine_record.online_bounds.items()
                                )
                            }
                        )
                        if engine_record.online_bounds
                        else None
                    ),
                    "ended_at": utc_now(),
                },
            )
            completed += 1
            print(
                f"[run] {completed}/{EXPECTED_RUNS} "
                f"{spec.fault_type.value} {mode_id} "
                f"full_correct={record['views']['full_fixed_180']['score']['task_correct']} "
                f"pre_eos_correct={record['views']['pre_first_eos']['score']['task_correct']}",
                flush=True,
            )
    if completed != EXPECTED_RUNS:
        raise AssertionError("Wrong number of fault runs")


def derive_runs(context: Mapping[str, Any]) -> None:
    root = context["output_root"]
    fingerprint = context["extension_fingerprint"]
    entry = context["entry"]
    completed = 0
    for spec in context["specs"]:
        source_position = entry["source_positions"][spec.sample_position]
        for mode in MAIN_MODES:
            mode_id = mode.mode_id
            raw = load_artifact(_run_path(root, spec.spec_id, mode_id))
            _assert_extension(raw, fingerprint)
            control = load_artifact(
                _control_path(root, source_position, mode_id)
            )
            control_derived = load_artifact(
                _derived_control_path(root, source_position, mode_id)
            )
            views: dict[str, Any] = {}
            for view_id in ("full_fixed_180", "pre_first_eos"):
                status_value = control_derived["views"][view_id]["clean_status"]
                if raw["terminal_status"] == "completed":
                    if view_id == "full_fixed_180":
                        faulty_tokens = raw["output"]["token_ids"]
                        clean_tokens = control["output"]["token_ids"]
                        faulty_text = raw["output"]["text"]
                        clean_text = control["output"]["text"]
                        reachable = None
                    else:
                        faulty_tokens = raw["views"][view_id]["token_ids"]
                        clean_tokens = control["views"][view_id]["token_ids"]
                        faulty_text = raw["views"][view_id]["text"]
                        clean_text = control["views"][view_id]["text"]
                        reachable = fault_reachable(
                            spec.target_step,
                            control["frozen_first_eos_index"],
                        )
                    score = raw["views"][view_id]["score"]
                    exact_tokens = list(faulty_tokens) == list(clean_tokens)
                    exact_text = faulty_text == clean_text
                    faulty_correct = bool(score["task_correct"])
                else:
                    score = None
                    exact_tokens = False
                    exact_text = False
                    faulty_correct = False
                    reachable = (
                        None
                        if view_id == "full_fixed_180"
                        else fault_reachable(
                            spec.target_step,
                            control["frozen_first_eos_index"],
                        )
                    )
                outcome = view_outcome(
                    terminal_status=raw["terminal_status"],
                    clean_status_value=status_value,
                    reachable=reachable,
                    faulty_correct=faulty_correct,
                    exact_tokens=exact_tokens,
                )
                views[view_id] = {
                    "score": score,
                    "clean_status": status_value,
                    "clean_task_correct": bool(
                        control["views"][view_id]["score"]["task_correct"]
                    ),
                    "faulty_task_correct": faulty_correct,
                    "exact_token_match_mode_clean": exact_tokens,
                    "exact_text_match_mode_clean": exact_text,
                    "fault_reachable": reachable,
                    "outcome": outcome,
                    "evaluable": outcome in EVALUABLE_OUTCOMES,
                    "sdc": outcome == "SDC",
                    "due": outcome == "DUE",
                }
            payload = {
                "schema_version": 1,
                "extension_fingerprint": fingerprint,
                "run_id": raw["run_id"],
                "raw_artifact_sha256": raw["artifact_sha256"],
                "spec_id": spec.spec_id,
                "source_position": int(source_position),
                "dataset_index": spec.dataset_index,
                "mode_id": mode_id,
                "target_step": spec.target_step,
                "views": views,
                "created_at": utc_now(),
            }
            path = _derived_run_path(root, spec.spec_id, mode_id)
            if path.exists():
                derived = load_artifact(path)
                _assert_extension(derived, fingerprint)
                if (
                    derived["raw_artifact_sha256"] != raw["artifact_sha256"]
                    or derived["views"] != payload["views"]
                ):
                    raise RuntimeError(f"Existing derived run is stale: {path}")
            else:
                atomic_write_artifact(path, payload)
            completed += 1
    if completed != EXPECTED_RUNS:
        raise AssertionError("Wrong number of derived runs")


def _exact_mcnemar(improve: int, worsen: int) -> float:
    discordant = improve + worsen
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(improve, worsen) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def _wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        / denominator
    )
    return [center - half, center + half]


def audit_and_summarize(context: Mapping[str, Any]) -> dict[str, Any]:
    _verify_input_hashes(context)
    root = context["output_root"]
    fingerprint = context["extension_fingerprint"]
    expected_control_names = {
        f"{source:02d}__{mode.mode_id}.json"
        for source in context["entry"]["source_positions"]
        for mode in MAIN_MODES
    }
    expected_run_names = {
        f"{spec.spec_id}__{mode.mode_id}.json"
        for spec in context["specs"]
        for mode in MAIN_MODES
    }
    for directory, expected in (
        (root / "raw" / "controls", expected_control_names),
        (root / "derived" / "controls", expected_control_names),
        (root / "raw" / "runs", expected_run_names),
        (root / "derived" / "runs", expected_run_names),
    ):
        observed = {path.name for path in directory.glob("*.json")}
        if observed != expected:
            raise RuntimeError(
                f"Artifact set mismatch in {directory}: "
                f"missing={len(expected-observed)} extra={len(observed-expected)}"
            )

    critical_count = int(context["bounds_artifact"]["critical_site_count"])
    control_rows: list[dict[str, Any]] = []
    for source_position in context["entry"]["source_positions"]:
        for mode in MAIN_MODES:
            mode_id = mode.mode_id
            raw = load_artifact(
                _control_path(root, source_position, mode_id)
            )
            derived = load_artifact(
                _derived_control_path(root, source_position, mode_id)
            )
            _assert_extension(raw, fingerprint)
            _assert_extension(derived, fingerprint)
            if raw["status"] != "completed":
                raise RuntimeError("A clean control is incomplete")
            if len(raw["output"]["token_ids"]) != GENERATION_STEPS:
                raise RuntimeError("A clean control is not fixed-length")
            if raw["engine"]["injection_count"] != 0:
                raise RuntimeError("A clean control injected a fault")
            recomputed_eos = first_eos_index(
                raw["output"]["token_ids"], context["eos_ids"]
            )
            if recomputed_eos != raw["frozen_first_eos_index"]:
                raise RuntimeError("A control EOS index is incorrect")
            if (
                mode.bounds_source is BoundsSource.NONE
                and raw["engine"]["correction_elements"] != 0
            ):
                raise RuntimeError(
                    "No-protection control corrected activations"
                )
            if (
                mode.bounds_source is BoundsSource.FIRST_TOKEN
                and raw["engine"]["online_bounds_key_count"] != critical_count
            ):
                raise RuntimeError("First-token bounds are incomplete")
            if derived["raw_artifact_sha256"] != raw["artifact_sha256"]:
                raise RuntimeError("Derived control lineage mismatch")
            control_rows.append(derived)

    spec_by_id = {spec.spec_id: spec for spec in context["specs"]}
    run_rows: list[dict[str, Any]] = []
    for spec in context["specs"]:
        for mode in MAIN_MODES:
            mode_id = mode.mode_id
            raw = load_artifact(_run_path(root, spec.spec_id, mode_id))
            derived = load_artifact(
                _derived_run_path(root, spec.spec_id, mode_id)
            )
            _assert_extension(raw, fingerprint)
            _assert_extension(derived, fingerprint)
            if raw["terminal_status"] not in {"completed", "due"}:
                raise RuntimeError("Invalid run entered accepted artifacts")
            engine = raw["engine"]
            trace = engine["injection_trace"]
            if engine["injection_count"] != 1 or trace is None:
                raise RuntimeError("Fault injection count is not exactly one")
            if (
                trace["spec_id"] != spec.spec_id
                or trace["observed_step"] != spec.target_step
                or trace["site_key"] != spec.site_key
                or trace["flat_index"] != spec.flat_index
                or trace["bit_positions"] != list(spec.bit_positions)
                or not trace["bit_flip_verified"]
                or trace["hamming_distance"] != len(spec.bit_positions)
            ):
                raise RuntimeError("Injection trace differs from FaultSpec")
            before = int(trace["before_bits_hex"], 16)
            after = int(trace["after_bits_hex"], 16)
            mask = sum(1 << bit for bit in spec.bit_positions)
            if after != (before ^ mask):
                raise RuntimeError("FP16 XOR audit failed")
            if raw["terminal_status"] == "completed":
                if len(raw["output"]["token_ids"]) != GENERATION_STEPS:
                    raise RuntimeError("Fault output is not fixed-length")
                recomputed_eos = first_eos_index(
                    raw["output"]["token_ids"], context["eos_ids"]
                )
                if recomputed_eos != raw["frozen_first_eos_index"]:
                    raise RuntimeError("A fault EOS index is incorrect")
            if (
                mode.bounds_source is BoundsSource.NONE
                and engine["correction_elements"] != 0
            ):
                raise RuntimeError(
                    "No-protection fault run corrected activations"
                )
            if (
                mode.bounds_source is BoundsSource.FIRST_TOKEN
                and engine["online_bounds_key_count"] != critical_count
            ):
                raise RuntimeError("First-token run bounds are incomplete")
            if derived["raw_artifact_sha256"] != raw["artifact_sha256"]:
                raise RuntimeError("Derived run lineage mismatch")
            for view_id, view in derived["views"].items():
                if view["outcome"] not in (
                    EVALUABLE_OUTCOMES
                    | {
                        "DUE",
                        "INVALID_RUN",
                        "NON_EVALUABLE_CLEAN_FAILURE",
                        "NON_EVALUABLE_UNREACHABLE_UNDER_EOS",
                    }
                ):
                    raise RuntimeError(f"Unknown outcome in {view_id}")
                if view_id == "pre_first_eos":
                    control = load_artifact(
                        _control_path(
                            root, derived["source_position"], mode_id
                        )
                    )
                    expected_reach = fault_reachable(
                        spec.target_step,
                        control["frozen_first_eos_index"],
                    )
                    if view["fault_reachable"] != expected_reach:
                        raise RuntimeError("Reachability audit failed")
            run_rows.append(derived)

    clean_summary: dict[str, Any] = {}
    for view_id in ("full_fixed_180", "pre_first_eos"):
        counts = Counter(
            row["views"][view_id]["clean_status"] for row in control_rows
        )
        clean_summary[view_id] = {
            "planned_prompt_mode_controls": EXPECTED_CONTROLS,
            "status_counts": dict(sorted(counts.items())),
            "protection_clean_regressions": sum(
                row["views"][view_id]["clean_status"]
                == "PROTECTION_CLEAN_REGRESSION"
                for row in control_rows
            ),
        }

    result_tables: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for view_id in ("full_fixed_180", "pre_first_eos"):
        mode_tables = {}
        for mode in MAIN_MODES:
            rows = [
                row for row in run_rows if row["mode_id"] == mode.mode_id
            ]
            outcomes = Counter(
                row["views"][view_id]["outcome"] for row in rows
            )
            evaluable = sum(
                row["views"][view_id]["outcome"] in EVALUABLE_OUTCOMES
                for row in rows
            )
            sdc = outcomes["SDC"]
            mode_tables[mode.mode_id] = {
                "planned_fault_mode_cells": EXPECTED_SPECS,
                "evaluable": evaluable,
                "outcome_counts": dict(sorted(outcomes.items())),
                "sdc": sdc,
                "sdc_rate_evaluable": sdc / evaluable if evaluable else None,
                "sdc_wilson95_evaluable": _wilson(sdc, evaluable),
                "safety_endpoint_sdc_or_due": sdc + outcomes["DUE"],
            }
        result_tables[view_id] = mode_tables

        by_key = {
            (row["spec_id"], row["mode_id"]): row for row in run_rows
        }
        paired[view_id] = {}
        for treatment in (
            "paper_clamp_first_token_bounds",
            "paper_clamp_offline_bounds",
        ):
            improve = worsen = paired_n = 0
            for spec_id in spec_by_id:
                baseline = by_key[(spec_id, "no_protection")]["views"][
                    view_id
                ]["outcome"]
                treated = by_key[(spec_id, treatment)]["views"][
                    view_id
                ]["outcome"]
                if (
                    baseline not in EVALUABLE_OUTCOMES
                    or treated not in EVALUABLE_OUTCOMES
                ):
                    continue
                paired_n += 1
                improve += baseline == "SDC" and treated != "SDC"
                worsen += baseline != "SDC" and treated == "SDC"
            paired[view_id][treatment] = {
                "planned_pairs": EXPECTED_SPECS,
                "evaluable_intersection": paired_n,
                "improve": improve,
                "worsen": worsen,
                "exact_mcnemar_p": _exact_mcnemar(improve, worsen),
            }

    audit_payload = {
        "schema_version": 1,
        "extension_fingerprint": fingerprint,
        "status": "passed",
        "post_hoc_sensitivity": True,
        "counts": {
            "clean_controls": len(control_rows),
            "fault_runs": len(run_rows),
            "derived_controls": len(control_rows),
            "derived_runs": len(run_rows),
        },
        "invariants": {
            "parent_inputs_unchanged": True,
            "exactly_10_prompts": True,
            "exactly_90_unique_fault_specs": True,
            "exactly_3_modes_per_spec": True,
            "fixed_180_tokens_for_completed_runs": True,
            "one_verified_xor_per_fault_run": True,
            "eos_derived_from_token_ids": True,
            "mode_matched_clean_controls": True,
            "non_evaluable_not_in_sdc_denominator": True,
        },
        "audited_at": utc_now(),
    }
    audit_path = root / "audit.json"
    if audit_path.exists():
        audit = load_artifact(audit_path)
        _assert_extension(audit, fingerprint)
        if audit.get("status") != "passed":
            raise RuntimeError("Existing extension audit did not pass")
    else:
        audit = atomic_write_artifact(audit_path, audit_payload)

    summary_payload = {
        "schema_version": 1,
        "extension_fingerprint": fingerprint,
        "status": "completed",
        "primary_metric": "full_fixed_180",
        "post_hoc_sensitivity_metric": "pre_first_eos",
        "warning": (
            "The pre-first-EOS analysis is post hoc and does not overturn "
            "the frozen full-180 safety-gate failure."
        ),
        "clean_controls": clean_summary,
        "fault_results": result_tables,
        "paired_comparisons": paired,
        "created_at": utc_now(),
    }
    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = load_artifact(summary_path)
        _assert_extension(summary, fingerprint)
    else:
        summary = atomic_write_artifact(summary_path, summary_payload)

    completion_payload = {
        "schema_version": 1,
        "extension_fingerprint": fingerprint,
        "status": "completed",
        "post_hoc_sensitivity": True,
        "audit_artifact_sha256": audit["artifact_sha256"],
        "summary_artifact_sha256": summary["artifact_sha256"],
        "completed_at": utc_now(),
    }
    completion_path = root / "completion.json"
    if completion_path.exists():
        completion = load_artifact(completion_path)
        _assert_extension(completion, fingerprint)
    else:
        completion = atomic_write_artifact(completion_path, completion_payload)
    return {"audit": audit, "summary": summary, "completion": completion}


def self_test() -> None:
    eos = (2, 9)
    assert first_eos_index([], eos) is None
    assert first_eos_index([2, 4], eos) == 0
    assert first_eos_index([1, 9, 2], eos) == 1
    assert first_eos_index([1, 3], eos) is None
    assert tokens_before_eos([1, 2, 3], None) == [1, 2, 3]
    assert tokens_before_eos([1, 2, 3], 0) == []
    assert tokens_before_eos([1, 2, 3], 1) == [1]
    assert fault_reachable(4, None)
    assert fault_reachable(3, 4)
    assert fault_reachable(4, 4)
    assert not fault_reachable(5, 4)
    assert (
        clean_status(
            mode_id="paper_clamp_first_token_bounds",
            baseline_correct=True,
            mode_correct=False,
        )
        == "PROTECTION_CLEAN_REGRESSION"
    )
    assert (
        view_outcome(
            terminal_status="completed",
            clean_status_value="PROTECTION_CLEAN_REGRESSION",
            reachable=True,
            faulty_correct=True,
            exact_tokens=True,
        )
        == "NON_EVALUABLE_CLEAN_FAILURE"
    )
    assert (
        view_outcome(
            terminal_status="completed",
            clean_status_value="CLEAN_OK",
            reachable=False,
            faulty_correct=True,
            exact_tokens=True,
        )
        == "NON_EVALUABLE_UNREACHABLE_UNDER_EOS"
    )
    assert (
        view_outcome(
            terminal_status="completed",
            clean_status_value="CLEAN_OK",
            reachable=True,
            faulty_correct=False,
            exact_tokens=False,
        )
        == "SDC"
    )
    print("[self-test] passed", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runner_lock_path = output_root / ".runner.lock"
    with runner_lock_path.open("a+", encoding="utf-8") as runner_lock:
        try:
            fcntl.flock(
                runner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise SystemExit(
                f"Another extension process holds {runner_lock_path}"
            ) from exc
        context = build_context(args.formal_root.resolve(), output_root)
        lock = ensure_extension_lock(context)
        print(
            f"[extension] fingerprint={context['extension_fingerprint']} "
            f"lock={lock['artifact_sha256']}",
            flush=True,
        )
        if args.prepare_only:
            print("[extension] preparation complete", flush=True)
            return 0
        if args.audit_only:
            derive_controls(context)
            derive_runs(context)
            result = audit_and_summarize(context)
            print(
                json.dumps(
                    {
                        "status": result["completion"]["status"],
                        "extension_fingerprint": context[
                            "extension_fingerprint"
                        ],
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 0

        runner = context["runner"]
        with runner.loaded_model(MODEL_KEY) as (model, tokenizer, adapter):
            runner.ensure_model_preflight(
                MODEL_KEY, model, tokenizer, adapter
            )
            bounds, observed_bounds_artifact = runner.ensure_offline_bounds(
                MODEL_KEY,
                DATASET_KEY,
                model,
                tokenizer,
                adapter,
            )
            if (
                observed_bounds_artifact["artifact_sha256"]
                != context["bounds_artifact"]["artifact_sha256"]
            ):
                raise RuntimeError("Offline bounds changed after locking")
            with FT2HookEngine(adapter) as engine:
                controls = run_controls(
                    context,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    engine=engine,
                    bounds=bounds,
                )
                derive_controls(context)
                run_faults(
                    context,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    engine=engine,
                    bounds=bounds,
                    controls=controls,
                )
        torch.cuda.empty_cache()
        derive_runs(context)
        result = audit_and_summarize(context)
        print(
            json.dumps(
                {
                    "status": result["completion"]["status"],
                    "extension_fingerprint": context[
                        "extension_fingerprint"
                    ],
                    "summary": str(
                        context["output_root"] / "summary.json"
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
