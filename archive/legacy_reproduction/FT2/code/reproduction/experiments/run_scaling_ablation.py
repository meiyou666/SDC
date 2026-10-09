#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reproduction" / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from ft2_formal.artifacts import (
    atomic_write_artifact,
    json_sha256,
    load_artifact,
    sha256_file,
    utc_now,
)
from ft2_formal.campaign import (
    _input_metadata,
    _output_payload,
)
from ft2_formal.decoding import fixed_greedy_generate, prepare_prompt
from ft2_formal.engine import FT2HookEngine
from ft2_formal.manifest import load_manifest
from ft2_formal.reduced_campaign import FORMAL_OUTPUT_ROOT, ReducedFormalCampaign
from ft2_formal.schema import BoundsSource, Correction, ProtectionSpec
from ft2_formal.tasks import (
    DATASETS,
    MODELS,
    load_task_dataset,
    prompt_and_references,
)


EOS_SCRIPT = (
    REPO_ROOT
    / "reproduction"
    / "experiments"
    / "run_qwen_gsm_eos_extension.py"
)
_eos_spec = importlib.util.spec_from_file_location("ft2_eos_extension", EOS_SCRIPT)
if _eos_spec is None or _eos_spec.loader is None:
    raise RuntimeError("Cannot import the frozen EOS extension")
EOS = importlib.util.module_from_spec(_eos_spec)
_eos_spec.loader.exec_module(EOS)

MODEL_KEY = "qwen2_math_7b"
DATASET_KEY = "gsm8k"
PAIR_ID = "qwen2_math_7b__gsm8k"
FACTORS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
FACTOR_LABELS = {
    1.0: "first_token_s1p00",
    1.25: "first_token_s1p25",
    1.5: "first_token_s1p50",
    1.75: "first_token_s1p75",
    2.0: "first_token_s2p00",
    2.25: "first_token_s2p25",
    2.5: "first_token_s2p50",
}
NO_PROTECTION = "no_protection"
LOGICAL_MODES = (NO_PROTECTION,) + tuple(FACTOR_LABELS[x] for x in FACTORS)
NEW_FACTORS = tuple(x for x in FACTORS if x != 2.0)
REUSED_MODES = (NO_PROTECTION, FACTOR_LABELS[2.0])
EXPECTED_PROMPTS = 10
EXPECTED_SPECS = 30
EXPECTED_LOGICAL_CONTROLS = 80
EXPECTED_LOGICAL_RUNS = 240
EXPECTED_REFERENCE_CONTROLS = 20
EXPECTED_REFERENCE_RUNS = 60
EXPECTED_NEW_CONTROLS = 60
EXPECTED_NEW_RUNS = 180
GENERATION_STEPS = 180
DEFAULT_EOS_ROOT = (
    REPO_ROOT / "reproduction" / "results" / "qwen_gsm_eos_sensitivity_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "reproduction" / "results" / "scaling_ablation_v1"
)
EVALUABLE_OUTCOMES = {"SDC", "MASKED_IDENTICAL", "MASKED_SEMANTIC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FT2 first-token scaling-factor ablation on frozen GSM EXP faults"
    )
    parser.add_argument("--formal-root", type=Path, default=FORMAL_OUTPUT_ROOT)
    parser.add_argument("--eos-root", type=Path, default=DEFAULT_EOS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--audit-only", action="store_true")
    stage.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def factor_for_label(label: str) -> float | None:
    if label == NO_PROTECTION:
        return None
    for factor, candidate in FACTOR_LABELS.items():
        if candidate == label:
            return factor
    raise KeyError(label)


def protection_for_factor(factor: float) -> ProtectionSpec:
    return ProtectionSpec(
        BoundsSource.FIRST_TOKEN,
        Correction.PAPER_CLAMP,
        float(factor),
    )


def logical_control_id(fp: str, source_position: int, label: str) -> str:
    return json_sha256(
        {
            "schema": "ft2-scaling-control-v1",
            "scaling_fingerprint": fp,
            "source_position": int(source_position),
            "logical_mode": label,
        }
    )


def logical_run_id(fp: str, spec_id: str, label: str) -> str:
    return json_sha256(
        {
            "schema": "ft2-scaling-run-v1",
            "scaling_fingerprint": fp,
            "spec_id": spec_id,
            "logical_mode": label,
        }
    )


def _stable_file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": sha256_file(resolved),
    }


def build_context(
    formal_root: Path, eos_root: Path, output_root: Path
) -> dict[str, Any]:
    formal_root = formal_root.resolve()
    eos_root = eos_root.resolve()
    output_root = output_root.resolve()
    runner = ReducedFormalCampaign(formal_root)
    campaign = load_artifact(formal_root / "campaign.json")
    selection = load_artifact(formal_root / "selection.json")
    manifest_path = formal_root / "manifests" / f"{PAIR_ID}.json"
    manifest_meta, all_specs = load_manifest(manifest_path)
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = tuple(
        item
        for item in all_specs
        if item.fault_type.value == "fp16_exponent_bit"
    )
    eos_lock = load_artifact(eos_root / "extension.lock.json")
    eos_audit = load_artifact(eos_root / "audit.json")
    eos_completion = load_artifact(eos_root / "completion.json")
    eos_summary = load_artifact(eos_root / "summary.json")
    eos_fp = eos_lock["extension_fingerprint"]
    if runner.campaign_fingerprint != campaign["campaign_fingerprint"]:
        raise RuntimeError("Parent formal campaign fingerprint mismatch")
    if selection["campaign_fingerprint"] != runner.campaign_fingerprint:
        raise RuntimeError("Parent selection fingerprint mismatch")
    if manifest_meta["campaign_fingerprint"] != runner.campaign_fingerprint:
        raise RuntimeError("Parent manifest fingerprint mismatch")
    if (
        eos_audit.get("status") != "passed"
        or eos_completion.get("status") != "completed"
        or eos_audit.get("extension_fingerprint") != eos_fp
        or eos_completion.get("extension_fingerprint") != eos_fp
        or eos_summary.get("extension_fingerprint") != eos_fp
    ):
        raise RuntimeError("EOS parent extension is not completed and audited")
    entry = selection["pairs"][PAIR_ID]
    if len(entry["source_positions"]) != EXPECTED_PROMPTS:
        raise RuntimeError("Expected ten frozen prompts")
    if len(specs) != EXPECTED_SPECS:
        raise RuntimeError("Expected thirty frozen exponent FaultSpecs")
    if len({item.spec_id for item in specs}) != EXPECTED_SPECS:
        raise RuntimeError("Exponent FaultSpec ids are not unique")
    if any(item.target_step < 1 for item in specs):
        raise RuntimeError("Scaling ablation unexpectedly contains step-0 faults")

    inputs = {
        "scaling_script": _stable_file_record(Path(__file__)),
        "eos_script": _stable_file_record(EOS_SCRIPT),
        "formal_campaign": _stable_file_record(formal_root / "campaign.json"),
        "formal_selection": _stable_file_record(formal_root / "selection.json"),
        "formal_manifest": _stable_file_record(manifest_path),
        "eos_lock": _stable_file_record(eos_root / "extension.lock.json"),
        "eos_audit": _stable_file_record(eos_root / "audit.json"),
        "eos_completion": _stable_file_record(eos_root / "completion.json"),
        "eos_summary": _stable_file_record(eos_root / "summary.json"),
    }
    immutable = {
        "schema_version": 1,
        "experiment_id": "ft2_qwen_gsm_exp_scaling_ablation_v1",
        "parent_formal_campaign_fingerprint": runner.campaign_fingerprint,
        "parent_eos_extension_fingerprint": eos_fp,
        "inputs": inputs,
        "formal_manifest_sha256": manifest_document["manifest_sha256"],
        "selected_fault_type": "fp16_exponent_bit",
        "selected_spec_ids": [item.spec_id for item in specs],
        "source_positions": list(entry["source_positions"]),
        "dataset_indices": list(entry["dataset_indices"]),
        "factors": [f"{item:.2f}" for item in FACTORS],
        "logical_modes": list(LOGICAL_MODES),
        "reused_modes": list(REUSED_MODES),
        "generation": {
            "num_new_tokens": GENERATION_STEPS,
            "greedy": True,
            "stop_on_eos": False,
            "dtype": "float16",
            "batch_size": 1,
        },
        "metric_contract": EOS.METRIC_CONTRACT,
        "frozen_eos_ids": list(
            eos_lock["immutable"]["frozen_eos_ids"]
        ),
        "expected": {
            "prompts": EXPECTED_PROMPTS,
            "fault_specs": EXPECTED_SPECS,
            "logical_controls": EXPECTED_LOGICAL_CONTROLS,
            "logical_fault_runs": EXPECTED_LOGICAL_RUNS,
            "reused_controls": EXPECTED_REFERENCE_CONTROLS,
            "reused_fault_runs": EXPECTED_REFERENCE_RUNS,
            "new_gpu_controls": EXPECTED_NEW_CONTROLS,
            "new_gpu_fault_runs": EXPECTED_NEW_RUNS,
            "new_gpu_inferences": EXPECTED_NEW_CONTROLS + EXPECTED_NEW_RUNS,
        },
        "seed": 196,
    }
    fingerprint = json_sha256(immutable)
    planned_controls = [
        logical_control_id(fingerprint, source, label)
        for source in entry["source_positions"]
        for label in LOGICAL_MODES
    ]
    planned_runs = [
        logical_run_id(fingerprint, item.spec_id, label)
        for item in specs
        for label in LOGICAL_MODES
    ]
    return {
        "runner": runner,
        "formal_root": formal_root,
        "eos_root": eos_root,
        "output_root": output_root,
        "campaign": campaign,
        "selection": selection,
        "entry": entry,
        "manifest_meta": manifest_meta,
        "manifest_sha256": manifest_document["manifest_sha256"],
        "specs": specs,
        "eos_lock": eos_lock,
        "eos_audit": eos_audit,
        "eos_fp": eos_fp,
        "eos_ids": tuple(eos_lock["immutable"]["frozen_eos_ids"]),
        "immutable": immutable,
        "fingerprint": fingerprint,
        "planned_controls": planned_controls,
        "planned_runs": planned_runs,
    }


def ensure_lock(context: Mapping[str, Any]) -> dict[str, Any]:
    root = context["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "scaling.lock.json"
    if path.exists():
        lock = load_artifact(path)
        if (
            lock.get("scaling_fingerprint") != context["fingerprint"]
            or lock.get("immutable") != context["immutable"]
            or lock.get("planned_control_ids") != context["planned_controls"]
            or lock.get("planned_run_ids") != context["planned_runs"]
        ):
            raise RuntimeError("Scaling identity changed; refusing mixed results")
        return lock
    lock = atomic_write_artifact(
        path,
        {
            "schema_version": 1,
            "scaling_fingerprint": context["fingerprint"],
            "immutable": context["immutable"],
            "planned_control_ids": context["planned_controls"],
            "planned_run_ids": context["planned_runs"],
            "created_at": utc_now(),
        },
    )
    atomic_write_artifact(
        root / "protocol.json",
        {
            "schema_version": 1,
            "scaling_fingerprint": context["fingerprint"],
            "status": "frozen",
            "scope": "qwen2_math_7b/gsm8k/fp16_exponent_bit",
            "factors": [f"{item:.2f}" for item in FACTORS],
            "primary_metric": "full_fixed_180",
            "post_hoc_sensitivity_metric": "pre_first_eos",
            "reuse": {
                "no_protection": "audited EOS extension artifacts",
                "first_token_s2p00": "audited EOS extension artifacts",
            },
            "expected": context["immutable"]["expected"],
            "created_at": utc_now(),
        },
    )
    return lock


def _reference_control_path(root: Path, source: int, label: str) -> Path:
    return root / "references" / "controls" / f"{source:02d}__{label}.json"


def _reference_run_path(root: Path, spec_id: str, label: str) -> Path:
    return root / "references" / "runs" / f"{spec_id}__{label}.json"


def _raw_control_path(root: Path, source: int, label: str) -> Path:
    return root / "raw" / "controls" / f"{source:02d}__{label}.json"


def _raw_run_path(root: Path, spec_id: str, label: str) -> Path:
    return root / "raw" / "runs" / f"{spec_id}__{label}.json"


def _derived_control_path(root: Path, source: int, label: str) -> Path:
    return root / "derived" / "controls" / f"{source:02d}__{label}.json"


def _derived_run_path(root: Path, spec_id: str, label: str) -> Path:
    return root / "derived" / "runs" / f"{spec_id}__{label}.json"


def _source_mode(label: str) -> str:
    if label == NO_PROTECTION:
        return "no_protection"
    if label == FACTOR_LABELS[2.0]:
        return "paper_clamp_first_token_bounds"
    raise ValueError(f"{label} is not a reused mode")


def _write_or_validate_reference(
    path: Path,
    payload: Mapping[str, Any],
    *,
    fingerprint: str,
) -> dict[str, Any]:
    if path.exists():
        record = load_artifact(path)
        if (
            record.get("scaling_fingerprint") != fingerprint
            or record.get("source_artifact_sha256")
            != payload["source_artifact_sha256"]
            or record.get("logical_id") != payload["logical_id"]
        ):
            raise RuntimeError(f"Stale reference artifact: {path}")
        return record
    return atomic_write_artifact(path, payload)


def prepare_references(context: Mapping[str, Any]) -> None:
    root = context["output_root"]
    eos_root = context["eos_root"]
    fp = context["fingerprint"]
    for source in context["entry"]["source_positions"]:
        for label in REUSED_MODES:
            source_mode = _source_mode(label)
            source_path = EOS._control_path(eos_root, source, source_mode)
            source_record = load_artifact(source_path)
            if source_record["extension_fingerprint"] != context["eos_fp"]:
                raise RuntimeError("Referenced EOS control fingerprint mismatch")
            relative = source_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            _write_or_validate_reference(
                _reference_control_path(root, source, label),
                {
                    "schema_version": 1,
                    "scaling_fingerprint": fp,
                    "logical_id": logical_control_id(fp, source, label),
                    "logical_mode": label,
                    "source_path": relative,
                    "source_artifact_sha256": source_record["artifact_sha256"],
                    "source_extension_fingerprint": context["eos_fp"],
                    "created_at": utc_now(),
                },
                fingerprint=fp,
            )
    for item in context["specs"]:
        for label in REUSED_MODES:
            source_mode = _source_mode(label)
            source_path = EOS._run_path(
                eos_root, item.spec_id, source_mode
            )
            source_record = load_artifact(source_path)
            if source_record["extension_fingerprint"] != context["eos_fp"]:
                raise RuntimeError("Referenced EOS run fingerprint mismatch")
            relative = source_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            _write_or_validate_reference(
                _reference_run_path(root, item.spec_id, label),
                {
                    "schema_version": 1,
                    "scaling_fingerprint": fp,
                    "logical_id": logical_run_id(fp, item.spec_id, label),
                    "logical_mode": label,
                    "source_path": relative,
                    "source_artifact_sha256": source_record["artifact_sha256"],
                    "source_extension_fingerprint": context["eos_fp"],
                    "created_at": utc_now(),
                },
                fingerprint=fp,
            )


def _load_referenced(path: Path, fingerprint: str) -> dict[str, Any]:
    ref = load_artifact(path)
    if ref["scaling_fingerprint"] != fingerprint:
        raise RuntimeError("Reference belongs to another scaling campaign")
    source = load_artifact(REPO_ROOT / ref["source_path"])
    if source["artifact_sha256"] != ref["source_artifact_sha256"]:
        raise RuntimeError("Referenced source artifact changed")
    return source


def effective_control(
    context: Mapping[str, Any], source: int, label: str
) -> dict[str, Any]:
    if label in REUSED_MODES:
        return _load_referenced(
            _reference_control_path(context["output_root"], source, label),
            context["fingerprint"],
        )
    return load_artifact(
        _raw_control_path(context["output_root"], source, label)
    )


def effective_run(
    context: Mapping[str, Any], spec_id: str, label: str
) -> dict[str, Any]:
    if label in REUSED_MODES:
        return _load_referenced(
            _reference_run_path(context["output_root"], spec_id, label),
            context["fingerprint"],
        )
    return load_artifact(
        _raw_run_path(context["output_root"], spec_id, label)
    )


def run_new_controls(
    context: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    engine: FT2HookEngine,
) -> None:
    root = context["output_root"]
    fp = context["fingerprint"]
    dataset = load_task_dataset(DATASET_KEY)
    split = DATASETS[DATASET_KEY].evaluation_split
    completed = 0
    for source, dataset_index in zip(
        context["entry"]["source_positions"],
        context["entry"]["dataset_indices"],
    ):
        canonical = context["runner"]._load_screen_record(
            MODEL_KEY, DATASET_KEY, source
        )
        prompt, references = prompt_and_references(
            DATASET_KEY, dataset[split][dataset_index]
        )
        prepared = prepare_prompt(
            tokenizer, prompt, device="cuda", max_input_tokens=1024
        )
        if prepared.prompt_sha256 != canonical["input"]["prompt_sha256"]:
            raise RuntimeError("Scaling control prompt differs from screening")
        for factor in NEW_FACTORS:
            label = FACTOR_LABELS[factor]
            path = _raw_control_path(root, source, label)
            expected_id = logical_control_id(fp, source, label)
            if path.exists():
                record = load_artifact(path)
                if (
                    record.get("scaling_fingerprint") != fp
                    or record.get("logical_control_id") != expected_id
                    or record.get("status") != "completed"
                    or len(record["output"]["token_ids"]) != GENERATION_STEPS
                ):
                    raise RuntimeError(f"Invalid existing scaling control: {path}")
                completed += 1
                continue
            protection = protection_for_factor(factor)
            engine.start_inference(
                fault=None, protection=protection, offline_bounds=None
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
                        "scaling_fingerprint": fp,
                        "logical_control_id": expected_id,
                        "status": "invalid",
                        "source_position": source,
                        "dataset_index": dataset_index,
                        "logical_mode": label,
                        "scaling_factor": factor,
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
            record_engine = output.engine_record
            if (
                record_engine.injection_count != 0
                or len(record_engine.online_bounds)
                != len(adapter.critical_sites)
                or len(output.token_ids) != GENERATION_STEPS
            ):
                raise AssertionError("Scaling clean-control invariant failed")
            views = EOS._view_payload(
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
                    "scaling_fingerprint": fp,
                    "logical_control_id": expected_id,
                    "status": "completed",
                    "model_key": MODEL_KEY,
                    "dataset_key": DATASET_KEY,
                    "source_position": int(source),
                    "dataset_index": int(dataset_index),
                    "logical_mode": label,
                    "scaling_factor": factor,
                    "protection": {
                        "bounds_source": "first_token",
                        "correction": "paper_clamp",
                        "scaling_factor": factor,
                    },
                    "parent_screening_artifact_sha256": canonical["artifact_sha256"],
                    "input": _input_metadata(tokenizer, prompt, prepared),
                    "references": list(references),
                    "frozen_eos_ids": list(context["eos_ids"]),
                    "frozen_first_eos_index": views["pre_first_eos"][
                        "first_eos_index"
                    ],
                    "output": _output_payload(output),
                    "views": views,
                    "engine": record_engine.to_dict(),
                    "started_at": started,
                    "ended_at": utc_now(),
                },
            )
            completed += 1
            print(
                f"[control] {completed}/{EXPECTED_NEW_CONTROLS} "
                f"source={source} factor={factor:.2f} "
                f"full={record['views']['full_fixed_180']['score']['task_correct']} "
                f"pre={record['views']['pre_first_eos']['score']['task_correct']}",
                flush=True,
            )
    if completed != EXPECTED_NEW_CONTROLS:
        raise AssertionError("Wrong number of new scaling controls")


def derive_controls(context: Mapping[str, Any]) -> None:
    root = context["output_root"]
    fp = context["fingerprint"]
    for source in context["entry"]["source_positions"]:
        raw = {
            label: effective_control(context, source, label)
            for label in LOGICAL_MODES
        }
        baseline = {
            view: bool(
                raw[NO_PROTECTION]["views"][view]["score"]["task_correct"]
            )
            for view in ("full_fixed_180", "pre_first_eos")
        }
        for label in LOGICAL_MODES:
            statuses = {
                view: EOS.clean_status(
                    mode_id=(
                        NO_PROTECTION
                        if label == NO_PROTECTION
                        else "paper_clamp_first_token_bounds"
                    ),
                    baseline_correct=baseline[view],
                    mode_correct=bool(
                        raw[label]["views"][view]["score"]["task_correct"]
                    ),
                )
                for view in ("full_fixed_180", "pre_first_eos")
            }
            payload = {
                "schema_version": 1,
                "scaling_fingerprint": fp,
                "logical_control_id": logical_control_id(fp, source, label),
                "source_artifact_sha256": raw[label]["artifact_sha256"],
                "source_position": int(source),
                "dataset_index": raw[label]["dataset_index"],
                "logical_mode": label,
                "scaling_factor": factor_for_label(label),
                "views": {
                    view: {
                        "score": raw[label]["views"][view]["score"],
                        "clean_status": statuses[view],
                    }
                    for view in ("full_fixed_180", "pre_first_eos")
                },
                "frozen_first_eos_index": raw[label][
                    "frozen_first_eos_index"
                ],
                "correction_elements": raw[label]["engine"][
                    "correction_elements"
                ],
                "created_at": utc_now(),
            }
            path = _derived_control_path(root, source, label)
            if path.exists():
                old = load_artifact(path)
                if (
                    old["scaling_fingerprint"] != fp
                    or old["source_artifact_sha256"]
                    != raw[label]["artifact_sha256"]
                    or old["views"] != payload["views"]
                ):
                    raise RuntimeError(f"Stale derived scaling control: {path}")
            else:
                atomic_write_artifact(path, payload)


def run_new_faults(
    context: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    engine: FT2HookEngine,
) -> None:
    root = context["output_root"]
    fp = context["fingerprint"]
    entry = context["entry"]
    dataset = load_task_dataset(DATASET_KEY)
    split = DATASETS[DATASET_KEY].evaluation_split
    completed = 0
    for item in context["specs"]:
        source = entry["source_positions"][item.sample_position]
        canonical = context["runner"]._load_screen_record(
            MODEL_KEY, DATASET_KEY, source
        )
        prompt, references = prompt_and_references(
            DATASET_KEY, dataset[split][item.dataset_index]
        )
        prepared = prepare_prompt(
            tokenizer, prompt, device="cuda", max_input_tokens=1024
        )
        if prepared.prompt_sha256 != canonical["input"]["prompt_sha256"]:
            raise RuntimeError("Scaling fault prompt differs from screening")
        site = adapter.sites_by_key[item.site_key]
        for factor in NEW_FACTORS:
            label = FACTOR_LABELS[factor]
            path = _raw_run_path(root, item.spec_id, label)
            expected_id = logical_run_id(fp, item.spec_id, label)
            if path.exists():
                record = load_artifact(path)
                if (
                    record.get("scaling_fingerprint") != fp
                    or record.get("logical_run_id") != expected_id
                    or record.get("terminal_status") != "completed"
                ):
                    raise RuntimeError(f"Invalid existing scaling run: {path}")
                completed += 1
                continue
            control = effective_control(context, source, label)
            protection = protection_for_factor(factor)
            started = utc_now()
            base = {
                "schema_version": 1,
                "scaling_fingerprint": fp,
                "logical_run_id": expected_id,
                "spec_id": item.spec_id,
                "model_key": MODEL_KEY,
                "dataset_key": DATASET_KEY,
                "source_position": int(source),
                "dataset_index": item.dataset_index,
                "logical_mode": label,
                "scaling_factor": factor,
                "protection": {
                    "bounds_source": "first_token",
                    "correction": "paper_clamp",
                    "scaling_factor": factor,
                },
                "fault": item.to_dict(),
                "target_site": {
                    "key": site.key,
                    "module_path": site.module_path,
                    "critical": site.critical,
                },
                "mode_clean_control_sha256": control["artifact_sha256"],
                "references": list(references),
                "started_at": started,
                "attempt": 1,
            }
            engine.start_inference(
                fault=item, protection=protection, offline_bounds=None
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
                atomic_write_artifact(
                    path,
                    {
                        **base,
                        "terminal_status": "invalid",
                        "due": False,
                        "engine": partial.to_dict(),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        "ended_at": utc_now(),
                    },
                )
                raise RuntimeError(
                    f"Framework-invalid scaling run {expected_id}"
                ) from exc
            record_engine = output.engine_record
            trace = record_engine.injection_trace
            if (
                record_engine.injection_count != 1
                or trace is None
                or not trace.bit_flip_verified
                or trace.hamming_distance != len(item.bit_positions)
                or len(record_engine.online_bounds)
                != len(adapter.critical_sites)
                or len(output.token_ids) != GENERATION_STEPS
            ):
                raise AssertionError("Scaling fault invariant failed")
            views = EOS._view_payload(
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
                    "frozen_first_eos_index": views["pre_first_eos"][
                        "first_eos_index"
                    ],
                    "output": _output_payload(output),
                    "views": views,
                    "engine": record_engine.to_dict(),
                    "ended_at": utc_now(),
                },
            )
            completed += 1
            print(
                f"[run] {completed}/{EXPECTED_NEW_RUNS} "
                f"factor={factor:.2f} full="
                f"{record['views']['full_fixed_180']['score']['task_correct']} "
                f"pre={record['views']['pre_first_eos']['score']['task_correct']}",
                flush=True,
            )
    if completed != EXPECTED_NEW_RUNS:
        raise AssertionError("Wrong number of new scaling fault runs")


def derive_runs(context: Mapping[str, Any]) -> None:
    root = context["output_root"]
    fp = context["fingerprint"]
    entry = context["entry"]
    completed = 0
    for item in context["specs"]:
        source = entry["source_positions"][item.sample_position]
        for label in LOGICAL_MODES:
            raw = effective_run(context, item.spec_id, label)
            control = effective_control(context, source, label)
            clean = load_artifact(
                _derived_control_path(root, source, label)
            )
            views: dict[str, Any] = {}
            for view in ("full_fixed_180", "pre_first_eos"):
                status = clean["views"][view]["clean_status"]
                if view == "full_fixed_180":
                    faulty_tokens = raw["output"]["token_ids"]
                    clean_tokens = control["output"]["token_ids"]
                    faulty_text = raw["output"]["text"]
                    clean_text = control["output"]["text"]
                    reachable = None
                else:
                    faulty_tokens = raw["views"][view]["token_ids"]
                    clean_tokens = control["views"][view]["token_ids"]
                    faulty_text = raw["views"][view]["text"]
                    clean_text = control["views"][view]["text"]
                    reachable = EOS.fault_reachable(
                        item.target_step,
                        control["frozen_first_eos_index"],
                    )
                score = raw["views"][view]["score"]
                exact_tokens = list(faulty_tokens) == list(clean_tokens)
                exact_text = faulty_text == clean_text
                faulty_correct = bool(score["task_correct"])
                outcome = EOS.view_outcome(
                    terminal_status=raw["terminal_status"],
                    clean_status_value=status,
                    reachable=reachable,
                    faulty_correct=faulty_correct,
                    exact_tokens=exact_tokens,
                )
                views[view] = {
                    "score": score,
                    "clean_status": status,
                    "clean_task_correct": bool(
                        control["views"][view]["score"]["task_correct"]
                    ),
                    "faulty_task_correct": faulty_correct,
                    "exact_token_match_mode_clean": exact_tokens,
                    "exact_text_match_mode_clean": exact_text,
                    "fault_reachable": reachable,
                    "outcome": outcome,
                    "evaluable": outcome in EVALUABLE_OUTCOMES,
                    "sdc": outcome == "SDC",
                }
            payload = {
                "schema_version": 1,
                "scaling_fingerprint": fp,
                "logical_run_id": logical_run_id(fp, item.spec_id, label),
                "source_artifact_sha256": raw["artifact_sha256"],
                "spec_id": item.spec_id,
                "source_position": int(source),
                "dataset_index": item.dataset_index,
                "logical_mode": label,
                "scaling_factor": factor_for_label(label),
                "target_step": item.target_step,
                "views": views,
                "created_at": utc_now(),
            }
            path = _derived_run_path(root, item.spec_id, label)
            if path.exists():
                old = load_artifact(path)
                if (
                    old["scaling_fingerprint"] != fp
                    or old["source_artifact_sha256"]
                    != raw["artifact_sha256"]
                    or old["views"] != payload["views"]
                ):
                    raise RuntimeError(f"Stale derived scaling run: {path}")
            else:
                atomic_write_artifact(path, payload)
            completed += 1
    if completed != EXPECTED_LOGICAL_RUNS:
        raise AssertionError("Wrong number of logical scaling runs")


def _verify_inputs(context: Mapping[str, Any]) -> None:
    for role, record in context["immutable"]["inputs"].items():
        if sha256_file(REPO_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"Frozen scaling input changed: {role}")


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


def _mcnemar(improve: int, worsen: int) -> float:
    n = improve + worsen
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(improve, worsen) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def audit_and_summarize(context: Mapping[str, Any]) -> dict[str, Any]:
    _verify_inputs(context)
    root = context["output_root"]
    fp = context["fingerprint"]
    expected_new_controls = {
        f"{source:02d}__{FACTOR_LABELS[factor]}.json"
        for source in context["entry"]["source_positions"]
        for factor in NEW_FACTORS
    }
    expected_new_runs = {
        f"{item.spec_id}__{FACTOR_LABELS[factor]}.json"
        for item in context["specs"]
        for factor in NEW_FACTORS
    }
    expected_ref_controls = {
        f"{source:02d}__{label}.json"
        for source in context["entry"]["source_positions"]
        for label in REUSED_MODES
    }
    expected_ref_runs = {
        f"{item.spec_id}__{label}.json"
        for item in context["specs"]
        for label in REUSED_MODES
    }
    expected_derived_controls = {
        f"{source:02d}__{label}.json"
        for source in context["entry"]["source_positions"]
        for label in LOGICAL_MODES
    }
    expected_derived_runs = {
        f"{item.spec_id}__{label}.json"
        for item in context["specs"]
        for label in LOGICAL_MODES
    }
    for directory, expected in (
        (root / "raw" / "controls", expected_new_controls),
        (root / "raw" / "runs", expected_new_runs),
        (root / "references" / "controls", expected_ref_controls),
        (root / "references" / "runs", expected_ref_runs),
        (root / "derived" / "controls", expected_derived_controls),
        (root / "derived" / "runs", expected_derived_runs),
    ):
        observed = {item.name for item in directory.glob("*.json")}
        if observed != expected:
            raise RuntimeError(
                f"Scaling artifact set mismatch in {directory}: "
                f"missing={len(expected-observed)} extra={len(observed-expected)}"
            )

    control_rows = []
    for source in context["entry"]["source_positions"]:
        for label in LOGICAL_MODES:
            raw = effective_control(context, source, label)
            derived = load_artifact(
                _derived_control_path(root, source, label)
            )
            if (
                len(raw["output"]["token_ids"]) != GENERATION_STEPS
                or raw["engine"]["injection_count"] != 0
                or derived["source_artifact_sha256"]
                != raw["artifact_sha256"]
            ):
                raise RuntimeError("Scaling control audit failed")
            control_rows.append(derived)

    run_rows = []
    raw_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for item in context["specs"]:
        for label in LOGICAL_MODES:
            raw = effective_run(context, item.spec_id, label)
            derived = load_artifact(
                _derived_run_path(root, item.spec_id, label)
            )
            trace = raw["engine"]["injection_trace"]
            if (
                raw["terminal_status"] != "completed"
                or len(raw["output"]["token_ids"]) != GENERATION_STEPS
                or raw["engine"]["injection_count"] != 1
                or trace is None
                or trace["spec_id"] != item.spec_id
                or trace["observed_step"] != item.target_step
                or trace["site_key"] != item.site_key
                or trace["flat_index"] != item.flat_index
                or trace["bit_positions"] != list(item.bit_positions)
                or not trace["bit_flip_verified"]
                or trace["hamming_distance"] != len(item.bit_positions)
                or derived["source_artifact_sha256"]
                != raw["artifact_sha256"]
            ):
                raise RuntimeError("Scaling fault audit failed")
            before = int(trace["before_bits_hex"], 16)
            after = int(trace["after_bits_hex"], 16)
            mask = sum(1 << bit for bit in item.bit_positions)
            if after != (before ^ mask):
                raise RuntimeError("Scaling XOR audit failed")
            run_rows.append(derived)
            raw_rows[(item.spec_id, label)] = raw

    clean_table = {}
    result_table: dict[str, Any] = {}
    paired_table: dict[str, Any] = {}
    telemetry = {}
    for label in LOGICAL_MODES:
        controls = [x for x in control_rows if x["logical_mode"] == label]
        raws = [raw_rows[(item.spec_id, label)] for item in context["specs"]]
        telemetry[label] = {
            "scaling_factor": factor_for_label(label),
            "faults_detected": sum(
                bool(x["engine"]["injection_trace"]["detected"]) for x in raws
            ),
            "fault_detection_rate": sum(
                bool(x["engine"]["injection_trace"]["detected"]) for x in raws
            )
            / EXPECTED_SPECS,
            "mean_fault_run_correction_elements": sum(
                int(x["engine"]["correction_elements"]) for x in raws
            )
            / EXPECTED_SPECS,
            "mean_clean_control_correction_elements": sum(
                int(x["correction_elements"]) for x in controls
            )
            / EXPECTED_PROMPTS,
        }
        clean_table[label] = {
            view: dict(
                sorted(
                    Counter(
                        x["views"][view]["clean_status"] for x in controls
                    ).items()
                )
            )
            for view in ("full_fixed_180", "pre_first_eos")
        }

    by_key = {
        (x["spec_id"], x["logical_mode"]): x for x in run_rows
    }
    for view in ("full_fixed_180", "pre_first_eos"):
        result_table[view] = {}
        paired_table[view] = {}
        for label in LOGICAL_MODES:
            rows = [x for x in run_rows if x["logical_mode"] == label]
            outcomes = Counter(x["views"][view]["outcome"] for x in rows)
            evaluable = sum(
                x["views"][view]["outcome"] in EVALUABLE_OUTCOMES
                for x in rows
            )
            sdc = outcomes["SDC"]
            result_table[view][label] = {
                "planned": EXPECTED_SPECS,
                "evaluable": evaluable,
                "outcome_counts": dict(sorted(outcomes.items())),
                "sdc": sdc,
                "sdc_rate_evaluable": sdc / evaluable if evaluable else None,
                "sdc_wilson95_evaluable": _wilson(sdc, evaluable),
            }
            if label == NO_PROTECTION:
                continue
            improve = worsen = paired_n = 0
            for item in context["specs"]:
                baseline = by_key[(item.spec_id, NO_PROTECTION)]["views"][
                    view
                ]["outcome"]
                treated = by_key[(item.spec_id, label)]["views"][view][
                    "outcome"
                ]
                if (
                    baseline not in EVALUABLE_OUTCOMES
                    or treated not in EVALUABLE_OUTCOMES
                ):
                    continue
                paired_n += 1
                improve += baseline == "SDC" and treated != "SDC"
                worsen += baseline != "SDC" and treated == "SDC"
            paired_table[view][label] = {
                "planned": EXPECTED_SPECS,
                "evaluable_intersection": paired_n,
                "improve": improve,
                "worsen": worsen,
                "exact_mcnemar_p": _mcnemar(improve, worsen),
            }

    audit_payload = {
        "schema_version": 1,
        "scaling_fingerprint": fp,
        "status": "passed",
        "counts": {
            "new_controls": EXPECTED_NEW_CONTROLS,
            "new_fault_runs": EXPECTED_NEW_RUNS,
            "reused_controls": EXPECTED_REFERENCE_CONTROLS,
            "reused_fault_runs": EXPECTED_REFERENCE_RUNS,
            "logical_controls": len(control_rows),
            "logical_fault_runs": len(run_rows),
        },
        "invariants": {
            "parent_eos_audit_passed": True,
            "only_frozen_exponent_specs": True,
            "reused_artifact_hashes_verified": True,
            "one_verified_xor_per_fault_run": True,
            "mode_matched_clean_controls": True,
            "full_primary_and_pre_eos_sensitivity_kept_separate": True,
        },
        "audited_at": utc_now(),
    }
    audit_path = root / "audit.json"
    if audit_path.exists():
        audit = load_artifact(audit_path)
        if audit["scaling_fingerprint"] != fp or audit["status"] != "passed":
            raise RuntimeError("Existing scaling audit is invalid")
    else:
        audit = atomic_write_artifact(audit_path, audit_payload)

    summary_payload = {
        "schema_version": 1,
        "scaling_fingerprint": fp,
        "status": "completed",
        "scope": "qwen2_math_7b/gsm8k/fp16_exponent_bit",
        "primary_metric": "full_fixed_180",
        "post_hoc_sensitivity_metric": "pre_first_eos",
        "warning": (
            "The pre-first-EOS view is post hoc; clean regressions and "
            "non-evaluable cells remain visible in the primary view."
        ),
        "clean_controls": clean_table,
        "fault_results": result_table,
        "paired_vs_no_protection": paired_table,
        "telemetry": telemetry,
        "created_at": utc_now(),
    }
    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = load_artifact(summary_path)
        if summary["scaling_fingerprint"] != fp:
            raise RuntimeError("Existing scaling summary is invalid")
    else:
        summary = atomic_write_artifact(summary_path, summary_payload)
    completion_path = root / "completion.json"
    if completion_path.exists():
        completion = load_artifact(completion_path)
        if completion["scaling_fingerprint"] != fp:
            raise RuntimeError("Existing scaling completion is invalid")
    else:
        completion = atomic_write_artifact(
            completion_path,
            {
                "schema_version": 1,
                "scaling_fingerprint": fp,
                "status": "completed",
                "audit_artifact_sha256": audit["artifact_sha256"],
                "summary_artifact_sha256": summary["artifact_sha256"],
                "completed_at": utc_now(),
            },
        )
    return {"audit": audit, "summary": summary, "completion": completion}


def self_test() -> None:
    assert len(LOGICAL_MODES) == 8
    assert len(set(LOGICAL_MODES)) == 8
    assert len(NEW_FACTORS) == 6
    for factor, label in FACTOR_LABELS.items():
        assert factor_for_label(label) == factor
        protection = protection_for_factor(factor)
        assert protection.scaling_factor == factor
        assert protection.bounds_source is BoundsSource.FIRST_TOKEN
    assert factor_for_label(NO_PROTECTION) is None
    print("[self-test] passed", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".runner.lock").open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(
                lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise SystemExit("Another scaling process holds the runner lock") from exc
        context = build_context(
            args.formal_root.resolve(),
            args.eos_root.resolve(),
            output_root,
        )
        lock = ensure_lock(context)
        prepare_references(context)
        print(
            f"[scaling] fingerprint={context['fingerprint']} "
            f"lock={lock['artifact_sha256']}",
            flush=True,
        )
        if args.prepare_only:
            print("[scaling] preparation complete", flush=True)
            return 0
        if args.audit_only:
            derive_controls(context)
            derive_runs(context)
            result = audit_and_summarize(context)
            print(
                json.dumps(
                    {
                        "status": result["completion"]["status"],
                        "scaling_fingerprint": context["fingerprint"],
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 0

        expected_controls = [
            _raw_control_path(
                output_root, source, FACTOR_LABELS[factor]
            )
            for source in context["entry"]["source_positions"]
            for factor in NEW_FACTORS
        ]
        expected_runs = [
            _raw_run_path(
                output_root, item.spec_id, FACTOR_LABELS[factor]
            )
            for item in context["specs"]
            for factor in NEW_FACTORS
        ]
        if not all(path.exists() for path in expected_controls + expected_runs):
            runner = context["runner"]
            with runner.loaded_model(MODEL_KEY) as (
                model,
                tokenizer,
                adapter,
            ):
                runner.ensure_model_preflight(
                    MODEL_KEY, model, tokenizer, adapter
                )
                with FT2HookEngine(adapter) as engine:
                    run_new_controls(
                        context,
                        model=model,
                        tokenizer=tokenizer,
                        adapter=adapter,
                        engine=engine,
                    )
                    derive_controls(context)
                    run_new_faults(
                        context,
                        model=model,
                        tokenizer=tokenizer,
                        adapter=adapter,
                        engine=engine,
                    )
        torch.cuda.empty_cache()
        derive_controls(context)
        derive_runs(context)
        result = audit_and_summarize(context)
        print(
            json.dumps(
                {
                    "status": result["completion"]["status"],
                    "scaling_fingerprint": context["fingerprint"],
                    "summary": str(output_root / "summary.json"),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
