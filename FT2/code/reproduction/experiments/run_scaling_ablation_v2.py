#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import inspect
import json
import math
import os
import re
import string
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


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
from ft2_formal.campaign import _input_metadata, _output_payload
from ft2_formal.decoding import fixed_greedy_generate, prepare_prompt
from ft2_formal.engine import FT2HookEngine
from ft2_formal.main18k import DEFAULT_MAIN18K_ROOT, Main18kCampaign
from ft2_formal.manifest import load_manifest
from ft2_formal.schema import BoundsSource, Correction, ProtectionSpec
from ft2_formal.tasks import DATASETS, load_task_dataset, prompt_and_references


MODEL_KEY = "qwen2_math_7b"
DATASET_KEY = "gsm8k"
PAIR_ID = f"{MODEL_KEY}__{DATASET_KEY}"
FAULT_TYPE = "fp16_exponent_bit"
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
REUSED_MODES = (NO_PROTECTION, FACTOR_LABELS[2.0])
NEW_FACTORS = tuple(x for x in FACTORS if x != 2.0)
EXPECTED_PROMPTS = 10
EXPECTED_SPECS = 400
EXPECTED_NEW_CONTROLS = EXPECTED_PROMPTS * len(NEW_FACTORS)
EXPECTED_NEW_RUNS = EXPECTED_SPECS * len(NEW_FACTORS)
EXPECTED_LOGICAL_CONTROLS = EXPECTED_PROMPTS * len(LOGICAL_MODES)
EXPECTED_LOGICAL_RUNS = EXPECTED_SPECS * len(LOGICAL_MODES)
GENERATION_STEPS = 180
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "reproduction" / "results" / "scaling_ablation_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired RTX 4090 FT2 scaling-factor ablation using the 400 frozen "
            "Qwen2-Math-7B/GSM8K exponent-fault specs from main_18k_v1"
        )
    )
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN18K_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--audit-only", action="store_true")
    stage.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def protection_for_factor(factor: float) -> ProtectionSpec:
    return ProtectionSpec(
        BoundsSource.FIRST_TOKEN,
        Correction.PAPER_CLAMP,
        float(factor),
    )


def install_generation_update_compat(model: Any) -> bool:
    """Accept the newer optional kwarg when running transformers 4.49."""
    update = model._update_model_kwargs_for_generation
    if "standardize_cache_format" in inspect.signature(update).parameters:
        return False

    def compatible_update(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("standardize_cache_format", None)
        return update(*args, **kwargs)

    model._update_model_kwargs_for_generation = compatible_update
    return True


def factor_for_label(label: str) -> float | None:
    if label == NO_PROTECTION:
        return None
    for factor, candidate in FACTOR_LABELS.items():
        if candidate == label:
            return factor
    raise KeyError(label)


def normalize_gsm8k(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\\[|\\]", "", normalized)
    normalized = re.sub(r"\\text\{([^}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split())


def author_precision_recall_f1(
    prediction: str, ground_truth: str
) -> tuple[float, float, float]:
    prediction_tokens = normalize_gsm8k(prediction).split()
    ground_truth_tokens = normalize_gsm8k(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if not prediction_tokens or not ground_truth_tokens:
        exact = float(prediction_tokens == ground_truth_tokens)
        return exact, exact, exact
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def author_outcome(
    faulty_text: str, clean_text: str, reference: str
) -> tuple[str, dict[str, Any]]:
    precision, recall, f1 = author_precision_recall_f1(
        faulty_text, reference
    )
    if faulty_text == clean_text:
        outcome = "MASKED_IDENTICAL"
    elif recall == 1.0:
        outcome = "MASKED_SEMANTIC"
    else:
        outcome = "SDC"
    return outcome, {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "task_correct": recall == 1.0,
    }


def stable_file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": sha256_file(resolved),
    }


def new_control_path(root: Path, source: int, label: str) -> Path:
    return root / "raw" / "controls" / f"{source:02d}__{label}.json"


def new_run_path(root: Path, spec_id: str, label: str) -> Path:
    return root / "raw" / "runs" / f"{spec_id}__{label}.json"


def main_control_path(main_root: Path, source: int, label: str) -> Path:
    source_mode = (
        "no_protection"
        if label == NO_PROTECTION
        else "paper_clamp_first_token_bounds"
    )
    return (
        main_root
        / "controls"
        / PAIR_ID
        / f"{source:02d}__{source_mode}.json"
    )


def main_run_path(main_root: Path, spec_id: str, label: str) -> Path:
    source_mode = (
        "no_protection"
        if label == NO_PROTECTION
        else "paper_clamp_first_token_bounds"
    )
    return main_root / "runs" / PAIR_ID / f"{spec_id}__{source_mode}.json"


def effective_control(
    context: Mapping[str, Any], source: int, label: str
) -> dict[str, Any]:
    if label in REUSED_MODES:
        return load_artifact(
            main_control_path(context["main_root"], source, label)
        )
    return load_artifact(
        new_control_path(context["output_root"], source, label)
    )


def effective_run(
    context: Mapping[str, Any], spec_id: str, label: str
) -> dict[str, Any]:
    if label in REUSED_MODES:
        return load_artifact(
            main_run_path(context["main_root"], spec_id, label)
        )
    return load_artifact(
        new_run_path(context["output_root"], spec_id, label)
    )


def build_context(main_root: Path, output_root: Path) -> dict[str, Any]:
    main_root = main_root.resolve()
    output_root = output_root.resolve()
    runner = Main18kCampaign(output_root=main_root)
    campaign = load_artifact(main_root / "campaign.json")
    selection = load_artifact(main_root / "selection.json")
    audit = load_artifact(main_root / "audit.json")
    completion = load_artifact(main_root / "completion.json")
    manifest_path = main_root / "manifests" / f"{PAIR_ID}.json"
    manifest_meta, all_specs = load_manifest(manifest_path)
    specs = tuple(
        item for item in all_specs if item.fault_type.value == FAULT_TYPE
    )
    if campaign.get("campaign_fingerprint") != runner.campaign_fingerprint:
        raise RuntimeError("Main campaign fingerprint mismatch")
    if selection.get("campaign_fingerprint") != runner.campaign_fingerprint:
        raise RuntimeError("Main selection fingerprint mismatch")
    if manifest_meta.get("campaign_fingerprint") != runner.campaign_fingerprint:
        raise RuntimeError("Main manifest fingerprint mismatch")
    if audit.get("status") != "passed" or completion.get("status") != "completed":
        raise RuntimeError("Main campaign is not completed and audited")
    entry = selection["pairs"][PAIR_ID]
    if len(entry["source_positions"]) != EXPECTED_PROMPTS:
        raise RuntimeError("Expected ten frozen prompts")
    if len(specs) != EXPECTED_SPECS:
        raise RuntimeError(f"Expected {EXPECTED_SPECS} frozen exponent specs")
    if len({item.spec_id for item in specs}) != EXPECTED_SPECS:
        raise RuntimeError("FaultSpec ids are not unique")
    if any(item.target_step < 1 for item in specs):
        raise RuntimeError("Scaling ablation contains step-zero faults")
    source_files = sorted(
        (REPO_ROOT / "reproduction" / "src" / "ft2_formal").glob("*.py")
    )
    source_files.append(Path(__file__))
    immutable = {
        "schema_version": 1,
        "experiment_id": "ft2_qwen_gsm_exp_scaling_ablation_v2",
        "parent_campaign_fingerprint": runner.campaign_fingerprint,
        "main_inputs": {
            "campaign": stable_file_record(main_root / "campaign.json"),
            "selection": stable_file_record(main_root / "selection.json"),
            "manifest": stable_file_record(manifest_path),
            "audit": stable_file_record(main_root / "audit.json"),
            "completion": stable_file_record(main_root / "completion.json"),
        },
        "source_files": [stable_file_record(path) for path in source_files],
        "scope": {
            "model": MODEL_KEY,
            "dataset": DATASET_KEY,
            "fault_type": FAULT_TYPE,
            "prompts": EXPECTED_PROMPTS,
            "fault_specs_per_factor": EXPECTED_SPECS,
            "trials_per_prompt": EXPECTED_SPECS // EXPECTED_PROMPTS,
        },
        "factors": list(FACTORS),
        "logical_modes": list(LOGICAL_MODES),
        "reused_modes": list(REUSED_MODES),
        "generation": {
            "num_new_tokens": GENERATION_STEPS,
            "greedy": True,
            "stop_on_eos": False,
            "dtype": "float16",
            "batch_size": 1,
        },
        "scoring": {
            "source": "author_repository_qwengsmprotect",
            "reference_policy": "first_reference_only",
            "correct": "normalized_reference_token_recall_equals_1",
            "identical": "decoded_text_equals_same_mode_clean_text",
            "clean_gate": False,
        },
        "expected": {
            "new_controls": EXPECTED_NEW_CONTROLS,
            "new_fault_runs": EXPECTED_NEW_RUNS,
            "logical_controls": EXPECTED_LOGICAL_CONTROLS,
            "logical_fault_runs": EXPECTED_LOGICAL_RUNS,
        },
        "seed": 196,
    }
    fingerprint = json_sha256(immutable)
    return {
        "main_root": main_root,
        "output_root": output_root,
        "runner": runner,
        "campaign": campaign,
        "selection": selection,
        "entry": entry,
        "manifest_path": manifest_path,
        "specs": specs,
        "immutable": immutable,
        "fingerprint": fingerprint,
    }


def ensure_protocol(context: Mapping[str, Any]) -> None:
    root = context["output_root"]
    fp = context["fingerprint"]
    for directory in (root / "raw" / "controls", root / "raw" / "runs"):
        directory.mkdir(parents=True, exist_ok=True)
    lock_path = root / "scaling.lock.json"
    if lock_path.exists():
        lock = load_artifact(lock_path)
        if lock.get("scaling_fingerprint") != fp:
            raise RuntimeError("Existing scaling lock does not match protocol")
    else:
        atomic_write_artifact(
            lock_path,
            {
                "schema_version": 1,
                "scaling_fingerprint": fp,
                "immutable": context["immutable"],
                "created_at": utc_now(),
            },
        )
    protocol_path = root / "protocol.json"
    payload = {
        "schema_version": 1,
        "scaling_fingerprint": fp,
        "status": "frozen",
        "scope": "qwen2_math_7b/gsm8k/fp16_exponent_bit",
        "factors": list(FACTORS),
        "fault_specs_per_factor": EXPECTED_SPECS,
        "new_gpu_inferences": EXPECTED_NEW_CONTROLS + EXPECTED_NEW_RUNS,
        "primary_metric": "author_binary_reference_recall",
        "created_at": utc_now(),
    }
    if protocol_path.exists():
        protocol = load_artifact(protocol_path)
        if protocol.get("scaling_fingerprint") != fp:
            raise RuntimeError("Existing protocol does not match scaling lock")
    else:
        atomic_write_artifact(protocol_path, payload)


def run_new_controls(
    context: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    engine: FT2HookEngine,
) -> None:
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
            path = new_control_path(context["output_root"], source, label)
            if path.exists():
                record = load_artifact(path)
                if (
                    record.get("scaling_fingerprint") != context["fingerprint"]
                    or record.get("status") != "completed"
                    or len(record["output"]["token_ids"]) != GENERATION_STEPS
                ):
                    raise RuntimeError(f"Invalid existing control: {path}")
                completed += 1
                continue
            engine.start_inference(
                fault=None,
                protection=protection_for_factor(factor),
                offline_bounds=None,
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
                        "scaling_fingerprint": context["fingerprint"],
                        "status": "invalid",
                        "source_position": source,
                        "dataset_index": dataset_index,
                        "logical_mode": label,
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
                or len(record_engine.online_bounds) != len(adapter.critical_sites)
                or len(output.token_ids) != GENERATION_STEPS
            ):
                raise AssertionError("Clean-control invariant failed")
            reference = references[0]
            _, clean_score = author_outcome(
                output.text, output.text, reference
            )
            atomic_write_artifact(
                path,
                {
                    "schema_version": 1,
                    "scaling_fingerprint": context["fingerprint"],
                    "status": "completed",
                    "model_key": MODEL_KEY,
                    "dataset_key": DATASET_KEY,
                    "pair_id": PAIR_ID,
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
                    "output": _output_payload(output),
                    "author_score": clean_score,
                    "engine": record_engine.to_dict(),
                    "started_at": started,
                    "ended_at": utc_now(),
                },
            )
            completed += 1
            print(
                f"[control] {completed}/{EXPECTED_NEW_CONTROLS} "
                f"source={source} factor={factor:.2f} "
                f"author_correct={clean_score['task_correct']}",
                flush=True,
            )
    if completed != EXPECTED_NEW_CONTROLS:
        raise AssertionError("Wrong number of new controls")


def run_new_faults(
    context: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    engine: FT2HookEngine,
) -> None:
    dataset = load_task_dataset(DATASET_KEY)
    split = DATASETS[DATASET_KEY].evaluation_split
    entry = context["entry"]
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
            path = new_run_path(context["output_root"], item.spec_id, label)
            if path.exists():
                record = load_artifact(path)
                if (
                    record.get("scaling_fingerprint") != context["fingerprint"]
                    or record.get("terminal_status") != "completed"
                ):
                    raise RuntimeError(f"Invalid existing run: {path}")
                completed += 1
                continue
            control = effective_control(context, source, label)
            started = utc_now()
            base = {
                "schema_version": 1,
                "scaling_fingerprint": context["fingerprint"],
                "spec_id": item.spec_id,
                "model_key": MODEL_KEY,
                "dataset_key": DATASET_KEY,
                "pair_id": PAIR_ID,
                "source_position": int(source),
                "dataset_index": int(item.dataset_index),
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
                fault=item,
                protection=protection_for_factor(factor),
                offline_bounds=None,
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
                        "engine": partial.to_dict(),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        "ended_at": utc_now(),
                    },
                )
                raise
            record_engine = output.engine_record
            trace = record_engine.injection_trace
            if (
                record_engine.injection_count != 1
                or trace is None
                or not trace.bit_flip_verified
                or trace.hamming_distance != len(item.bit_positions)
                or len(record_engine.online_bounds) != len(adapter.critical_sites)
                or len(output.token_ids) != GENERATION_STEPS
            ):
                raise AssertionError("Fault-run invariant failed")
            outcome, score = author_outcome(
                output.text,
                control["output"]["text"],
                references[0],
            )
            atomic_write_artifact(
                path,
                {
                    **base,
                    "terminal_status": "completed",
                    "output": _output_payload(output),
                    "author_score": score,
                    "outcome": outcome,
                    "engine": record_engine.to_dict(),
                    "ended_at": utc_now(),
                },
            )
            completed += 1
            if completed % 25 == 0 or completed == EXPECTED_NEW_RUNS:
                print(
                    f"[run] {completed}/{EXPECTED_NEW_RUNS} "
                    f"spec={item.spec_id[:10]} factor={factor:.2f} "
                    f"outcome={outcome}",
                    flush=True,
                )
    if completed != EXPECTED_NEW_RUNS:
        raise AssertionError("Wrong number of new fault runs")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_mcnemar(improve: int, worsen: int) -> float:
    n = improve + worsen
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(improve, worsen) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def verify_parent_inputs(context: Mapping[str, Any]) -> None:
    for record in context["immutable"]["main_inputs"].values():
        path = REPO_ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Parent input changed: {path}")


def audit_and_summarize(context: Mapping[str, Any]) -> dict[str, Any]:
    verify_parent_inputs(context)
    root = context["output_root"]
    expected_controls = {
        f"{source:02d}__{FACTOR_LABELS[factor]}.json"
        for source in context["entry"]["source_positions"]
        for factor in NEW_FACTORS
    }
    expected_runs = {
        f"{item.spec_id}__{FACTOR_LABELS[factor]}.json"
        for item in context["specs"]
        for factor in NEW_FACTORS
    }
    observed_controls = {
        path.name for path in (root / "raw" / "controls").glob("*.json")
    }
    observed_runs = {
        path.name for path in (root / "raw" / "runs").glob("*.json")
    }
    if observed_controls != expected_controls:
        raise RuntimeError(
            f"Control set mismatch: missing={len(expected_controls-observed_controls)} "
            f"extra={len(observed_controls-expected_controls)}"
        )
    if observed_runs != expected_runs:
        raise RuntimeError(
            f"Run set mismatch: missing={len(expected_runs-observed_runs)} "
            f"extra={len(observed_runs-expected_runs)}"
        )

    controls: dict[tuple[int, str], dict[str, Any]] = {}
    for source in context["entry"]["source_positions"]:
        for label in LOGICAL_MODES:
            record = effective_control(context, source, label)
            if (
                record["engine"]["injection_count"] != 0
                or len(record["output"]["token_ids"]) != GENERATION_STEPS
            ):
                raise RuntimeError("Control audit failed")
            controls[(source, label)] = record

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for item in context["specs"]:
        source = context["entry"]["source_positions"][item.sample_position]
        for label in LOGICAL_MODES:
            raw = effective_run(context, item.spec_id, label)
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
            ):
                raise RuntimeError("Fault-run audit failed")
            before = int(trace["before_bits_hex"], 16)
            after = int(trace["after_bits_hex"], 16)
            mask = sum(1 << bit for bit in item.bit_positions)
            if after != (before ^ mask):
                raise RuntimeError("XOR audit failed")
            reference = raw["references"][0]
            outcome, score = author_outcome(
                raw["output"]["text"],
                controls[(source, label)]["output"]["text"],
                reference,
            )
            if label not in REUSED_MODES:
                if raw.get("outcome") != outcome or raw.get("author_score") != score:
                    raise RuntimeError("Stored author score mismatch")
            rows[(item.spec_id, label)] = {
                "outcome": outcome,
                "score": score,
                "raw": raw,
                "source_position": source,
            }

    clean_table: dict[str, Any] = {}
    factor_table: dict[str, Any] = {}
    telemetry: dict[str, Any] = {}
    for label in LOGICAL_MODES:
        clean_rows = [
            controls[(source, label)]
            for source in context["entry"]["source_positions"]
        ]
        clean_scores = [
            author_precision_recall_f1(x["output"]["text"], x["references"][0])
            for x in clean_rows
        ]
        clean_correct = sum(score[1] == 1.0 for score in clean_scores)
        clean_table[label] = {
            "total": EXPECTED_PROMPTS,
            "author_correct": clean_correct,
            "author_incorrect": EXPECTED_PROMPTS - clean_correct,
            "mean_reference_recall_literal_repo": sum(x[1] for x in clean_scores)
            / EXPECTED_PROMPTS,
        }
        mode_rows = [rows[(item.spec_id, label)] for item in context["specs"]]
        outcomes = Counter(x["outcome"] for x in mode_rows)
        sdc = outcomes["SDC"]
        factor_table[label] = {
            "scaling_factor": factor_for_label(label),
            "planned": EXPECTED_SPECS,
            "evaluable": EXPECTED_SPECS,
            "outcome_counts": dict(sorted(outcomes.items())),
            "sdc": sdc,
            "sdc_rate": sdc / EXPECTED_SPECS,
            "sdc_wilson95": wilson(sdc, EXPECTED_SPECS),
            "mean_reference_recall_literal_repo": sum(
                x["score"]["recall"] for x in mode_rows
            )
            / EXPECTED_SPECS,
        }
        telemetry[label] = {
            "scaling_factor": factor_for_label(label),
            "faults_detected": sum(
                bool(x["raw"]["engine"]["injection_trace"]["detected"])
                for x in mode_rows
            ),
            "fault_detection_rate": sum(
                bool(x["raw"]["engine"]["injection_trace"]["detected"])
                for x in mode_rows
            )
            / EXPECTED_SPECS,
            "mean_fault_run_correction_elements": sum(
                int(x["raw"]["engine"]["correction_elements"])
                for x in mode_rows
            )
            / EXPECTED_SPECS,
            "mean_clean_control_correction_elements": sum(
                int(x["engine"]["correction_elements"]) for x in clean_rows
            )
            / EXPECTED_PROMPTS,
        }

    paired_vs_factor2: dict[str, Any] = {}
    reference_label = FACTOR_LABELS[2.0]
    for factor in FACTORS:
        label = FACTOR_LABELS[factor]
        improve = worsen = both_sdc = both_ok = 0
        for item in context["specs"]:
            reference_sdc = rows[(item.spec_id, reference_label)]["outcome"] == "SDC"
            candidate_sdc = rows[(item.spec_id, label)]["outcome"] == "SDC"
            improve += reference_sdc and not candidate_sdc
            worsen += not reference_sdc and candidate_sdc
            both_sdc += reference_sdc and candidate_sdc
            both_ok += not reference_sdc and not candidate_sdc
        paired_vs_factor2[label] = {
            "reference_factor": 2.0,
            "candidate_better": improve,
            "candidate_worse": worsen,
            "both_sdc": both_sdc,
            "both_non_sdc": both_ok,
            "discordant": improve + worsen,
            "exact_mcnemar_p": exact_mcnemar(improve, worsen),
        }

    summary_payload = {
        "schema_version": 1,
        "analysis_id": "ft2_scaling_ablation_author_logic_v2",
        "scaling_fingerprint": context["fingerprint"],
        "status": "completed",
        "scope": "qwen2_math_7b/gsm8k/fp16_exponent_bit",
        "counts": {
            "prompts": EXPECTED_PROMPTS,
            "fault_specs_per_mode": EXPECTED_SPECS,
            "logical_controls": EXPECTED_LOGICAL_CONTROLS,
            "logical_fault_runs": EXPECTED_LOGICAL_RUNS,
            "new_controls": EXPECTED_NEW_CONTROLS,
            "new_fault_runs": EXPECTED_NEW_RUNS,
        },
        "scoring": context["immutable"]["scoring"],
        "clean_controls": clean_table,
        "factor_results": factor_table,
        "paired_vs_factor_2": paired_vs_factor2,
        "telemetry": telemetry,
        "created_at": utc_now(),
    }
    summary = atomic_write_artifact(root / "summary.json", summary_payload)

    csv_lines = [
        "mode,scaling_factor,runs,masked_identical,masked_semantic,sdc,sdc_rate,"
        "wilson95_low,wilson95_high,mean_recall,clean_correct,clean_total,"
        "fault_detection_rate,mean_fault_corrections,mean_clean_corrections"
    ]
    for label in LOGICAL_MODES:
        result = factor_table[label]
        outcome = result["outcome_counts"]
        telem = telemetry[label]
        clean = clean_table[label]
        csv_lines.append(
            ",".join(
                [
                    label,
                    "" if result["scaling_factor"] is None else f'{result["scaling_factor"]:.2f}',
                    str(result["planned"]),
                    str(outcome.get("MASKED_IDENTICAL", 0)),
                    str(outcome.get("MASKED_SEMANTIC", 0)),
                    str(result["sdc"]),
                    f'{result["sdc_rate"]:.10f}',
                    f'{result["sdc_wilson95"][0]:.10f}',
                    f'{result["sdc_wilson95"][1]:.10f}',
                    f'{result["mean_reference_recall_literal_repo"]:.10f}',
                    str(clean["author_correct"]),
                    str(clean["total"]),
                    f'{telem["fault_detection_rate"]:.10f}',
                    f'{telem["mean_fault_run_correction_elements"]:.6f}',
                    f'{telem["mean_clean_control_correction_elements"]:.6f}',
                ]
            )
        )
    (root / "per_factor.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    audit_payload = {
        "schema_version": 1,
        "scaling_fingerprint": context["fingerprint"],
        "status": "passed",
        "counts": summary_payload["counts"],
        "invariants": {
            "main_parent_completed_and_audited": True,
            "same_400_fault_specs_for_every_mode": True,
            "only_frozen_exponent_specs": True,
            "one_verified_xor_per_fault_run": True,
            "fixed_180_token_generation": True,
            "same_mode_clean_comparator": True,
            "no_clean_evaluability_gate": True,
            "author_reference_recall_scorer": True,
        },
        "summary_artifact_sha256": summary["artifact_sha256"],
        "audited_at": utc_now(),
    }
    audit = atomic_write_artifact(root / "audit.json", audit_payload)
    completion = atomic_write_artifact(
        root / "completion.json",
        {
            "schema_version": 1,
            "scaling_fingerprint": context["fingerprint"],
            "status": "completed",
            "summary_artifact_sha256": summary["artifact_sha256"],
            "audit_artifact_sha256": audit["artifact_sha256"],
            "completed_at": utc_now(),
        },
    )
    checksums = []
    for name in ("summary.json", "per_factor.csv", "audit.json", "completion.json"):
        checksums.append(f"{sha256_file(root / name)}  {name}")
    (root / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "audit": audit, "completion": completion}


def self_test() -> None:
    assert len(FACTORS) == 7
    assert len(NEW_FACTORS) == 6
    assert EXPECTED_NEW_CONTROLS == 60
    assert EXPECTED_NEW_RUNS == 2400
    assert EXPECTED_LOGICAL_RUNS == 3200
    assert normalize_gsm8k(r"The \text{answer} is 5") == "answer is 5"
    _, recall, _ = author_precision_recall_f1("there are 5 people", "5")
    assert recall == 1.0
    assert author_outcome("x 5", "y 5", "5")[0] == "MASKED_SEMANTIC"
    assert author_outcome("x 4", "y 5", "5")[0] == "SDC"
    assert exact_mcnemar(8, 1) == 0.0390625
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".runner.lock").open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another scaling process holds the runner lock") from exc
        context = build_context(args.main_root.resolve(), output_root)
        ensure_protocol(context)
        print(
            f"[scaling-v2] fingerprint={context['fingerprint']} "
            f"specs={len(context['specs'])} new_inferences="
            f"{EXPECTED_NEW_CONTROLS + EXPECTED_NEW_RUNS}",
            flush=True,
        )
        if args.prepare_only:
            print("[scaling-v2] preparation complete", flush=True)
            return 0
        if args.audit_only:
            result = audit_and_summarize(context)
            print(json.dumps({
                "status": result["completion"]["status"],
                "summary": str(output_root / "summary.json"),
            }, indent=2), flush=True)
            return 0

        expected_controls = [
            new_control_path(output_root, source, FACTOR_LABELS[factor])
            for source in context["entry"]["source_positions"]
            for factor in NEW_FACTORS
        ]
        expected_runs = [
            new_run_path(output_root, item.spec_id, FACTOR_LABELS[factor])
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
                if install_generation_update_compat(model):
                    print(
                        "[compat] transformers generation update wrapper enabled",
                        flush=True,
                    )
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
                    run_new_faults(
                        context,
                        model=model,
                        tokenizer=tokenizer,
                        adapter=adapter,
                        engine=engine,
                    )
        torch.cuda.empty_cache()
        result = audit_and_summarize(context)
        print(json.dumps({
            "status": result["completion"]["status"],
            "summary": str(output_root / "summary.json"),
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
