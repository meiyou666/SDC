#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
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
from reproduction.experiments.run_scaling_ablation_v2 import (
    author_outcome,
    author_precision_recall_f1,
    exact_mcnemar,
    install_generation_update_compat,
    stable_file_record,
    wilson,
)


MODEL_KEY = "qwen2_math_7b"
DATASET_KEY = "gsm8k"
PAIR_ID = f"{MODEL_KEY}__{DATASET_KEY}"
FAULT_TYPE = "fp16_exponent_bit"
PROJECTIONS = (
    "v_proj",
    "k_proj",
    "q_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
)
PAPER_EXPECTED_CRITICAL = {
    "v_proj": True,
    "k_proj": False,
    "q_proj": False,
    "o_proj": True,
    "up_proj": True,
    "gate_proj": False,
    "down_proj": True,
}
FULL_MODE = "protect_all_linear"
OMIT_MODES = {name: f"leave_{name}_unprotected" for name in PROJECTIONS}
LOGICAL_MODES = (FULL_MODE,) + tuple(OMIT_MODES[name] for name in PROJECTIONS)
MODE_TO_OMIT = {FULL_MODE: None, **{value: key for key, value in OMIT_MODES.items()}}
EXPECTED_PROMPTS = 10
EXPECTED_SPECS = 400
EXPECTED_CONTROLS = EXPECTED_PROMPTS * len(LOGICAL_MODES)
EXPECTED_RUNS = EXPECTED_SPECS * len(LOGICAL_MODES)
GENERATION_STEPS = 180
SCALING_FACTOR = 2.0
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "reproduction" / "results" / "critical_layer_identification_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-Section-4.1-style critical-layer identification on RTX 4090: "
            "protect every linear layer except one projection class"
        )
    )
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN18K_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--audit-only", action="store_true")
    stage.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def protection() -> ProtectionSpec:
    return ProtectionSpec(
        BoundsSource.FIRST_TOKEN,
        Correction.PAPER_CLAMP,
        SCALING_FACTOR,
    )


def protected_keys(adapter: Any, mode: str) -> tuple[str, ...]:
    omitted = MODE_TO_OMIT[mode]
    keys = tuple(
        site.key for site in adapter.sites if site.projection != omitted
    )
    expected = len(adapter.sites) if omitted is None else len(adapter.sites) - 28
    if len(keys) != expected:
        raise AssertionError(
            f"Unexpected protection cardinality for {mode}: {len(keys)} != {expected}"
        )
    return keys


def control_path(root: Path, source: int, mode: str) -> Path:
    return root / "raw" / "controls" / f"{source:02d}__{mode}.json"


def run_path(root: Path, spec_id: str, mode: str) -> Path:
    return root / "raw" / "runs" / f"{spec_id}__{mode}.json"


def baseline_control_path(main_root: Path, source: int) -> Path:
    return (
        main_root
        / "controls"
        / PAIR_ID
        / f"{source:02d}__no_protection.json"
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
        raise RuntimeError(f"Expected {EXPECTED_SPECS} frozen EXP specs")
    if len({item.spec_id for item in specs}) != EXPECTED_SPECS:
        raise RuntimeError("FaultSpec ids are not unique")
    if any(item.target_step < 1 for item in specs):
        raise RuntimeError("Critical-layer study requires post-first-token faults")
    if set(item.projection for item in specs).difference(PROJECTIONS):
        raise RuntimeError("Manifest contains an unknown Qwen projection")

    source_files = sorted(
        (REPO_ROOT / "reproduction" / "src" / "ft2_formal").glob("*.py")
    )
    source_files.extend(
        [
            REPO_ROOT / "reproduction" / "experiments" / "run_scaling_ablation_v2.py",
            REPO_ROOT / "performance" / "sigcode" / "evaluation" / "qwengsmprotect.py",
            REPO_ROOT / "performance" / "sigcode" / "modeling" / "modeling_qwen2_protected.py",
            Path(__file__),
        ]
    )
    immutable = {
        "schema_version": 1,
        "experiment_id": "ft2_qwen_gsm_critical_layer_identification_v1",
        "paper_method": (
            "Protect every linear layer except the projection class under test; "
            "inject faults across all linear layers and measure SDC."
        ),
        "paper_mapping": "Qwen Table-1 architectural analogue, not the GPT-J/SQuAD Figure-6 model",
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
            "fault_specs_per_mode": EXPECTED_SPECS,
            "trials_per_prompt": EXPECTED_SPECS // EXPECTED_PROMPTS,
            "projection_classes": list(PROJECTIONS),
        },
        "logical_modes": list(LOGICAL_MODES),
        "mode_to_omitted_projection": MODE_TO_OMIT,
        "paper_expected_critical": PAPER_EXPECTED_CRITICAL,
        "protection": {
            "bounds_source": "first_token",
            "correction": "paper_clamp",
            "scaling_factor": SCALING_FACTOR,
            "full_mode_scope": "all_196_qwen_linear_sites",
        },
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
            "identical": "decoded_faulty_text_equals_unprotected_clean_text",
            "clean_comparator": "frozen_no_protection_control_from_main_18k_v1",
            "clean_gate": False,
        },
        "statistics": {
            "interval": "Wilson 95%",
            "paired_test": "two-sided exact McNemar versus protect_all_linear",
            "inferential_critical": "candidate_worse > candidate_better and p < 0.05",
        },
        "expected": {
            "controls": EXPECTED_CONTROLS,
            "fault_runs": EXPECTED_RUNS,
            "gpu_inferences": EXPECTED_CONTROLS + EXPECTED_RUNS,
            "nominal_4090_hours_at_10_seconds_per_inference": (
                EXPECTED_CONTROLS + EXPECTED_RUNS
            ) * 10 / 3600,
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
    for directory in (root / "raw" / "controls", root / "raw" / "runs"):
        directory.mkdir(parents=True, exist_ok=True)
    lock_path = root / "critical_layers.lock.json"
    if lock_path.exists():
        lock = load_artifact(lock_path)
        if lock.get("critical_layers_fingerprint") != context["fingerprint"]:
            raise RuntimeError("Existing critical-layer lock differs from protocol")
    else:
        atomic_write_artifact(
            lock_path,
            {
                "schema_version": 1,
                "critical_layers_fingerprint": context["fingerprint"],
                "immutable": context["immutable"],
                "created_at": utc_now(),
            },
        )
    protocol_path = root / "protocol.json"
    payload = {
        "schema_version": 1,
        "critical_layers_fingerprint": context["fingerprint"],
        "status": "frozen",
        "scope": "qwen2_math_7b/gsm8k/fp16_exponent_bit",
        "paper_method": context["immutable"]["paper_method"],
        "logical_modes": list(LOGICAL_MODES),
        "fault_specs_per_mode": EXPECTED_SPECS,
        "gpu_inferences": EXPECTED_CONTROLS + EXPECTED_RUNS,
        "primary_metric": "author_binary_reference_recall",
        "created_at": utc_now(),
    }
    if protocol_path.exists():
        protocol = load_artifact(protocol_path)
        if protocol.get("critical_layers_fingerprint") != context["fingerprint"]:
            raise RuntimeError("Existing protocol differs from critical-layer lock")
    else:
        atomic_write_artifact(protocol_path, payload)


def baseline_controls(context: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result = {}
    for source in context["entry"]["source_positions"]:
        record = load_artifact(baseline_control_path(context["main_root"], source))
        if record.get("status") != "completed":
            raise RuntimeError(f"Invalid unprotected clean control for source {source}")
        result[int(source)] = record
    return result


def run_mode(
    context: Mapping[str, Any],
    *,
    mode: str,
    model: Any,
    tokenizer: Any,
    adapter: Any,
) -> None:
    dataset = load_task_dataset(DATASET_KEY)
    split = DATASETS[DATASET_KEY].evaluation_split
    baselines = baseline_controls(context)
    keys = protected_keys(adapter, mode)
    omitted = MODE_TO_OMIT[mode]
    protected_projection_classes = [
        name for name in PROJECTIONS if name != omitted
    ]
    with FT2HookEngine(adapter, protected_site_keys=keys) as engine:
        completed_controls = 0
        for source, dataset_index in zip(
            context["entry"]["source_positions"],
            context["entry"]["dataset_indices"],
        ):
            path = control_path(context["output_root"], source, mode)
            if path.exists():
                record = load_artifact(path)
                if (
                    record.get("critical_layers_fingerprint") != context["fingerprint"]
                    or record.get("status") != "completed"
                    or len(record["output"]["token_ids"]) != GENERATION_STEPS
                ):
                    raise RuntimeError(f"Invalid existing control: {path}")
                completed_controls += 1
                continue
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
                raise RuntimeError("Control prompt differs from frozen screening")
            engine.start_inference(
                fault=None,
                protection=protection(),
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
                        "critical_layers_fingerprint": context["fingerprint"],
                        "status": "invalid",
                        "source_position": source,
                        "logical_mode": mode,
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
                or len(record_engine.online_bounds) != len(keys)
                or len(output.token_ids) != GENERATION_STEPS
            ):
                raise AssertionError("Clean-control invariant failed")
            reference = references[0]
            _, clean_score = author_outcome(output.text, output.text, reference)
            baseline = baselines[int(source)]
            atomic_write_artifact(
                path,
                {
                    "schema_version": 1,
                    "critical_layers_fingerprint": context["fingerprint"],
                    "status": "completed",
                    "model_key": MODEL_KEY,
                    "dataset_key": DATASET_KEY,
                    "pair_id": PAIR_ID,
                    "source_position": int(source),
                    "dataset_index": int(dataset_index),
                    "logical_mode": mode,
                    "omitted_projection": omitted,
                    "protected_projection_classes": protected_projection_classes,
                    "protected_site_count": len(keys),
                    "protection": {
                        "bounds_source": "first_token",
                        "correction": "paper_clamp",
                        "scaling_factor": SCALING_FACTOR,
                    },
                    "unprotected_clean_control_sha256": baseline["artifact_sha256"],
                    "equals_unprotected_clean": (
                        output.text == baseline["output"]["text"]
                    ),
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
            completed_controls += 1
            print(
                f"[control] mode={mode} {completed_controls}/{EXPECTED_PROMPTS} "
                f"source={source} clean_equals_base={output.text == baseline['output']['text']}",
                flush=True,
            )

        completed_runs = 0
        for item in context["specs"]:
            path = run_path(context["output_root"], item.spec_id, mode)
            if path.exists():
                record = load_artifact(path)
                if (
                    record.get("critical_layers_fingerprint") != context["fingerprint"]
                    or record.get("terminal_status") != "completed"
                ):
                    raise RuntimeError(f"Invalid existing run: {path}")
                completed_runs += 1
                continue
            source = context["entry"]["source_positions"][item.sample_position]
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
                raise RuntimeError("Fault prompt differs from frozen screening")
            baseline = baselines[int(source)]
            mode_control = load_artifact(
                control_path(context["output_root"], source, mode)
            )
            site = adapter.sites_by_key[item.site_key]
            target_protected = item.site_key in set(keys)
            started = utc_now()
            base = {
                "schema_version": 1,
                "critical_layers_fingerprint": context["fingerprint"],
                "spec_id": item.spec_id,
                "model_key": MODEL_KEY,
                "dataset_key": DATASET_KEY,
                "pair_id": PAIR_ID,
                "source_position": int(source),
                "dataset_index": int(item.dataset_index),
                "logical_mode": mode,
                "omitted_projection": omitted,
                "protected_site_count": len(keys),
                "fault": item.to_dict(),
                "target_site": {
                    "key": site.key,
                    "module_path": site.module_path,
                    "projection": site.projection,
                    "paper_expected_critical": PAPER_EXPECTED_CRITICAL[site.projection],
                    "protected_in_this_mode": target_protected,
                },
                "unprotected_clean_control_sha256": baseline["artifact_sha256"],
                "mode_clean_control_sha256": mode_control["artifact_sha256"],
                "references": list(references),
                "started_at": started,
                "attempt": 1,
            }
            engine.start_inference(
                fault=item,
                protection=protection(),
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
                or len(record_engine.online_bounds) != len(keys)
                or len(output.token_ids) != GENERATION_STEPS
                or (target_protected and trace.bounds is None)
                or (not target_protected and trace.bounds is not None)
            ):
                raise AssertionError("Fault-run invariant failed")
            outcome, score = author_outcome(
                output.text,
                baseline["output"]["text"],
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
            completed_runs += 1
            if completed_runs % 25 == 0 or completed_runs == EXPECTED_SPECS:
                print(
                    f"[run] mode={mode} {completed_runs}/{EXPECTED_SPECS} "
                    f"spec={item.spec_id[:10]} target={item.projection} outcome={outcome}",
                    flush=True,
                )


def verify_parent_inputs(context: Mapping[str, Any]) -> None:
    for record in context["immutable"]["main_inputs"].values():
        path = REPO_ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Parent input changed: {path}")


def outcome_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["outcome"] for row in rows).items()))


def audit_and_summarize(context: Mapping[str, Any]) -> dict[str, Any]:
    verify_parent_inputs(context)
    root = context["output_root"]
    expected_controls = {
        f"{source:02d}__{mode}.json"
        for source in context["entry"]["source_positions"]
        for mode in LOGICAL_MODES
    }
    expected_runs = {
        f"{item.spec_id}__{mode}.json"
        for item in context["specs"]
        for mode in LOGICAL_MODES
    }
    observed_controls = {
        path.name for path in (root / "raw" / "controls").glob("*.json")
    }
    observed_runs = {
        path.name for path in (root / "raw" / "runs").glob("*.json")
    }
    if observed_controls != expected_controls:
        raise RuntimeError(
            f"Control mismatch: missing={len(expected_controls-observed_controls)} "
            f"extra={len(observed_controls-expected_controls)}"
        )
    if observed_runs != expected_runs:
        raise RuntimeError(
            f"Run mismatch: missing={len(expected_runs-observed_runs)} "
            f"extra={len(observed_runs-expected_runs)}"
        )

    baselines = baseline_controls(context)
    controls: dict[tuple[int, str], dict[str, Any]] = {}
    for source in context["entry"]["source_positions"]:
        for mode in LOGICAL_MODES:
            record = load_artifact(control_path(root, source, mode))
            expected_sites = 196 if mode == FULL_MODE else 168
            if (
                record["status"] != "completed"
                or record["engine"]["injection_count"] != 0
                or len(record["engine"]["online_bounds"]) != expected_sites
                or len(record["output"]["token_ids"]) != GENERATION_STEPS
                or record["unprotected_clean_control_sha256"]
                != baselines[int(source)]["artifact_sha256"]
            ):
                raise RuntimeError("Control audit failed")
            controls[(int(source), mode)] = record

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for item in context["specs"]:
        source = int(context["entry"]["source_positions"][item.sample_position])
        for mode in LOGICAL_MODES:
            raw = load_artifact(run_path(root, item.spec_id, mode))
            trace = raw["engine"]["injection_trace"]
            expected_target_protected = MODE_TO_OMIT[mode] != item.projection
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
                or bool(trace["bounds"] is not None) != expected_target_protected
                or raw["unprotected_clean_control_sha256"]
                != baselines[source]["artifact_sha256"]
            ):
                raise RuntimeError("Fault-run audit failed")
            before = int(trace["before_bits_hex"], 16)
            after = int(trace["after_bits_hex"], 16)
            mask = sum(1 << bit for bit in item.bit_positions)
            if after != (before ^ mask):
                raise RuntimeError("XOR audit failed")
            outcome, score = author_outcome(
                raw["output"]["text"],
                baselines[source]["output"]["text"],
                raw["references"][0],
            )
            if raw.get("outcome") != outcome or raw.get("author_score") != score:
                raise RuntimeError("Stored author score mismatch")
            rows[(item.spec_id, mode)] = {
                "outcome": outcome,
                "score": score,
                "raw": raw,
                "spec": item,
                "source_position": source,
            }

    clean_table: dict[str, Any] = {}
    mode_table: dict[str, Any] = {}
    telemetry: dict[str, Any] = {}
    by_injected_projection: dict[str, Any] = {}
    for mode in LOGICAL_MODES:
        clean_rows = [
            controls[(int(source), mode)]
            for source in context["entry"]["source_positions"]
        ]
        clean_scores = [
            author_precision_recall_f1(row["output"]["text"], row["references"][0])
            for row in clean_rows
        ]
        clean_correct = sum(score[1] == 1.0 for score in clean_scores)
        clean_equal = sum(bool(row["equals_unprotected_clean"]) for row in clean_rows)
        clean_table[mode] = {
            "total": EXPECTED_PROMPTS,
            "author_correct": clean_correct,
            "author_incorrect": EXPECTED_PROMPTS - clean_correct,
            "equals_unprotected_clean": clean_equal,
            "differs_from_unprotected_clean": EXPECTED_PROMPTS - clean_equal,
            "mean_reference_recall_literal_repo": sum(x[1] for x in clean_scores)
            / EXPECTED_PROMPTS,
        }
        mode_rows = [rows[(item.spec_id, mode)] for item in context["specs"]]
        outcomes = outcome_counts(mode_rows)
        sdc = outcomes.get("SDC", 0)
        mode_table[mode] = {
            "omitted_projection": MODE_TO_OMIT[mode],
            "planned": EXPECTED_SPECS,
            "evaluable": EXPECTED_SPECS,
            "outcome_counts": outcomes,
            "sdc": sdc,
            "sdc_rate": sdc / EXPECTED_SPECS,
            "sdc_wilson95": wilson(sdc, EXPECTED_SPECS),
            "mean_reference_recall_literal_repo": sum(
                row["score"]["recall"] for row in mode_rows
            ) / EXPECTED_SPECS,
        }
        telemetry[mode] = {
            "faults_detected": sum(
                bool(row["raw"]["engine"]["injection_trace"]["detected"])
                for row in mode_rows
            ),
            "fault_detection_rate": sum(
                bool(row["raw"]["engine"]["injection_trace"]["detected"])
                for row in mode_rows
            ) / EXPECTED_SPECS,
            "mean_fault_run_correction_elements": sum(
                int(row["raw"]["engine"]["correction_elements"])
                for row in mode_rows
            ) / EXPECTED_SPECS,
            "mean_clean_control_correction_elements": sum(
                int(row["engine"]["correction_elements"]) for row in clean_rows
            ) / EXPECTED_PROMPTS,
        }
        by_injected_projection[mode] = {}
        for projection in PROJECTIONS:
            subset = [
                row for row in mode_rows if row["spec"].projection == projection
            ]
            subset_outcomes = outcome_counts(subset)
            subset_sdc = subset_outcomes.get("SDC", 0)
            by_injected_projection[mode][projection] = {
                "runs": len(subset),
                "outcome_counts": subset_outcomes,
                "sdc": subset_sdc,
                "sdc_rate": subset_sdc / len(subset) if subset else None,
                "sdc_wilson95": wilson(subset_sdc, len(subset)),
            }

    full_sdc = mode_table[FULL_MODE]["sdc"]
    paired: dict[str, Any] = {}
    classification: dict[str, Any] = {}
    for projection in PROJECTIONS:
        mode = OMIT_MODES[projection]
        candidate_worse = candidate_better = both_sdc = both_ok = 0
        for item in context["specs"]:
            full_bad = rows[(item.spec_id, FULL_MODE)]["outcome"] == "SDC"
            candidate_bad = rows[(item.spec_id, mode)]["outcome"] == "SDC"
            candidate_worse += not full_bad and candidate_bad
            candidate_better += full_bad and not candidate_bad
            both_sdc += full_bad and candidate_bad
            both_ok += not full_bad and not candidate_bad
        p_value = exact_mcnemar(candidate_worse, candidate_better)
        inferential_critical = (
            candidate_worse > candidate_better and p_value < 0.05
        )
        descriptive_critical = mode_table[mode]["sdc"] > full_sdc
        paired[projection] = {
            "mode": mode,
            "reference_mode": FULL_MODE,
            "candidate_worse": candidate_worse,
            "candidate_better": candidate_better,
            "both_sdc": both_sdc,
            "both_non_sdc": both_ok,
            "discordant": candidate_worse + candidate_better,
            "exact_mcnemar_p": p_value,
            "sdc_rate_difference": (
                mode_table[mode]["sdc_rate"] - mode_table[FULL_MODE]["sdc_rate"]
            ),
        }
        classification[projection] = {
            "paper_expected_critical": PAPER_EXPECTED_CRITICAL[projection],
            "descriptive_critical": descriptive_critical,
            "inferential_critical": inferential_critical,
            "inferential_agrees_with_paper": (
                inferential_critical == PAPER_EXPECTED_CRITICAL[projection]
            ),
            "descriptive_agrees_with_paper": (
                descriptive_critical == PAPER_EXPECTED_CRITICAL[projection]
            ),
        }

    summary_payload = {
        "schema_version": 1,
        "analysis_id": "ft2_critical_layer_identification_author_logic_v1",
        "critical_layers_fingerprint": context["fingerprint"],
        "status": "completed",
        "scope": "qwen2_math_7b/gsm8k/fp16_exponent_bit",
        "paper_mapping": context["immutable"]["paper_mapping"],
        "counts": {
            "prompts": EXPECTED_PROMPTS,
            "fault_specs_per_mode": EXPECTED_SPECS,
            "logical_modes": len(LOGICAL_MODES),
            "controls": EXPECTED_CONTROLS,
            "fault_runs": EXPECTED_RUNS,
            "gpu_inferences": EXPECTED_CONTROLS + EXPECTED_RUNS,
        },
        "scoring": context["immutable"]["scoring"],
        "clean_controls": clean_table,
        "mode_results": mode_table,
        "paired_vs_all_linear_protected": paired,
        "criticality_classification": classification,
        "by_injected_projection": by_injected_projection,
        "telemetry": telemetry,
        "created_at": utc_now(),
    }
    summary = atomic_write_artifact(root / "summary.json", summary_payload)

    csv_lines = [
        "mode,omitted_projection,runs,masked_identical,masked_semantic,sdc,sdc_rate,"
        "wilson95_low,wilson95_high,clean_equal_base,clean_correct,clean_total,"
        "fault_detection_rate,mean_fault_corrections,mean_clean_corrections"
    ]
    for mode in LOGICAL_MODES:
        result = mode_table[mode]
        counts = result["outcome_counts"]
        clean = clean_table[mode]
        telem = telemetry[mode]
        csv_lines.append(
            ",".join(
                [
                    mode,
                    result["omitted_projection"] or "",
                    str(result["planned"]),
                    str(counts.get("MASKED_IDENTICAL", 0)),
                    str(counts.get("MASKED_SEMANTIC", 0)),
                    str(result["sdc"]),
                    f'{result["sdc_rate"]:.10f}',
                    f'{result["sdc_wilson95"][0]:.10f}',
                    f'{result["sdc_wilson95"][1]:.10f}',
                    str(clean["equals_unprotected_clean"]),
                    str(clean["author_correct"]),
                    str(clean["total"]),
                    f'{telem["fault_detection_rate"]:.10f}',
                    f'{telem["mean_fault_run_correction_elements"]:.6f}',
                    f'{telem["mean_clean_control_correction_elements"]:.6f}',
                ]
            )
        )
    (root / "per_mode.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    class_lines = [
        "projection,paper_expected_critical,descriptive_critical,inferential_critical,"
        "omit_sdc,omit_sdc_rate,full_sdc,full_sdc_rate,candidate_worse,"
        "candidate_better,mcnemar_p"
    ]
    for projection in PROJECTIONS:
        mode = OMIT_MODES[projection]
        result = mode_table[mode]
        pair = paired[projection]
        cls = classification[projection]
        class_lines.append(
            ",".join(
                [
                    projection,
                    str(cls["paper_expected_critical"]).lower(),
                    str(cls["descriptive_critical"]).lower(),
                    str(cls["inferential_critical"]).lower(),
                    str(result["sdc"]),
                    f'{result["sdc_rate"]:.10f}',
                    str(mode_table[FULL_MODE]["sdc"]),
                    f'{mode_table[FULL_MODE]["sdc_rate"]:.10f}',
                    str(pair["candidate_worse"]),
                    str(pair["candidate_better"]),
                    f'{pair["exact_mcnemar_p"]:.12g}',
                ]
            )
        )
    (root / "criticality.csv").write_text(
        "\n".join(class_lines) + "\n", encoding="utf-8"
    )

    audit_payload = {
        "schema_version": 1,
        "critical_layers_fingerprint": context["fingerprint"],
        "status": "passed",
        "counts": summary_payload["counts"],
        "invariants": {
            "main_parent_completed_and_audited": True,
            "same_400_fault_specs_for_every_mode": True,
            "all_linear_sites_protected_except_tested_projection": True,
            "only_frozen_exponent_specs": True,
            "one_verified_xor_per_fault_run": True,
            "target_bounds_presence_matches_protection_scope": True,
            "fixed_180_token_generation": True,
            "unprotected_clean_comparator_for_every_mode": True,
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
            "critical_layers_fingerprint": context["fingerprint"],
            "status": "completed",
            "summary_artifact_sha256": summary["artifact_sha256"],
            "audit_artifact_sha256": audit["artifact_sha256"],
            "completed_at": utc_now(),
        },
    )
    checksums = []
    for name in (
        "summary.json",
        "per_mode.csv",
        "criticality.csv",
        "audit.json",
        "completion.json",
    ):
        checksums.append(f"{sha256_file(root / name)}  {name}")
    (root / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "audit": audit, "completion": completion}


def self_test() -> None:
    assert len(PROJECTIONS) == 7
    assert len(LOGICAL_MODES) == 8
    assert EXPECTED_CONTROLS == 80
    assert EXPECTED_RUNS == 3200
    assert EXPECTED_CONTROLS + EXPECTED_RUNS == 3280
    assert sum(PAPER_EXPECTED_CRITICAL.values()) == 4
    assert MODE_TO_OMIT[FULL_MODE] is None
    assert MODE_TO_OMIT["leave_gate_proj_unprotected"] == "gate_proj"
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
            raise SystemExit("Another critical-layer process holds the runner lock") from exc
        context = build_context(args.main_root.resolve(), output_root)
        ensure_protocol(context)
        print(
            f"[critical-layers] fingerprint={context['fingerprint']} "
            f"modes={len(LOGICAL_MODES)} specs_per_mode={len(context['specs'])} "
            f"gpu_inferences={EXPECTED_CONTROLS + EXPECTED_RUNS}",
            flush=True,
        )
        if args.prepare_only:
            print("[critical-layers] preparation complete", flush=True)
            return 0
        if args.audit_only:
            result = audit_and_summarize(context)
            print(
                json.dumps(
                    {
                        "status": result["completion"]["status"],
                        "summary": str(output_root / "summary.json"),
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 0

        expected_controls = [
            control_path(output_root, source, mode)
            for source in context["entry"]["source_positions"]
            for mode in LOGICAL_MODES
        ]
        expected_runs = [
            run_path(output_root, item.spec_id, mode)
            for item in context["specs"]
            for mode in LOGICAL_MODES
        ]
        if not all(path.exists() for path in expected_controls + expected_runs):
            runner = context["runner"]
            with runner.loaded_model(MODEL_KEY) as (model, tokenizer, adapter):
                if install_generation_update_compat(model):
                    print(
                        "[compat] transformers generation update wrapper enabled",
                        flush=True,
                    )
                runner.ensure_model_preflight(MODEL_KEY, model, tokenizer, adapter)
                for index, mode in enumerate(LOGICAL_MODES, start=1):
                    print(
                        f"[mode] {index}/{len(LOGICAL_MODES)} {mode} "
                        f"omit={MODE_TO_OMIT[mode]}",
                        flush=True,
                    )
                    run_mode(
                        context,
                        mode=mode,
                        model=model,
                        tokenizer=tokenizer,
                        adapter=adapter,
                    )
        torch.cuda.empty_cache()
        result = audit_and_summarize(context)
        print(
            json.dumps(
                {
                    "status": result["completion"]["status"],
                    "summary": str(output_root / "summary.json"),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
