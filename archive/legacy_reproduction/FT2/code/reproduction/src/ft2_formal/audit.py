from __future__ import annotations

import csv
import io
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    atomic_write_artifact,
    atomic_write_text,
    canonical_json_bytes,
    load_artifact,
    sha256_file,
    stable_run_id,
    utc_now,
)
from .campaign import MAIN_MODES, PAIR_ORDER, pair_id
from .manifest import load_manifest
from .schema import FaultSpec
from .tasks import DATASETS, MODELS


def _wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half, center + half


def _expected_critical_sites(model_key: str) -> int:
    return 96 if model_key == "opt_2_7b" else 112


def _expected_candidate_sites(model_key: str) -> int:
    return 192 if model_key == "opt_2_7b" else 196


def _run_path(
    root: Path,
    model_key: str,
    dataset_key: str,
    spec_id: str,
    mode_id: str,
) -> Path:
    return (
        root
        / "runs"
        / pair_id(model_key, dataset_key)
        / f"{spec_id}__{mode_id}.json"
    )


def _trace_invariants(
    record: Mapping[str, Any],
    spec: FaultSpec,
) -> None:
    engine = record["engine"]
    if int(engine["injection_count"]) != 1:
        raise AssertionError("Fault run did not execute exactly one XOR")
    trace = engine.get("injection_trace")
    if not isinstance(trace, dict):
        raise AssertionError("Fault run has no injection trace")
    before = int(trace["before_bits_hex"], 16)
    after = int(trace["after_bits_hex"], 16)
    mask = 0
    for bit in spec.bit_positions:
        mask |= 1 << bit
    if after != (before ^ mask):
        raise AssertionError("Observed FP16 word does not match XOR mask")
    if (before ^ after).bit_count() != len(spec.bit_positions):
        raise AssertionError("Observed FP16 Hamming distance is wrong")
    if trace["site_key"] != spec.site_key:
        raise AssertionError("Observed site differs from manifest")
    if int(trace["observed_step"]) != spec.target_step:
        raise AssertionError("Observed generation step differs from manifest")
    if int(trace["flat_index"]) != spec.flat_index:
        raise AssertionError("Observed scalar coordinate differs from manifest")
    if tuple(int(value) for value in trace["bit_positions"]) != spec.bit_positions:
        raise AssertionError("Observed bit positions differ from manifest")


def _control_invariants(
    root: Path,
    campaign_fingerprint: str,
    model_key: str,
    dataset_key: str,
    selection_entry: Mapping[str, Any],
) -> None:
    steps = DATASETS[dataset_key].generation_steps
    for source_position in selection_entry["source_positions"]:
        for protection in MAIN_MODES:
            path = (
                root
                / "controls"
                / pair_id(model_key, dataset_key)
                / f"{source_position:02d}__{protection.mode_id}.json"
            )
            control = load_artifact(path)
            if control["campaign_fingerprint"] != campaign_fingerprint:
                raise AssertionError("Control campaign fingerprint mismatch")
            if control["status"] != "completed":
                raise AssertionError("A clean mode control is not complete")
            if not control["score"]["task_correct"]:
                raise AssertionError("A clean mode control is task-incorrect")
            if len(control["output"]["token_ids"]) != steps:
                raise AssertionError("A clean mode control has wrong token length")


def _bounds_invariants(
    root: Path,
    campaign_fingerprint: str,
    model_key: str,
    dataset_key: str,
) -> dict[str, Any]:
    path = (
        root
        / "offline_bounds"
        / f"{pair_id(model_key, dataset_key)}.json"
    )
    artifact = load_artifact(path)
    if artifact["campaign_fingerprint"] != campaign_fingerprint:
        raise AssertionError("Bounds campaign fingerprint mismatch")
    if artifact["selection"]["count"] != 200:
        raise AssertionError("Offline bounds did not use 200 examples")
    if (
        len(artifact["selection"]["indices"]) != 200
        or len(set(artifact["selection"]["indices"])) != 200
        or len(artifact["samples"]) != 200
    ):
        raise AssertionError("Offline calibration sample list is invalid")
    expected = _expected_critical_sites(model_key)
    if (
        artifact["critical_site_count"] != expected
        or len(artifact["bounds"]) != expected
    ):
        raise AssertionError("Offline bounds site count is invalid")
    for site_key, value in artifact["bounds"].items():
        raw_min = float(value["raw_min"])
        raw_max = float(value["raw_max"])
        factor = float(value["scaling_factor"])
        if (
            not math.isfinite(raw_min)
            or not math.isfinite(raw_max)
            or raw_min > raw_max
            or factor != 2.0
        ):
            raise AssertionError(f"Invalid offline bounds for {site_key}")
    return artifact


def _completed_run_invariants(
    record: Mapping[str, Any],
    spec: FaultSpec,
    model_key: str,
    dataset_key: str,
    mode_id: str,
) -> None:
    steps = DATASETS[dataset_key].generation_steps
    engine = record["engine"]
    if not engine["complete"]:
        raise AssertionError("Completed run has incomplete engine telemetry")
    if int(engine["final_step"]) != steps - 1:
        raise AssertionError("Completed run ended at the wrong generation step")
    hook_calls = engine["hook_calls"]
    if len(hook_calls) != _expected_candidate_sites(model_key):
        raise AssertionError("Hook registry cardinality changed")
    if any(int(value) != steps for value in hook_calls.values()):
        raise AssertionError("A target Linear did not run once per step")
    if len(record["output"]["token_ids"]) != steps:
        raise AssertionError("Fault output has wrong fixed length")

    first_mode = "paper_clamp_first_token_bounds"
    if mode_id == first_mode:
        if int(engine["online_bounds_key_count"]) != _expected_critical_sites(
            model_key
        ):
            raise AssertionError("First-token bounds count is invalid")
    elif int(engine["online_bounds_key_count"]) != 0:
        raise AssertionError("Non-first-token mode persisted online bounds")

    if mode_id == "no_protection" and int(engine["correction_elements"]) != 0:
        raise AssertionError("No-protection mode performed correction")

    exact_text = (
        record["output"]["text"] == record["mode_clean"]["output"]["text"]
    )
    exact_tokens = (
        record["output"]["token_ids"]
        == record["mode_clean"]["output"]["token_ids"]
    )
    task_correct = bool(record["score"]["task_correct"])
    expected_outcome = (
        "MASKED_IDENTICAL"
        if exact_text
        else "MASKED_SEMANTIC"
        if task_correct
        else "SDC"
    )
    classification = record["classification"]
    if classification["outcome"] != expected_outcome:
        raise AssertionError("Stored outcome differs from recomputed outcome")
    expected_flags = {
        "exact_masked": exact_text,
        "semantic_masked": (not exact_text) and task_correct,
        "masked": task_correct,
        "sdc": not task_correct,
        "due": False,
        "exact_text_match_mode_clean": exact_text,
        "exact_token_match_mode_clean": exact_tokens,
    }
    for key, expected in expected_flags.items():
        if bool(classification[key]) != expected:
            raise AssertionError(f"Stored classification flag is wrong: {key}")
    if exact_text and not task_correct:
        raise AssertionError("Identical clean text cannot receive a different score")
    if not record["canonical_clean"]["score"]["task_correct"]:
        raise AssertionError("Canonical clean precondition is false")
    if not record["mode_clean"]["score"]["task_correct"]:
        raise AssertionError("Mode clean precondition is false")


def _due_run_invariants(record: Mapping[str, Any]) -> None:
    if not record["classification"]["due"]:
        raise AssertionError("DUE terminal record lacks DUE classification")
    if record["classification"]["outcome"] != "DUE":
        raise AssertionError("DUE record has wrong outcome")
    if record["engine"]["complete"]:
        raise AssertionError("DUE record unexpectedly has complete engine state")
    if record.get("due_reason") != "DUE_INVALID_FIRST_TOKEN_BOUNDS":
        raise AssertionError("DUE reason is not on the frozen allowlist")
    if record["protection"]["bounds_source"] != "first_token":
        raise AssertionError("Only first-token bounds may produce this DUE")
    if int(record["fault"]["target_step"]) != 0:
        raise AssertionError("Only a step-zero fault may produce this DUE")
    error = record.get("error", {})
    if error.get("type") != "ValueError" or not str(error.get("message", "")).startswith("Bounds values must be finite"):
        raise AssertionError("DUE exception does not match nonfinite bounds")
    if not isinstance(record.get("error"), dict):
        raise AssertionError("DUE record lacks exception evidence")


def _summary_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["model_key"],
            record["dataset_key"],
            record["fault"]["fault_type"],
            record["mode_id"],
        )
        grouped[key].append(record)
    rows = []
    for key in sorted(grouped):
        group = grouped[key]
        outcomes = Counter(
            record["classification"]["outcome"] for record in group
        )
        completed = outcomes["MASKED_IDENTICAL"] + outcomes[
            "MASKED_SEMANTIC"
        ] + outcomes["SDC"]
        due = outcomes["DUE"]
        sdc = outcomes["SDC"]
        lower, upper = _wilson(sdc, completed)
        corrected_target = sum(
            1
            for record in group
            if record["engine"]["injection_trace"]["correction_action"] != "none"
        )
        rows.append(
            {
                "model": key[0],
                "dataset": key[1],
                "fault_type": key[2],
                "mode": key[3],
                "total": len(group),
                "completed": completed,
                "masked_identical": outcomes["MASKED_IDENTICAL"],
                "masked_semantic": outcomes["MASKED_SEMANTIC"],
                "masked_total": outcomes["MASKED_IDENTICAL"]
                + outcomes["MASKED_SEMANTIC"],
                "sdc": sdc,
                "due": due,
                "sdc_rate_excluding_due": (
                    sdc / completed if completed else None
                ),
                "sdc_wilson95_low": lower,
                "sdc_wilson95_high": upper,
                "harmful_rate_sdc_plus_due": (
                    (sdc + due) / len(group) if group else None
                ),
                "target_corrected": corrected_target,
            }
        )
    if len(rows) != 45:
        raise AssertionError(f"Expected 45 summary cells, observed {len(rows)}")
    return rows


def _paired_transitions(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    by_spec: dict[str, dict[str, str]] = defaultdict(dict)
    for record in records:
        by_spec[record["spec_id"]][record["mode_id"]] = record[
            "classification"
        ]["outcome"]
    base_mode = "no_protection"
    result: dict[str, Counter[str]] = {
        "first_token_vs_no_protection": Counter(),
        "offline_vs_no_protection": Counter(),
    }
    for outcomes in by_spec.values():
        if set(outcomes) != {mode.mode_id for mode in MAIN_MODES}:
            raise AssertionError("A fault spec is missing one of three modes")
        base = outcomes[base_mode]
        first = outcomes["paper_clamp_first_token_bounds"]
        offline = outcomes["paper_clamp_offline_bounds"]
        result["first_token_vs_no_protection"][f"{base}->{first}"] += 1
        result["offline_vs_no_protection"][f"{base}->{offline}"] += 1
    return {
        key: dict(sorted(counter.items()))
        for key, counter in result.items()
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def _audit_campaign(
    output_root: str | Path,
    *,
    prompts_per_pair: int,
    specs_per_pair: int,
    artifact_stem: str,
    campaign_fingerprint: str | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    if prompts_per_pair <= 0 or specs_per_pair <= 0 or not artifact_stem:
        raise ValueError("Audit dimensions and artifact_stem must be positive")
    root = Path(output_root)
    campaign = load_artifact(root / "campaign.json")
    observed_fingerprint = campaign["campaign_fingerprint"]
    if (
        campaign_fingerprint is not None
        and observed_fingerprint != campaign_fingerprint
    ):
        raise AssertionError("Campaign fingerprint argument does not match lock")
    selection = load_artifact(root / "selection.json")
    if selection["campaign_fingerprint"] != observed_fingerprint:
        raise AssertionError("Selection campaign fingerprint mismatch")
    expected_pairs = {
        pair_id(model_key, dataset_key)
        for model_key, dataset_key in PAIR_ORDER
    }
    if set(selection["pairs"]) != expected_pairs:
        raise AssertionError("Selection does not contain five expected pairs")

    records: list[dict[str, Any]] = []
    spec_ids: set[str] = set()
    expected_run_paths: set[Path] = set()
    pair_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    fault_type_counts: Counter[str] = Counter()
    due_count = 0

    for model_key, dataset_key in PAIR_ORDER:
        identifier = pair_id(model_key, dataset_key)
        entry = selection["pairs"][identifier]
        if any(
            len(entry[field]) != prompts_per_pair
            for field in ("source_positions", "dataset_indices", "target_steps")
        ):
            raise AssertionError(
                f"{identifier} does not have {prompts_per_pair} prompts"
            )
        for source_position in entry["source_positions"]:
            screen = load_artifact(
                root
                / "screening"
                / model_key
                / dataset_key
                / f"{source_position:02d}.json"
            )
            if not screen["score"]["task_correct"]:
                raise AssertionError("Selected screening output is incorrect")
        _control_invariants(
            root,
            observed_fingerprint,
            model_key,
            dataset_key,
            entry,
        )
        _bounds_invariants(
            root,
            observed_fingerprint,
            model_key,
            dataset_key,
        )
        metadata, specs = load_manifest(
            root / "manifests" / f"{identifier}.json"
        )
        if metadata["campaign_fingerprint"] != observed_fingerprint:
            raise AssertionError("Manifest campaign fingerprint mismatch")
        if len(specs) != specs_per_pair:
            raise AssertionError(
                f"{identifier} manifest does not have {specs_per_pair} specs"
            )
        for spec in specs:
            if spec.spec_id in spec_ids:
                raise AssertionError("Duplicate FaultSpec across pair manifests")
            spec_ids.add(spec.spec_id)
            for protection in MAIN_MODES:
                mode_id = protection.mode_id
                path = _run_path(
                    root, model_key, dataset_key, spec.spec_id, mode_id
                )
                expected_run_paths.add(path.resolve())
                record = load_artifact(path)
                if record["campaign_fingerprint"] != observed_fingerprint:
                    raise AssertionError("Run campaign fingerprint mismatch")
                expected_run_id = stable_run_id(
                    campaign_fingerprint=observed_fingerprint,
                    spec_id=spec.spec_id,
                    mode_id=mode_id,
                )
                if (
                    record["run_id"] != expected_run_id
                    or record["spec_id"] != spec.spec_id
                    or record["mode_id"] != mode_id
                    or record["fault"] != spec.to_dict()
                ):
                    raise AssertionError("Run identity differs from manifest")
                expected_source_position = entry["source_positions"][
                    spec.sample_position
                ]
                if (
                    int(record["selection_rank"]) != spec.sample_position
                    or int(record["source_position"]) != expected_source_position
                ):
                    raise AssertionError("Run selection coordinates are wrong")
                _trace_invariants(record, spec)
                terminal = record["terminal_status"]
                if terminal == "completed":
                    _completed_run_invariants(
                        record, spec, model_key, dataset_key, mode_id
                    )
                elif terminal == "due":
                    _due_run_invariants(record)
                    due_count += 1
                else:
                    raise AssertionError(f"Invalid terminal status: {terminal}")
                records.append(record)
                pair_counts[identifier] += 1
                mode_counts[mode_id] += 1
                fault_type_counts[spec.fault_type.value] += 1

    actual_run_paths = {
        path.resolve()
        for path in (root / "runs").glob("*/*.json")
    }
    if actual_run_paths != expected_run_paths:
        missing = expected_run_paths.difference(actual_run_paths)
        extra = actual_run_paths.difference(expected_run_paths)
        raise AssertionError(
            f"Run file set mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    expected_specs = specs_per_pair * len(PAIR_ORDER)
    expected_runs = expected_specs * len(MAIN_MODES)
    expected_pair_runs = specs_per_pair * len(MAIN_MODES)
    expected_fault_type_runs = expected_specs
    if len(spec_ids) != expected_specs or len(records) != expected_runs:
        raise AssertionError(
            f"Expected {expected_specs} specs/{expected_runs} runs, got {len(spec_ids)}/{len(records)}"
        )
    if set(pair_counts.values()) != {expected_pair_runs} or len(pair_counts) != len(PAIR_ORDER):
        raise AssertionError(
            f"Each pair must have exactly {expected_pair_runs} runs"
        )
    if set(mode_counts.values()) != {expected_specs} or len(mode_counts) != len(MAIN_MODES):
        raise AssertionError(
            f"Each protection mode must have exactly {expected_specs} runs"
        )
    if set(fault_type_counts.values()) != {expected_fault_type_runs} or len(fault_type_counts) != 3:
        raise AssertionError(
            f"Each fault type must have exactly {expected_fault_type_runs} runs"
        )

    rows = _summary_rows(records)
    transitions = _paired_transitions(records)
    outcome_counts = Counter(
        record["classification"]["outcome"] for record in records
    )
    frozen_auditor_sha256 = next(
        item["sha256"]
        for item in campaign["immutable"]["source_files"]
        if item["path"] == "reproduction/src/ft2_formal/audit.py"
    )
    current_auditor_sha256 = sha256_file(Path(__file__))
    audit = {
        "schema_version": 1,
        "campaign_fingerprint": observed_fingerprint,
        "auditor": {
            "frozen_generation_sha256": frozen_auditor_sha256,
            "executed_sha256": current_auditor_sha256,
            "post_campaign_revision": (
                current_auditor_sha256 != frozen_auditor_sha256
            ),
            "revision_reason": "parse persisted FP16 words from hexadecimal telemetry",
        },
        "status": "passed",
        "checked_at": utc_now(),
        "counts": {
            "pairs": 5,
            "selected_prompts": prompts_per_pair * len(PAIR_ORDER),
            "fault_specs": len(spec_ids),
            "fault_runs": len(records),
            "summary_cells": len(rows),
            "due": due_count,
            "invalid": 0,
            "by_pair": dict(sorted(pair_counts.items())),
            "by_mode": dict(sorted(mode_counts.items())),
            "by_fault_type": dict(sorted(fault_type_counts.items())),
            "by_outcome": dict(sorted(outcome_counts.items())),
        },
        "paired_transitions": transitions,
        "acceptance_checks": {
            "campaign_identity_verified": True,
            "five_pairs_clean_correct_prompts": True,
            "offline_bounds_200_examples_and_full_sites": True,
            "unique_fault_specs": True,
            "three_modes_per_spec": True,
            "all_expected_terminal_runs": True,
            "one_verified_xor_per_run": True,
            "hamming_distances_verified": True,
            "fixed_generation_lengths_verified": True,
            "hook_call_counts_verified": True,
            "no_protection_has_zero_corrections": True,
            "no_invalid_runs": True,
        },
        "summary": rows,
    }
    if write_outputs:
        raw_lines = b"\n".join(
            canonical_json_bytes(record)
            for record in sorted(records, key=lambda item: item["run_id"])
        ) + b"\n"
        atomic_write_text(
            root / "raw" / f"{artifact_stem}_runs.jsonl",
            raw_lines.decode("ascii"),
        )
        _write_csv(
            root / "summaries" / f"{artifact_stem}_summary.csv", rows
        )
        audit = atomic_write_artifact(root / "audit.json", audit)
    return audit

def audit_pilot(
    output_root: str | Path,
    *,
    campaign_fingerprint: str | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    return _audit_campaign(
        output_root,
        prompts_per_pair=2,
        specs_per_pair=12,
        artifact_stem="pilot",
        campaign_fingerprint=campaign_fingerprint,
        write_outputs=write_outputs,
    )


def audit_reduced_formal(
    output_root: str | Path,
    *,
    campaign_fingerprint: str | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    return _audit_campaign(
        output_root,
        prompts_per_pair=10,
        specs_per_pair=90,
        artifact_stem="formal_reduced",
        campaign_fingerprint=campaign_fingerprint,
        write_outputs=write_outputs,
    )
