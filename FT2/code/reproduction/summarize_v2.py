#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

from ft2_repro.manifest import (
    ManifestEntry,
    ModelShape,
    load_manifest,
    manifest_sha256,
    validate_manifest,
)
from ft2_repro.metrics import (
    compare_decoded_text,
    compare_token_ids,
    exact_mcnemar,
    squad_semantic_score,
    token_sha256,
    wilson_interval,
)


MODES = ("unprotected", "repository_zero", "paper_clamp")
COMPLETED = {"MASKED", "SEMANTIC_SDC"}


class IncompleteCampaign(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_sha256(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_unique_jsonl(path: Path) -> list[Dict[str, Any]]:
    require(path.exists(), f"Missing required file: {path}")
    rows: list[Dict[str, Any]] = []
    run_ids: set[str] = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        require(isinstance(row, dict), f"{path}:{line_number} is not an object")
        run_id = str(row.get("run_id", ""))
        require(bool(run_id), f"{path}:{line_number} has no run_id")
        require(run_id not in run_ids, f"Duplicate run_id: {run_id}")
        run_ids.add(run_id)
        rows.append(row)
    return rows


def unique_index(
    rows: Iterable[Dict[str, Any]],
    key: Callable[[Dict[str, Any]], Any],
    label: str,
) -> Dict[Any, Dict[str, Any]]:
    result: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        item_key = key(row)
        require(item_key not in result, f"Duplicate {label} key: {item_key}")
        result[item_key] = row
    return result


def validate_metadata(metadata: Dict[str, Any]) -> None:
    require(metadata.get("schema_version") == 2, "Metadata schema must be 2")
    require(tuple(metadata.get("modes", ())) == MODES, "Unexpected mode list")
    payload = dict(metadata)
    supplied = payload.pop("campaign_id", None)
    payload.pop("created_at", None)
    expected = sha256_bytes(canonical(payload).encode("utf-8"))
    require(supplied == expected, "Metadata campaign self-hash mismatch")


def validate_result(row: Dict[str, Any], num_new_tokens: int) -> Tuple[int, ...]:
    tokens = row.get("generated_token_ids")
    require(isinstance(tokens, list), "generated_token_ids must be a list")
    require(
        len(tokens) == num_new_tokens
        and all(isinstance(value, int) for value in tokens),
        "Generated token sequence has the wrong type or length",
    )
    token_tuple = tuple(int(value) for value in tokens)
    require(
        row.get("generated_token_sha256") == token_sha256(token_tuple),
        "Generated token hash mismatch",
    )
    require(isinstance(row.get("decoded_text"), str), "decoded_text is missing")
    lengths = row.get("forward_input_lengths")
    require(
        isinstance(lengths, list)
        and len(lengths) == num_new_tokens
        and isinstance(lengths[0], int)
        and lengths[0] > 0
        and lengths[1:] == [1] * (num_new_tokens - 1),
        "Forward input lengths violate the fixed-step KV-cache design",
    )
    elapsed = row.get("elapsed_seconds")
    require(
        isinstance(elapsed, (int, float)) and elapsed >= 0,
        "Invalid elapsed_seconds",
    )
    return token_tuple


def validate_injection(
    row: Dict[str, Any],
    plan: Mapping[str, Any],
) -> None:
    require(row.get("injection_count") == 1, "Injection count must be exactly 1")
    require(row.get("bit_flip_verified") is True, "Top-level bit flip unverified")
    injection = row.get("injection")
    require(isinstance(injection, dict), "Missing nested injection telemetry")
    require(
        injection.get("bit_flip_verified") is True,
        "Nested bit flip unverified",
    )
    expected_fields = {
        "spec_id": plan["spec_id"],
        "observed_step": plan["token_step"],
        "layer_index": plan["layer_index"],
        "projection": plan["projection"],
        "flat_index": plan["flat_index"],
        "bit_index": plan["bit_index"],
    }
    for field, expected in expected_fields.items():
        require(
            injection.get(field) == expected,
            f"Injection telemetry mismatch: {field}",
        )
    before = int(injection["before_bits_hex"], 16)
    after = int(injection["after_bits_hex"], 16)
    expected_after = before ^ (1 << int(plan["bit_index"]))
    require(
        after == expected_after
        and int(injection["expected_after_bits_hex"], 16) == expected_after,
        "FP16 XOR evidence mismatch",
    )
    require(
        row.get("injected_value_before_fp16_hex")
        == injection["before_bits_hex"],
        "Top-level before bits mismatch",
    )
    require(
        row.get("injected_value_after_fp16_hex")
        == injection["after_bits_hex"],
        "Top-level after bits mismatch",
    )
    require(
        row.get("protected_value_fp16_hex")
        == injection["protected_bits_hex"],
        "Top-level protected bits mismatch",
    )


def proportion(count: int, denominator: int) -> Dict[str, Any]:
    if denominator == 0:
        return {
            "count": count,
            "denominator": 0,
            "rate": None,
            "wilson_95": [None, None],
        }
    lower, upper = wilson_interval(count, denominator)
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator,
        "wilson_95": [lower, upper],
    }


def exact_cluster_sign_flip(values: Sequence[float]) -> Dict[str, Any]:
    observed = sum(values) / len(values) if values else None
    nonzero = [value for value in values if abs(value) > 1e-15]
    if not nonzero:
        return {
            "clusters": len(values),
            "nonzero_clusters": 0,
            "observed_mean_difference": observed,
            "exact_two_sided_p": 1.0,
        }
    observed_sum = abs(sum(nonzero))
    extreme = 0
    total = 1 << len(nonzero)
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        statistic = abs(
            sum(sign * value for sign, value in zip(signs, nonzero))
        )
        if statistic + 1e-15 >= observed_sum:
            extreme += 1
    return {
        "clusters": len(values),
        "nonzero_clusters": len(nonzero),
        "observed_mean_difference": observed,
        "exact_two_sided_p": extreme / total,
    }


def mode_statistics(
    rows: Sequence[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    selected = [row for row in rows if row["mode"] == mode]
    counts = Counter(row["outcome"] for row in selected)
    masked = counts["MASKED"]
    semantic_sdc = counts["SEMANTIC_SDC"]
    due = counts["DUE"]
    invalid = counts["INVALID"]
    eligible = masked + semantic_sdc + due
    mission_failure = semantic_sdc + due
    require(len(selected) == eligible + invalid, f"{mode} outcome partition")
    return {
        "planned": len(selected),
        "eligible": eligible,
        "masked": masked,
        "semantic_sdc": semantic_sdc,
        "due": due,
        "invalid": invalid,
        "mission_failure": mission_failure,
        "semantic_sdc_rate": proportion(semantic_sdc, eligible),
        "due_rate": proportion(due, eligible),
        "mission_failure_rate": proportion(mission_failure, eligible),
        "invalid_rate_planned": proportion(invalid, len(selected)),
        "interval_note": (
            "Wilson interval is trial-level descriptive; prompt clustering "
            "is ignored."
        ),
    }


def paired_statistics(
    fault_index: Mapping[Tuple[str, str], Dict[str, Any]],
    spec_ids: Sequence[str],
    mode_a: str,
    mode_b: str,
) -> Dict[str, Any]:
    cells = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    excluded: list[Dict[str, Any]] = []
    differences: Dict[int, list[int]] = defaultdict(list)
    for spec_id in spec_ids:
        row_a = fault_index[(spec_id, mode_a)]
        row_b = fault_index[(spec_id, mode_b)]
        require(
            row_a["sample_index"] == row_b["sample_index"],
            "Paired rows have different samples",
        )
        if not (row_a["eligible"] and row_b["eligible"]):
            excluded.append(
                {
                    "spec_id": spec_id,
                    "sample_index": row_a["sample_index"],
                    "outcome_a": row_a["outcome"],
                    "outcome_b": row_b["outcome"],
                }
            )
            continue
        failure_a = bool(row_a["mission_failure"])
        failure_b = bool(row_b["mission_failure"])
        cells[f"n{int(failure_a)}{int(failure_b)}"] += 1
        differences[int(row_a["sample_index"])].append(
            int(failure_a) - int(failure_b)
        )
    included = sum(cells.values())
    risk_a = (
        (cells["n10"] + cells["n11"]) / included if included else None
    )
    risk_b = (
        (cells["n01"] + cells["n11"]) / included if included else None
    )
    absolute_reduction = (
        risk_a - risk_b if risk_a is not None and risk_b is not None else None
    )
    sample_means = {
        str(sample): sum(values) / len(values)
        for sample, values in sorted(differences.items())
    }
    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "endpoint": "mission_failure = semantic_sdc OR DUE",
        "cells": cells,
        "cell_labels": {
            "n10": "mode_a failure / mode_b pass",
            "n01": "mode_a pass / mode_b failure",
        },
        "included_pairs": included,
        "excluded_invalid_pairs": excluded,
        "risk_a": risk_a,
        "risk_b": risk_b,
        "absolute_risk_reduction_a_minus_b": absolute_reduction,
        "relative_risk_reduction_vs_a": (
            absolute_reduction / risk_a
            if absolute_reduction is not None and risk_a not in (None, 0)
            else None
        ),
        "mcnemar_exact_two_sided_p_descriptive": exact_mcnemar(
            cells["n10"],
            cells["n01"],
        ),
        "sample_mean_differences": sample_means,
        "cluster_exact_sign_flip_primary": exact_cluster_sign_flip(
            list(sample_means.values())
        ),
        "notes": [
            "McNemar is trial-level descriptive.",
            "The exact sign-flip gives each prompt equal weight.",
        ],
    }


def audit_campaign(directory: Path) -> Tuple[Dict[str, Any], int]:
    metadata_path = directory / "metadata.json"
    manifest_path = directory / "manifest.json"
    require(metadata_path.exists(), "Missing metadata.json")
    require(manifest_path.exists(), "Missing manifest.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(isinstance(metadata, dict), "metadata.json must contain an object")
    validate_metadata(metadata)

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(raw_manifest, list), "manifest.json must contain a list")
    entries = load_manifest(manifest_path)
    require(
        canonical(raw_manifest)
        == canonical([entry.to_dict() for entry in entries]),
        "Manifest contains noncanonical, missing, or extra fields",
    )
    require(
        manifest_sha256(entries) == metadata["manifest_sha256"],
        "Manifest hash mismatch",
    )
    shape_data = metadata["model_shape"]
    shape = ModelShape(
        num_layers=int(shape_data["num_layers"]),
        projection_widths={
            str(name): int(width)
            for name, width in shape_data["projection_widths"].items()
        },
    )
    samples = int(metadata["samples"])
    trials = int(metadata["trials_per_sample"])
    num_new_tokens = int(metadata["num_new_tokens"])
    validate_manifest(
        entries,
        sample_count=samples,
        trials_per_sample=trials,
        shape=shape,
        num_new_tokens=num_new_tokens,
    )
    plans = {entry.spec_id: entry.to_dict() for entry in entries}
    require(len(plans) == samples * trials, "Manifest spec_id collision")
    expected_grid = set(itertools.product(range(samples), range(trials)))
    observed_grid = {
        (entry.fault.sample_index, entry.fault.trial_index)
        for entry in entries
    }
    require(observed_grid == expected_grid, "Manifest sample/trial grid mismatch")
    for entry in entries:
        sample = entry.fault.sample_index
        require(
            entry.dataset_index == metadata["dataset_indices"][sample],
            "Manifest dataset index mismatch",
        )
        require(
            entry.fault.token_step == metadata["fault_steps"][sample],
            "Manifest token step mismatch",
        )
        require(
            entry.campaign_seed == metadata["seed"],
            "Manifest seed mismatch",
        )

    clean_rows = load_unique_jsonl(directory / "clean.jsonl")
    fault_rows = load_unique_jsonl(directory / "faults.jsonl")
    isolation_rows = load_unique_jsonl(directory / "isolation_checks.jsonl")
    clean_index = unique_index(
        clean_rows,
        lambda row: (int(row["sample_index"]), str(row["mode"])),
        "clean",
    )
    fault_index = unique_index(
        fault_rows,
        lambda row: (str(row["spec_id"]), str(row["mode"])),
        "fault",
    )
    isolation_index = unique_index(
        isolation_rows,
        lambda row: int(row["sample_index"]),
        "isolation",
    )
    expected_clean = set(itertools.product(range(samples), MODES))
    expected_faults = set(itertools.product(plans, MODES))
    require(set(clean_index) == expected_clean, "Clean record grid is incomplete")
    require(set(fault_index) == expected_faults, "Fault record grid is incomplete")
    require(
        set(isolation_index) == set(range(samples)),
        "Isolation record grid is incomplete",
    )

    campaign_id = metadata["campaign_id"]
    manifest_hash = metadata["manifest_sha256"]
    clean_tokens: Dict[Tuple[int, str], Tuple[int, ...]] = {}
    clean_texts: Dict[Tuple[int, str], str] = {}
    for sample in range(samples):
        common_row = clean_index[(sample, "unprotected")]
        common_prompt = common_row["prompt_sha256"]
        common_references = common_row["reference_answers"]
        common_dataset = metadata["dataset_indices"][sample]
        for mode in MODES:
            row = clean_index[(sample, mode)]
            require(row.get("schema_version") == 2, "Clean schema mismatch")
            require(row.get("record_type") == "clean", "Clean record_type mismatch")
            require(
                row.get("run_id") == f"clean:{sample}:{mode}",
                "Clean run_id mismatch",
            )
            require(
                row.get("campaign_id") == campaign_id
                and row.get("manifest_sha256") == manifest_hash,
                "Clean campaign or manifest mismatch",
            )
            require(
                row.get("dataset_index") == common_dataset,
                "Clean dataset mapping mismatch",
            )
            require(
                row.get("prompt_sha256") == common_prompt
                and row.get("reference_answers") == common_references,
                "Clean prompt or references differ across modes",
            )
            require(row.get("status") == "COMPLETED", "Clean run incomplete")
            require(row.get("injection_count") == 0, "Clean run injected a fault")
            tokens = validate_result(row, num_new_tokens)
            clean_tokens[(sample, mode)] = tokens
            clean_texts[(sample, mode)] = row["decoded_text"]

        common_tokens = clean_tokens[(sample, "unprotected")]
        common_text = clean_texts[(sample, "unprotected")]
        for mode in MODES:
            row = clean_index[(sample, mode)]
            token_comparison = compare_token_ids(
                clean_tokens[(sample, mode)],
                common_tokens,
            )
            text_comparison = compare_decoded_text(
                clean_texts[(sample, mode)],
                common_text,
            )
            semantic = squad_semantic_score(
                clean_texts[(sample, mode)],
                common_references,
            )
            require(
                row.get("common_golden_sha256") == token_sha256(common_tokens),
                "Clean common token hash mismatch",
            )
            require(
                row.get("common_golden_text_sha256") == text_sha256(common_text),
                "Clean common text hash mismatch",
            )
            require(
                row.get("clean_token_drift") is not token_comparison["equal"]
                and row.get("clean_drift") is not token_comparison["equal"],
                "Clean token drift mismatch",
            )
            require(
                row.get("clean_text_drift") is not text_comparison["equal"],
                "Clean text drift mismatch",
            )
            require(
                canonical(row.get("semantic_score")) == canonical(semantic)
                and row.get("clean_semantic_correct")
                is semantic["semantic_correct"],
                "Clean semantic score mismatch",
            )
        require(
            common_row.get("clean_semantic_correct") is True,
            f"Sample {sample} failed clean eligibility",
        )

    invalid_rows: list[Dict[str, Any]] = []
    for entry in entries:
        spec_id = entry.spec_id
        plan = plans[spec_id]
        sample = entry.fault.sample_index
        for mode in MODES:
            row = fault_index[(spec_id, mode)]
            clean_row = clean_index[(sample, mode)]
            common_row = clean_index[(sample, "unprotected")]
            require(row.get("schema_version") == 2, "Fault schema mismatch")
            require(row.get("record_type") == "fault", "Fault record_type mismatch")
            require(
                row.get("run_id") == f"{spec_id}:{mode}",
                "Fault run_id mismatch",
            )
            require(
                row.get("campaign_id") == campaign_id
                and row.get("manifest_sha256") == manifest_hash,
                "Fault campaign or manifest mismatch",
            )
            require(
                canonical(row.get("fault")) == canonical(plan),
                "Fault record differs from manifest",
            )
            require(
                row.get("sample_index") == sample
                and row.get("dataset_index") == entry.dataset_index
                and row.get("trial_index") == entry.fault.trial_index,
                "Fault sample/dataset/trial mapping mismatch",
            )
            require(
                row.get("prompt_sha256") == common_row["prompt_sha256"]
                and row.get("reference_answers")
                == common_row["reference_answers"],
                "Fault prompt or references mismatch",
            )
            require(
                row.get("common_golden_sha256")
                == common_row["generated_token_sha256"]
                and row.get("mode_clean_sha256")
                == clean_row["generated_token_sha256"],
                "Fault paired token hashes mismatch",
            )
            require(
                row.get("common_golden_text_sha256")
                == text_sha256(common_row["decoded_text"])
                and row.get("mode_clean_text_sha256")
                == text_sha256(clean_row["decoded_text"]),
                "Fault paired text hashes mismatch",
            )
            outcome = row.get("outcome")
            require(
                outcome in COMPLETED | {"DUE", "INVALID"},
                "Unknown fault outcome",
            )
            if row.get("bit_flip_verified") is True:
                validate_injection(row, plan)
            if row.get("eligible") is True:
                require(
                    row.get("fault_step_reached") is True,
                    "Eligible fault never reached its injection step",
                )
                validate_injection(row, plan)
                require(
                    row.get("bounds_sha256") == clean_row["bounds_sha256"],
                    "Eligible fault calibration bounds mismatch",
                )

            if outcome in COMPLETED:
                tokens = validate_result(row, num_new_tokens)
                common_token_comparison = compare_token_ids(
                    tokens,
                    clean_tokens[(sample, "unprotected")],
                )
                mode_token_comparison = compare_token_ids(
                    tokens,
                    clean_tokens[(sample, mode)],
                )
                common_text_comparison = compare_decoded_text(
                    row["decoded_text"],
                    clean_texts[(sample, "unprotected")],
                )
                mode_text_comparison = compare_decoded_text(
                    row["decoded_text"],
                    clean_texts[(sample, mode)],
                )
                semantic = squad_semantic_score(
                    row["decoded_text"],
                    common_row["reference_answers"],
                )
                sdc_common = (
                    not common_text_comparison["equal"]
                    and not semantic["semantic_correct"]
                )
                sdc_mode = (
                    not mode_text_comparison["equal"]
                    and not semantic["semantic_correct"]
                )
                require(
                    row.get("eligible") is True
                    and row.get("due") is False
                    and row.get("invalid") is False,
                    "Completed fault eligibility flags mismatch",
                )
                require(row.get("error_phase") is None, "Completed fault has error")
                require(
                    row.get("semantic_sdc_common") is sdc_common
                    and row.get("semantic_sdc_mode_clean") is sdc_mode
                    and row.get("mission_failure") is sdc_common
                    and row.get("paper_masked") is (not sdc_common),
                    "Completed fault semantic outcome mismatch",
                )
                require(
                    row.get("outcome")
                    == ("SEMANTIC_SDC" if sdc_common else "MASKED"),
                    "Completed fault outcome label mismatch",
                )
                require(
                    row.get("status")
                    == (
                        "COMPLETED_SEMANTIC_SDC"
                        if sdc_common
                        else "COMPLETED_MASKED"
                    ),
                    "Completed fault status mismatch",
                )
                require(
                    canonical(row.get("semantic_score")) == canonical(semantic)
                    and row.get("semantic_correct")
                    is semantic["semantic_correct"],
                    "Completed fault semantic score mismatch",
                )
                require(
                    row.get("token_divergence_common")
                    is (not common_token_comparison["equal"])
                    and row.get("token_divergence_mode_clean")
                    is (not mode_token_comparison["equal"])
                    and row.get("text_divergence_common")
                    is (not common_text_comparison["equal"])
                    and row.get("text_divergence_mode_clean")
                    is (not mode_text_comparison["equal"]),
                    "Completed fault divergence mismatch",
                )
                require(
                    row.get("first_different_token_common")
                    == common_token_comparison["first_different_token"]
                    and row.get("different_token_count_common")
                    == common_token_comparison["different_token_count"]
                    and row.get("first_different_token_mode_clean")
                    == mode_token_comparison["first_different_token"]
                    and row.get("different_token_count_mode_clean")
                    == mode_token_comparison["different_token_count"],
                    "Completed fault token comparison detail mismatch",
                )
            elif outcome == "DUE":
                require(
                    row.get("status") == "DUE_EXCEPTION"
                    and row.get("error_phase") == "generation"
                    and row.get("eligible") is True
                    and row.get("due") is True
                    and row.get("invalid") is False
                    and row.get("mission_failure") is True,
                    "DUE flags mismatch",
                )
                require(
                    row.get("semantic_sdc_common") is None
                    and row.get("semantic_sdc_mode_clean") is None,
                    "DUE must not claim semantic SDC classification",
                )
                require(
                    bool(row.get("error_type")) and bool(row.get("error_message")),
                    "DUE lacks error evidence",
                )
            else:
                require(
                    row.get("status") == "INVALID_EXCEPTION"
                    and row.get("error_phase") in {"generation", "verification"}
                    and row.get("eligible") is False
                    and row.get("due") is False
                    and row.get("invalid") is True
                    and row.get("mission_failure") is None,
                    "INVALID flags mismatch",
                )
                require(
                    bool(row.get("error_type")) and bool(row.get("error_message")),
                    "INVALID lacks error evidence",
                )
                invalid_rows.append(
                    {
                        "run_id": row["run_id"],
                        "spec_id": spec_id,
                        "sample_index": sample,
                        "mode": mode,
                        "error_phase": row["error_phase"],
                        "error_type": row["error_type"],
                    }
                )

    for sample in range(samples):
        row = isolation_index[sample]
        common = clean_index[(sample, "unprotected")]
        require(row.get("schema_version") == 2, "Isolation schema mismatch")
        require(
            row.get("record_type") == "isolation_check"
            and row.get("run_id") == f"isolation:{sample}",
            "Isolation identity mismatch",
        )
        require(
            row.get("campaign_id") == campaign_id
            and row.get("manifest_sha256") == manifest_hash,
            "Isolation campaign or manifest mismatch",
        )
        require(
            row.get("dataset_index") == metadata["dataset_indices"][sample]
            and row.get("prompt_sha256") == common["prompt_sha256"],
            "Isolation sample mapping mismatch",
        )
        require(row.get("passed") is True, "Isolation check failed")
        require(
            row.get("golden_token_sha256")
            == common["generated_token_sha256"]
            and row.get("repeated_token_sha256")
            == common["generated_token_sha256"],
            "Isolation token hash mismatch",
        )
        require(
            row.get("bounds_sha256") == common["bounds_sha256"],
            "Isolation calibration bounds mismatch",
        )

    mode_results = {
        mode: mode_statistics(fault_rows, mode) for mode in MODES
    }
    spec_ids = [entry.spec_id for entry in entries]
    pair_results = {
        "unprotected_vs_repository_zero": paired_statistics(
            fault_index,
            spec_ids,
            "unprotected",
            "repository_zero",
        ),
        "unprotected_vs_paper_clamp": paired_statistics(
            fault_index,
            spec_ids,
            "unprotected",
            "paper_clamp",
        ),
        "repository_zero_vs_paper_clamp": paired_statistics(
            fault_index,
            spec_ids,
            "repository_zero",
            "paper_clamp",
        ),
    }
    if not invalid_rows:
        for result in pair_results.values():
            require(
                result["included_pairs"] == len(entries),
                "A formal paired comparison excluded trials",
            )
    statistics = {
        "by_mode": mode_results,
        "paired": pair_results,
    }
    summary: Dict[str, Any] = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "manifest_sha256": manifest_hash,
        "campaign_kind": metadata["campaign_kind"],
        "samples": samples,
        "trials_per_sample": trials,
        "planned_fault_specs": len(entries),
        "planned_mode_runs": len(entries) * len(MODES),
        "formal_complete": not invalid_rows,
        "invalid_terminal_rows": invalid_rows,
        "completeness_warning": (
            None
            if not invalid_rows
            else (
                f"{len(invalid_rows)} INVALID terminal rows; statistics are "
                "provisional and exclude invalid pairs."
            )
        ),
        "primary_endpoint": "mission_failure = semantic_sdc OR DUE",
        "primary_inference": "sample-cluster exact sign-flip",
        "guardrail": (
            "Reduced prompt-level pilot; Wilson and McNemar are descriptive "
            "because they ignore prompt clustering."
        ),
        "audit": {
            "metadata_self_hash": "verified",
            "manifest_hash_and_grid": "verified",
            "clean_grid_and_semantics": "verified",
            "fault_grid_pairing_and_xor": "verified",
            "isolation_grid": "verified",
        },
    }
    if invalid_rows:
        summary["provisional_statistics_excluding_invalid"] = statistics
    else:
        summary["statistics"] = statistics
    return summary, len(invalid_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict validator and statistical summary for FT2 campaign"
    )
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.campaign_dir / "summary_v2.json"
    summary, invalid_count = audit_campaign(args.campaign_dir)
    atomic_write_json(output, summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "formal_complete": summary["formal_complete"],
                "invalid_terminal_rows": invalid_count,
                "by_mode": (
                    summary.get("statistics")
                    or summary["provisional_statistics_excluding_invalid"]
                )["by_mode"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if invalid_count:
        raise IncompleteCampaign(
            f"{invalid_count} INVALID terminal rows; see {output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

