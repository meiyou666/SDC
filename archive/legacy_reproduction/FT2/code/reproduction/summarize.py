#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from ft2_repro.metrics import (
    exact_mcnemar,
    paired_binary_counts,
    wilson_interval,
)


MODES = ("unprotected", "repository_zero", "paper_clamp")


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    records = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            records.append(json.loads(raw))
    return records


def exact_cluster_sign_flip(differences: Sequence[int]) -> float:
    nonzero = [int(value) for value in differences if value != 0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    total = 0
    extreme = 0
    magnitudes = [abs(value) for value in nonzero]
    for signs in itertools.product((-1, 1), repeat=len(magnitudes)):
        total += 1
        statistic = abs(
            sum(sign * value for sign, value in zip(signs, magnitudes))
        )
        if statistic >= observed:
            extreme += 1
    return extreme / total


def mode_summary(
    records: Iterable[Mapping[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    selected = [record for record in records if record["mode"] == mode]
    valid = [
        record
        for record in selected
        if record.get("injection_count") == 1
        and str(record.get("status", "")).startswith("COMPLETED_")
    ]
    sdc = sum(record["sdc_common"] is True for record in valid)
    correct = sum(record["sdc_common"] is False for record in valid)
    due = sum(str(record.get("status", "")).startswith("DUE_") for record in selected)
    invalid = len(selected) - len(valid) - due
    interval = wilson_interval(sdc, len(valid)) if valid else (None, None)

    per_sample = defaultdict(lambda: {"sdc": 0, "valid": 0})
    for record in valid:
        bucket = per_sample[int(record["sample_index"])]
        bucket["valid"] += 1
        bucket["sdc"] += int(record["sdc_common"])

    return {
        "total_records": len(selected),
        "valid_injected_trials": len(valid),
        "completed_correct": correct,
        "sdc": sdc,
        "due": due,
        "invalid": invalid,
        "sdc_rate_all_valid": sdc / len(valid) if valid else None,
        "sdc_rate_wilson_95": list(interval),
        "per_sample": {
            str(key): value for key, value in sorted(per_sample.items())
        },
    }


def paired_summary(
    records: Sequence[Mapping[str, Any]],
    mode_a: str,
    mode_b: str,
) -> Dict[str, Any]:
    counts = paired_binary_counts(
        records,
        mode_a,
        mode_b,
        "sdc_common",
    )
    included = (
        counts["n00"]
        + counts["n01"]
        + counts["n10"]
        + counts["n11"]
    )
    failures_a = counts["n10"] + counts["n11"]
    failures_b = counts["n01"] + counts["n11"]
    risk_a = failures_a / included if included else None
    risk_b = failures_b / included if included else None
    absolute_reduction = (
        risk_a - risk_b if included else None
    )
    relative_reduction = (
        absolute_reduction / risk_a
        if included and risk_a and absolute_reduction is not None
        else None
    )

    by_sample: Dict[int, Dict[str, Dict[str, bool]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        if record["mode"] not in (mode_a, mode_b):
            continue
        if not isinstance(record.get("sdc_common"), bool):
            continue
        by_sample[int(record["sample_index"])][
            str(record["spec_id"])
        ][str(record["mode"])] = bool(record["sdc_common"])

    sample_differences = []
    for sample_pairs in by_sample.values():
        failure_a = 0
        failure_b = 0
        for pair in sample_pairs.values():
            if mode_a in pair and mode_b in pair:
                failure_a += int(pair[mode_a])
                failure_b += int(pair[mode_b])
        sample_differences.append(failure_a - failure_b)

    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "paired_counts": counts,
        "included_pairs": included,
        "risk_a": risk_a,
        "risk_b": risk_b,
        "absolute_risk_reduction": absolute_reduction,
        "relative_risk_reduction": relative_reduction,
        "exact_mcnemar_p": exact_mcnemar(
            counts["n10"],
            counts["n01"],
        ),
        "per_sample_failure_count_differences": sample_differences,
        "cluster_exact_sign_flip_p": exact_cluster_sign_flip(
            sample_differences
        ),
    }


def summarize_campaign(output_dir: Path) -> Dict[str, Any]:
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    clean = read_jsonl(output_dir / "clean.jsonl")
    faults = read_jsonl(output_dir / "faults.jsonl")
    checks = read_jsonl(output_dir / "isolation_checks.jsonl")

    spec_ids = {entry["spec_id"] for entry in manifest}
    if len(spec_ids) != len(manifest):
        raise ValueError("Manifest spec_id values are not unique")
    expected_specs = (
        int(metadata["samples"]) * int(metadata["trials_per_sample"])
    )
    if len(manifest) != expected_specs:
        raise ValueError("Manifest record count mismatch")
    if len(clean) != int(metadata["samples"]) * len(MODES):
        raise ValueError("Clean record count mismatch")
    if len(faults) != expected_specs * len(MODES):
        raise ValueError("Fault record count mismatch")

    fault_counter = Counter(
        (str(record["spec_id"]), str(record["mode"]))
        for record in faults
    )
    expected_keys = {
        (spec_id, mode)
        for spec_id in spec_ids
        for mode in MODES
    }
    if set(fault_counter) != expected_keys:
        raise ValueError("Fault pairing is incomplete")
    if set(fault_counter.values()) != {1}:
        raise ValueError("Duplicate paired fault record")

    clean_drift = {
        mode: sum(
            record["mode"] == mode and record["clean_drift"] is True
            for record in clean
        )
        for mode in MODES
    }
    isolation_passed = sum(check["passed"] is True for check in checks)
    if isolation_passed != int(metadata["samples"]):
        raise ValueError("One or more isolation checks failed")

    summary = {
        "schema_version": 1,
        "campaign_id": metadata["campaign_id"],
        "manifest_sha256": metadata["manifest_sha256"],
        "scope": {
            "samples": metadata["samples"],
            "trials_per_sample": metadata["trials_per_sample"],
            "paired_fault_specs": expected_specs,
            "generated_tokens": metadata["num_new_tokens"],
        },
        "clean_drift_samples": clean_drift,
        "isolation_checks_passed": isolation_passed,
        "modes": {
            mode: mode_summary(faults, mode) for mode in MODES
        },
        "paired_comparisons": {
            "unprotected_vs_repository_zero": paired_summary(
                faults,
                "unprotected",
                "repository_zero",
            ),
            "unprotected_vs_paper_clamp": paired_summary(
                faults,
                "unprotected",
                "paper_clamp",
            ),
            "repository_zero_vs_paper_clamp": paired_summary(
                faults,
                "repository_zero",
                "paper_clamp",
            ),
        },
        "interpretation_guardrail": (
            "Trial-level reduced mechanism reproduction; 10 prompts are "
            "clusters, so results do not establish the paper's full-scale "
            "population effect."
        ),
    }
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir / "summary.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize_campaign(args.output_dir),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
