#!/usr/bin/env python3
"""Re-score a completed FT2 campaign using the upstream author's logic.

The source run artifacts are read-only. Results are written to a separate
analysis directory so the original campaign remains auditable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import string


AUTHOR_REPO = "https://github.com/pipijing13/FT2-LLM-inference-protection"
AUTHOR_COMMIT = "90510aec5d26850739cf583c97c65cbd71256cb1"
ANALYSIS_ID = "author_repo_scoring_v1"
EXPECTED_RUNS = 18_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def normalize_qa(text: str) -> str:
    """Match normalize_answer in upstream QA/XTREME evaluation scripts."""
    lowered = text.lower()
    no_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def normalize_gsm8k(text: str) -> str:
    """Match normalize_answer in upstream GSM8K evaluation scripts.

    The upstream GSM8K variant defines punctuation removal but does not call it.
    It removes two LaTeX forms instead, so this intentionally preserves ordinary
    punctuation for literal repository compatibility.
    """
    normalized = text.lower()
    normalized = re.sub(r"\\[|\\]", "", normalized)
    normalized = re.sub(r"\\text\{([^}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split())


def author_tokens(text: str, dataset_key: str) -> list[str]:
    normalized = normalize_gsm8k(text) if dataset_key == "gsm8k" else normalize_qa(text)
    return normalized.split() if normalized else []


def author_precision_recall_f1(
    prediction: str, ground_truth: str, dataset_key: str
) -> tuple[float, float, float]:
    """Match compute_precision_recall_f1 in the upstream evaluation scripts."""
    prediction_tokens = author_tokens(prediction, dataset_key)
    ground_truth_tokens = author_tokens(ground_truth, dataset_key)
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common_tokens.values())
    if not prediction_tokens or not ground_truth_tokens:
        exact = float(prediction_tokens == ground_truth_tokens)
        return exact, exact, exact
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def nested_counter() -> defaultdict:
    return defaultdict(lambda: {
        "planned": 0,
        "recall_sum": 0.0,
        "outcomes": Counter(),
        "old_outcomes": Counter(),
    })


def finalize_bucket(bucket: dict) -> dict:
    planned = bucket["planned"]
    outcomes = dict(sorted(bucket["outcomes"].items()))
    sdc = outcomes.get("SDC", 0)
    mean_recall = bucket["recall_sum"] / planned if planned else 0.0
    return {
        "planned": planned,
        "evaluable": planned,
        "outcome_counts": outcomes,
        "sdc": sdc,
        "sdc_rate": sdc / planned if planned else 0.0,
        "mean_reference_recall_literal_repo": mean_recall,
        "sdc_proxy_one_minus_mean_recall_literal_repo": 1.0 - mean_recall,
        "original_outcome_counts": dict(sorted(bucket["old_outcomes"].items())),
    }


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    summary_path = results_dir / "summary.json"
    controls_dir = results_dir / "controls"
    runs_dir = results_dir / "runs"
    if output_dir == results_dir or results_dir not in output_dir.parents:
        raise ValueError("output-dir must be a dedicated child of results-dir")

    controls: dict[tuple[str, int], dict] = {}
    all_control_rows: list[dict] = []
    for pair_dir in sorted(controls_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        for path in sorted(pair_dir.glob("*.json")):
            record = load_json(path)
            key = (record["pair_id"], int(record["source_position"]))
            if record["mode_id"] == "no_protection":
                if key in controls:
                    raise AssertionError(f"Duplicate canonical control: {key}")
                controls[key] = record
            all_control_rows.append(record)

    cell_buckets = nested_counter()
    mode_buckets = nested_counter()
    pair_mode_buckets = nested_counter()
    overall_bucket = {
        "planned": 0,
        "recall_sum": 0.0,
        "outcomes": Counter(),
        "old_outcomes": Counter(),
    }
    reclassified_non_evaluable = Counter()
    total_runs = 0

    for pair_dir in sorted(runs_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        for path in sorted(pair_dir.glob("*.json")):
            record = load_json(path)
            if record.get("terminal_status") != "completed":
                raise AssertionError(f"Non-completed source run: {path}")
            if record.get("engine", {}).get("injection_count") != 1:
                raise AssertionError(f"Unexpected injection count: {path}")

            pair_id = record["pair_id"]
            mode_id = record["mode_id"]
            fault_type = record["fault"]["fault_type"]
            dataset_key = record["dataset_key"]
            source_position = int(record["source_position"])
            control = controls[(pair_id, source_position)]
            references = control["references"]
            if not references:
                raise AssertionError(f"Missing reference: {pair_id}/{source_position}")
            ground_truth = references[0]
            faulty_text = record["output"]["text"]
            # The upstream scripts generate `base_output` with the same protected
            # model immediately before injecting faults, then compare the faulty
            # text with that mode-local clean output (for example, lines 503-533
            # and 719-720 in qwengsmprotect.py).
            mode_clean_text = record["mode_clean"]["output"]["text"]
            _, recall, _ = author_precision_recall_f1(
                faulty_text, ground_truth, dataset_key
            )
            author_correct = recall == 1.0
            exact_text = faulty_text == mode_clean_text
            if exact_text:
                outcome = "MASKED_IDENTICAL"
            elif author_correct:
                outcome = "MASKED_SEMANTIC"
            else:
                outcome = "SDC"

            old_outcome = record["classification"]["outcome"]
            if old_outcome == "NON_EVALUABLE_CLEAN_FAILURE":
                reclassified_non_evaluable[outcome] += 1

            keys = [
                (cell_buckets, (pair_id, mode_id, fault_type)),
                (mode_buckets, mode_id),
                (pair_mode_buckets, (pair_id, mode_id)),
            ]
            for buckets, key in keys:
                bucket = buckets[key]
                bucket["planned"] += 1
                bucket["recall_sum"] += recall
                bucket["outcomes"][outcome] += 1
                bucket["old_outcomes"][old_outcome] += 1
            overall_bucket["planned"] += 1
            overall_bucket["recall_sum"] += recall
            overall_bucket["outcomes"][outcome] += 1
            overall_bucket["old_outcomes"][old_outcome] += 1
            total_runs += 1

    if total_runs != EXPECTED_RUNS:
        raise AssertionError(f"Expected {EXPECTED_RUNS} runs, found {total_runs}")

    clean_controls: dict[str, dict[str, dict]] = defaultdict(dict)
    canonical_failures: list[dict] = []
    for record in all_control_rows:
        pair_id = record["pair_id"]
        mode_id = record["mode_id"]
        dataset_key = record["dataset_key"]
        references = record["references"]
        _, recall, _ = author_precision_recall_f1(
            record["output"]["text"], references[0], dataset_key
        )
        slot = clean_controls[pair_id].setdefault(
            mode_id,
            {"total": 0, "author_correct": 0, "author_incorrect": 0, "recall_sum": 0.0},
        )
        slot["total"] += 1
        slot["recall_sum"] += recall
        if recall == 1.0:
            slot["author_correct"] += 1
        else:
            slot["author_incorrect"] += 1
            if mode_id == "no_protection":
                canonical_failures.append({
                    "pair_id": pair_id,
                    "source_position": record["source_position"],
                    "reference": references[0],
                    "recall": recall,
                })
    if canonical_failures:
        raise AssertionError(f"Canonical controls fail author scorer: {canonical_failures}")
    for modes in clean_controls.values():
        for stats in modes.values():
            stats["mean_reference_recall_literal_repo"] = stats.pop("recall_sum") / stats["total"]

    result_tables: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for (pair_id, mode_id, fault_type), bucket in sorted(cell_buckets.items()):
        result_tables[pair_id][mode_id][fault_type] = finalize_bucket(bucket)

    pair_mode_totals: dict[str, dict[str, dict]] = defaultdict(dict)
    for (pair_id, mode_id), bucket in sorted(pair_mode_buckets.items()):
        pair_mode_totals[pair_id][mode_id] = finalize_bucket(bucket)

    mode_totals = {
        mode_id: finalize_bucket(bucket)
        for mode_id, bucket in sorted(mode_buckets.items())
    }

    payload = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "results_dir": str(results_dir),
            "summary_sha256": sha256_file(summary_path),
            "campaign_fingerprint": load_json(summary_path)["campaign_fingerprint"],
            "source_runs": total_runs,
            "source_controls": len(all_control_rows),
            "upstream_author_repo": AUTHOR_REPO,
            "upstream_commit": AUTHOR_COMMIT,
            "analysis_script_sha256": sha256_file(Path(__file__)),
        },
        "scoring": {
            "primary": "author_intended_binary_reference_recall",
            "reference_selection": "first_reference_only",
            "correct_rule": "normalized reference token recall equals 1.0",
            "exact_mask_rule": "faulty decoded text equals the clean decoded text from the same protection mode",
            "clean_control_gate": False,
            "eos_truncation": False,
            "gsm8k_normalization": "literal upstream GSM8K normalize_answer (LaTeX cleanup; ordinary punctuation retained)",
            "qa_normalization": "literal upstream QA normalize_answer (lowercase, ASCII punctuation/articles removed, whitespace normalized)",
            "literal_repo_secondary_metric": "mean reference recall; upstream uses `re2 == 0` rather than assignment when recall < 1",
        },
        "counts": {
            "fault_runs": total_runs,
            "controls": len(all_control_rows),
            "reclassified_original_non_evaluable": sum(reclassified_non_evaluable.values()),
        },
        "outcome_counts": finalize_bucket(overall_bucket)["outcome_counts"],
        "overall": finalize_bucket(overall_bucket),
        "mode_totals": mode_totals,
        "pair_mode_totals": pair_mode_totals,
        "result_tables": result_tables,
        "clean_controls_author_scorer": clean_controls,
        "reclassified_original_non_evaluable": dict(sorted(reclassified_non_evaluable.items())),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_output = output_dir / "summary.json"
    write_json(summary_output, payload)

    csv_lines = [
        "pair_id,mode_id,fault_type,planned,masked_identical,masked_semantic,sdc,sdc_rate,mean_reference_recall_literal_repo,sdc_proxy_literal_repo,original_non_evaluable"
    ]
    for pair_id, modes in sorted(result_tables.items()):
        for mode_id, fault_types in sorted(modes.items()):
            for fault_type, stats in sorted(fault_types.items()):
                outcomes = stats["outcome_counts"]
                old = stats["original_outcome_counts"]
                csv_lines.append(
                    ",".join([
                        pair_id,
                        mode_id,
                        fault_type,
                        str(stats["planned"]),
                        str(outcomes.get("MASKED_IDENTICAL", 0)),
                        str(outcomes.get("MASKED_SEMANTIC", 0)),
                        str(stats["sdc"]),
                        f'{stats["sdc_rate"]:.10f}',
                        f'{stats["mean_reference_recall_literal_repo"]:.10f}',
                        f'{stats["sdc_proxy_one_minus_mean_recall_literal_repo"]:.10f}',
                        str(old.get("NON_EVALUABLE_CLEAN_FAILURE", 0)),
                    ])
                )
    (output_dir / "per_cell.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# FT2 author-logic re-scoring",
        "",
        f"Source runs: {total_runs}; source controls: {len(all_control_rows)}.",
        "All completed fault-injection runs are included; protected-clean correctness is not an evaluability gate.",
        "Primary SDC is binary: normalized reference-token recall must equal 1.0.",
        "",
        "## Overall by mode",
        "",
        "| Mode | Runs | Masked identical | Masked semantic | SDC | SDC rate | Literal repo 1-mean-recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode_id, stats in mode_totals.items():
        outcomes = stats["outcome_counts"]
        report_lines.append(
            f'| {mode_id} | {stats["planned"]} | {outcomes.get("MASKED_IDENTICAL", 0)} | '
            f'{outcomes.get("MASKED_SEMANTIC", 0)} | {stats["sdc"]} | '
            f'{stats["sdc_rate"]:.4%} | '
            f'{stats["sdc_proxy_one_minus_mean_recall_literal_repo"]:.4%} |'
        )
    report_lines.extend([
        "",
        "## Previously excluded runs",
        "",
        f'Original NON_EVALUABLE_CLEAN_FAILURE runs: {sum(reclassified_non_evaluable.values())}.',
    ])
    for outcome, count in sorted(reclassified_non_evaluable.items()):
        report_lines.append(f"- {outcome}: {count}")
    report_lines.extend([
        "",
        "## Qwen2-Math-7B / GSM8K by mode",
        "",
        "| Mode | Runs | SDC | SDC rate | Author-scored clean correct |",
        "|---|---:|---:|---:|---:|",
    ])
    target_pair = "qwen2_math_7b__gsm8k"
    for mode_id, stats in pair_mode_totals[target_pair].items():
        clean = clean_controls[target_pair][mode_id]
        report_lines.append(
            f'| {mode_id} | {stats["planned"]} | {stats["sdc"]} | '
            f'{stats["sdc_rate"]:.4%} | {clean["author_correct"]}/{clean["total"]} |'
        )
    report_lines.extend([
        "",
        "## Reproducibility notes",
        "",
        f"- Upstream commit: `{AUTHOR_COMMIT}`",
        f"- Source summary SHA-256: `{payload['source']['summary_sha256']}`",
        f"- Analysis script SHA-256: `{payload['source']['analysis_script_sha256']}`",
        "- Stored outputs are scored in full, without EOS truncation, matching the upstream fixed-step loops.",
        "- The author's literal mean-recall result is retained as a secondary metric; the primary binary result implements the paper's Masked/SDC intent.",
    ])
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    checksums = []
    for name in ("summary.json", "per_cell.csv", "report.md"):
        checksums.append(f"{sha256_file(output_dir / name)}  {name}")
    (output_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "fault_runs": total_runs,
        "outcome_counts": payload["outcome_counts"],
        "mode_totals": {
            mode: {"sdc": stats["sdc"], "sdc_rate": stats["sdc_rate"]}
            for mode, stats in mode_totals.items()
        },
        "reclassified_original_non_evaluable": payload["reclassified_original_non_evaluable"],
        "qwen2_math_7b__gsm8k": {
            mode: {
                "sdc": stats["sdc"],
                "sdc_rate": stats["sdc_rate"],
                "clean": clean_controls[target_pair][mode],
            }
            for mode, stats in pair_mode_totals[target_pair].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
