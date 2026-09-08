#!/usr/bin/env python3
"""Summarize VII-A detection + recompute runs (paper Section VII-A, alpha=0.05).

Reads each run's metrics.jsonl and summary.json and reports final eval loss,
eval delta vs the fault-free baseline, detection statistics, and recompute
statistics. Detection rate = tp/(tp+fn) over injected steps; recompute
precision = tp/(tp+fp) over detected steps.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def load_jsonl(path):
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def load_summary(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_values(records, key, finite_only=True):
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if not finite_only or math.isfinite(value):
                values.append(value)
    return values


def last_value(records, key):
    values = numeric_values(records, key, finite_only=False)
    return values[-1] if values else ""


def max_finite(records, key):
    values = numeric_values(records, key, finite_only=True)
    return max(values) if values else ""


def nonfinite_metric_count(records):
    count = 0
    for record in records:
        for value in record.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    count += 1
    return count


def confusion_rates(summary):
    confusion = summary.get("confusion_count", {})
    tp = confusion.get("tp", 0)
    fp = confusion.get("fp", 0)
    fn = confusion.get("fn", 0)
    detection_rate = tp / (tp + fn) if (tp + fn) else ""
    precision = tp / (tp + fp) if (tp + fp) else ""
    return tp, fp, fn, detection_rate, precision


def parse_run(run_dir, baseline_eval_loss, expected_step):
    records = load_jsonl(run_dir / "metrics.jsonl")
    summary = load_summary(run_dir / "summary.json")
    core = [record for record in records if "loss" in record]

    trigger_steps = [
        int(record.get("_step", -1))
        for record in core
        if float(record.get("trigger_nvbit", 0.0)) == 1.0
    ]
    detected_steps = [
        int(record.get("_step", -1))
        for record in core
        if float(record.get("detected_anomaly", 0.0)) == 1.0
    ]

    eval_loss = last_value(records, "eval_loss")
    eval_delta = ""
    if isinstance(eval_loss, float) and isinstance(baseline_eval_loss, float) and math.isfinite(eval_loss) and math.isfinite(baseline_eval_loss):
        eval_delta = eval_loss - baseline_eval_loss

    tp, fp, fn, detection_rate, precision = confusion_rates(summary)
    confusion = summary.get("confusion_count", {})
    total_time = summary.get("total_time", "")

    return {
        "final_eval_loss": eval_loss,
        "eval_delta_vs_baseline": eval_delta,
        "trigger_count": len(trigger_steps),
        "detected_count": len(detected_steps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "detection_rate": detection_rate,
        "recompute_precision": precision,
        "correct_recompute": confusion.get("count_correct_recompute", ""),
        "incorrect_recompute": confusion.get("count_incorrect_recompute", ""),
        "max_gradient_norm_pre": max_finite(core, "gradient_norm_pre"),
        "max_rt": max_finite(records, "rt/rt"),
        "parameter_difference": last_value(records, "parameter_difference"),
        "nonfinite_metric_count": nonfinite_metric_count(records),
        "total_time_s": total_time,
        "steps_completed": len(core),
        "expected_step": expected_step,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    baseline_records = load_jsonl(args.baseline_dir / "metrics.jsonl")
    baseline_eval_loss = last_value(baseline_records, "eval_loss")
    baseline_summary = load_summary(args.baseline_dir / "summary.json")

    rows = []

    baseline_row = parse_run(args.baseline_dir, baseline_eval_loss, args.expected_step)
    rows.append({
        "bit": "baseline",
        "mode": "fault_free",
        "alpha": args.alpha,
        **baseline_row,
    })

    for bit_dir in sorted(args.checkpoint_root.glob("bit_*")):
        bit_text = bit_dir.name.removeprefix("bit_")
        for mode_dir in sorted(bit_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            if not (mode_dir / "summary.json").is_file():
                continue
            row = parse_run(mode_dir, baseline_eval_loss, args.expected_step)
            rows.append({
                "bit": bit_text,
                "mode": mode_dir.name,
                "alpha": args.alpha if mode_dir.name == "recompute" else "",
                **row,
            })

    fieldnames = [
        "bit", "mode", "alpha",
        "final_eval_loss", "eval_delta_vs_baseline",
        "trigger_count", "detected_count",
        "tp", "fp", "fn", "detection_rate", "recompute_precision",
        "correct_recompute", "incorrect_recompute",
        "max_gradient_norm_pre", "max_rt", "parameter_difference",
        "nonfinite_metric_count", "total_time_s", "steps_completed", "expected_step",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Baseline final eval loss: {baseline_eval_loss}")
    print(f"Baseline confusion: {baseline_summary.get('confusion_count', {})}")
    for row in rows:
        print(
            f"bit={row['bit']:>8} mode={row['mode']:<14} "
            f"eval={row['final_eval_loss']!s:<10} "
            f"delta={row['eval_delta_vs_baseline']!s:<10} "
            f"det_rate={row['detection_rate']!s:<8} "
            f"prec={row['recompute_precision']!s:<8} "
            f"tp={row['tp']} fp={row['fp']} fn={row['fn']}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
