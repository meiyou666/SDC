#!/usr/bin/env python3
"""Summarize VII-B 60M runs (paper Table I, 60M column).

Reports the eval-loss curve (recorded every --eval-every steps), final eval
loss and delta vs the fault-free baseline, detection statistics, recompute
statistics, and steady-state throughput for the three conditions:
baseline / FI (no recompute) / FI + recompute.
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


def eval_curve(records):
    return [
        (int(record["_step"]), float(record["eval_loss"]))
        for record in records
        if "eval_loss" in record and isinstance(record.get("eval_loss"), (int, float))
    ]


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


def steady_state_s_per_it(records, warmup_steps=100):
    """Median per-update-step time in seconds, excluding the first warmup_steps."""
    core = [record for record in records if "loss" in record and "_time" in record]
    times = [(int(r["_step"]), float(r["_time"])) for r in core]
    deltas = []
    for (step_a, time_a), (step_b, time_b) in zip(times, times[1:]):
        if step_b <= warmup_steps:
            continue
        dt = time_b - time_a
        if dt > 0 and math.isfinite(dt):
            deltas.append(dt)
    if not deltas:
        return ""
    deltas.sort()
    return deltas[len(deltas) // 2]


def confusion_rates(summary):
    confusion = summary.get("confusion_count", {})
    tp = confusion.get("tp", 0)
    fp = confusion.get("fp", 0)
    fn = confusion.get("fn", 0)
    detection_rate = tp / (tp + fn) if (tp + fn) else ""
    precision = tp / (tp + fp) if (tp + fp) else ""
    return tp, fp, fn, detection_rate, precision


def parse_condition(condition_dir, baseline_eval_loss, expected_step, warmup_steps):
    records = load_jsonl(condition_dir / "metrics.jsonl")
    summary = load_summary(condition_dir / "summary.json")
    core = [record for record in records if "loss" in record]

    curve = eval_curve(records)
    final_eval = curve[-1][1] if curve else ""
    eval_delta = ""
    if isinstance(final_eval, float) and isinstance(baseline_eval_loss, float) \
            and math.isfinite(final_eval) and math.isfinite(baseline_eval_loss):
        eval_delta = final_eval - baseline_eval_loss

    trigger_count = sum(
        1 for record in core if float(record.get("trigger_nvbit", 0.0)) == 1.0
    )
    detected_count = sum(
        1 for record in core if float(record.get("detected_anomaly", 0.0)) == 1.0
    )
    tp, fp, fn, detection_rate, precision = confusion_rates(summary)
    confusion = summary.get("confusion_count", {})

    return {
        "final_eval_loss": final_eval,
        "eval_delta_vs_baseline": eval_delta,
        "eval_curve": curve,
        "trigger_count": trigger_count,
        "detected_count": detected_count,
        "tp": tp, "fp": fp, "fn": fn,
        "detection_rate": detection_rate,
        "recompute_precision": precision,
        "correct_recompute": confusion.get("count_correct_recompute", ""),
        "incorrect_recompute": confusion.get("count_incorrect_recompute", ""),
        "max_gradient_norm_pre": max_finite(core, "gradient_norm_pre"),
        "max_rt": max_finite(records, "rt/rt"),
        "nonfinite_metric_count": nonfinite_metric_count(records),
        "total_time_s": summary.get("total_time", ""),
        "steps_completed": len(core),
        "expected_step": expected_step,
        "s_per_it_steady_state": steady_state_s_per_it(records, warmup_steps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bit", type=int, required=True)
    args = parser.parse_args()

    baseline_records = load_jsonl(args.baseline_dir / "metrics.jsonl")
    baseline_curve = eval_curve(baseline_records)
    baseline_final = baseline_curve[-1][1] if baseline_curve else None

    conditions = ["baseline", "fi", "fi_recompute"]
    parsed = {}
    for condition in conditions:
        candidates = sorted((args.checkpoint_root / condition).glob("seed_*"))
        if not candidates:
            continue
        parsed[condition] = parse_condition(
            candidates[0], baseline_final, args.expected_step, args.warmup_steps
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Long-format CSV: one row per condition and eval step, with the eval loss
    # delta vs the baseline curve at the same step.
    baseline_curve_map = dict(baseline_curve)
    eval_steps = set(baseline_curve_map)
    for condition in conditions:
        if condition in parsed:
            eval_steps |= {step for step, _ in parsed[condition]["eval_curve"]}
    eval_steps = sorted(eval_steps)

    def step_delta(curve_map, step):
        value = curve_map.get(step)
        ref = baseline_curve_map.get(step)
        if isinstance(value, float) and isinstance(ref, float) \
                and math.isfinite(value) and math.isfinite(ref):
            return value - ref
        return ""

    fieldnames = ["condition", "alpha", "bit", "eval_step", "eval_loss", "eval_delta_vs_baseline"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in conditions:
            if condition not in parsed:
                continue
            row_base = {
                "condition": condition,
                "alpha": args.alpha if condition == "fi_recompute" else "",
                "bit": args.bit,
            }
            curve = dict(parsed[condition]["eval_curve"])
            for step in eval_steps:
                writer.writerow({**row_base, "eval_step": step, "eval_loss": curve.get(step, ""),
                                 "eval_delta_vs_baseline": step_delta(curve, step)})

    # Sidecar CSV: per-condition summary statistics.
    stats_path = args.output.with_name(args.output.stem + "_stats.csv")
    stats_fields = [
        "condition", "final_eval_loss", "eval_delta_vs_baseline",
        "trigger_count", "detected_count", "tp", "fp", "fn",
        "detection_rate", "recompute_precision",
        "correct_recompute", "incorrect_recompute",
        "max_gradient_norm_pre", "max_rt", "nonfinite_metric_count",
        "total_time_s", "steps_completed", "expected_step", "s_per_it_steady_state",
    ]
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stats_fields)
        writer.writeheader()
        for condition in conditions:
            if condition not in parsed:
                continue
            row = {key: value for key, value in parsed[condition].items() if key != "eval_curve"}
            row["condition"] = condition
            writer.writerow(row)

    print(f"Baseline final eval loss: {baseline_final}")
    for condition in conditions:
        if condition not in parsed:
            print(f"{condition}: MISSING")
            continue
        row = parsed[condition]
        print(
            f"{condition:<12} final_eval={row['final_eval_loss']!s:<10} "
            f"delta={row['eval_delta_vs_baseline']!s:<10} "
            f"det_rate={row['detection_rate']!s:<8} "
            f"prec={row['recompute_precision']!s:<8} "
            f"s/it={row['s_per_it_steady_state']!s:<8} "
            f"tp={row['tp']} fp={row['fp']} fn={row['fn']} "
            f"recomp_ok={row['correct_recompute']} recomp_bad={row['incorrect_recompute']}"
        )
    print(f"Wrote {args.output}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
