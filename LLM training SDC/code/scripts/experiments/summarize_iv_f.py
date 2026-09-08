#!/usr/bin/env python3
"""Summarize Section IV-F baseline/rate/duration runs using only stdlib."""

import argparse
import csv
import json
import math
import statistics
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


def max_value(records, key):
    values = numeric_values(records, key)
    return max(values) if values else ""


def parse_run(run_dir, checkpoint_root, expected_step):
    relative = run_dir.relative_to(checkpoint_root)
    parts = relative.parts
    condition = parts[0]
    setting = "baseline"
    if condition == "rate":
        setting = parts[1]
    elif condition == "duration":
        setting = parts[1]

    seed = int(parts[-1].removeprefix("seed_"))
    records = load_jsonl(run_dir / "metrics.jsonl")
    core = [record for record in records if "loss" in record]
    trigger_steps = [
        int(record.get("_step", -1))
        for record in core
        if float(record.get("trigger_nvbit", 0.0)) == 1.0
    ]
    nonfinite_count = 0
    for record in records:
        for value in record.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    nonfinite_count += 1

    checkpoint = run_dir / f"model_{expected_step}"
    complete = all(
        path.is_file()
        for path in (
            checkpoint / "model.safetensors",
            checkpoint / "optimizer.pt",
            checkpoint / "training_state.json",
            run_dir / "summary.json",
            run_dir / "run_config.json",
        )
    )
    return {
        "condition": condition,
        "setting": setting,
        "seed": seed,
        "status": "complete" if complete else "incomplete",
        "final_eval_loss": last_value(records, "eval_loss"),
        "final_train_loss": last_value(core, "loss"),
        "max_train_loss": max_value(core, "loss"),
        "max_gradient_norm_pre": max_value(core, "gradient_norm_pre"),
        "max_gradient_norm_post": max_value(core, "gradient_norm_post"),
        "max_attention_logits": max_value(records, "max_attention_logits"),
        "final_parameter_difference": last_value(records, "parameter_difference"),
        "max_rt": max_value(records, "rt/rt"),
        "trigger_count": len(trigger_steps),
        "first_trigger_step": trigger_steps[0] if trigger_steps else "",
        "last_trigger_step": trigger_steps[-1] if trigger_steps else "",
        "detected_count": sum(float(record.get("detected_anomaly", 0.0)) == 1.0 for record in core),
        "nonfinite_metric_count": nonfinite_count,
        "run_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/iv_f"))
    parser.add_argument("--output", type=Path, default=Path("logs/iv_f/iv_f_summary.csv"))
    parser.add_argument("--aggregate-output", type=Path, default=Path("logs/iv_f/iv_f_aggregate.csv"))
    parser.add_argument("--best-seed-file", type=Path, default=Path("checkpoints/iv_f/baseline/best_seed.txt"))
    parser.add_argument("--expected-step", type=int, default=1000)
    args = parser.parse_args()

    run_dirs = sorted({path.parent for path in args.checkpoint_root.rglob("metrics.jsonl")})
    rows = [parse_run(path, args.checkpoint_root, args.expected_step) for path in run_dirs]
    rows.sort(key=lambda row: (row["condition"], row["setting"], row["seed"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    aggregate_metrics = (
        "final_eval_loss",
        "max_train_loss",
        "max_gradient_norm_pre",
        "max_attention_logits",
        "final_parameter_difference",
        "max_rt",
        "trigger_count",
        "detected_count",
        "nonfinite_metric_count",
    )
    groups = {}
    for row in rows:
        if row["status"] == "complete":
            groups.setdefault((row["condition"], row["setting"]), []).append(row)
    aggregate_rows = []
    for (condition, setting), group_rows in sorted(groups.items()):
        aggregate = {
            "condition": condition,
            "setting": setting,
            "complete_runs": len(group_rows),
        }
        for metric in aggregate_metrics:
            values = [
                float(row[metric]) for row in group_rows
                if isinstance(row[metric], (int, float)) and math.isfinite(float(row[metric]))
            ]
            aggregate[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            aggregate[f"{metric}_std"] = statistics.pstdev(values) if values else ""
        aggregate_rows.append(aggregate)
    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    if aggregate_rows:
        with args.aggregate_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
            writer.writeheader()
            writer.writerows(aggregate_rows)

    baselines = [
        row for row in rows
        if row["condition"] == "baseline"
        and row["status"] == "complete"
        and isinstance(row["final_eval_loss"], float)
        and math.isfinite(row["final_eval_loss"])
    ]
    best = min(baselines, key=lambda row: row["final_eval_loss"]) if baselines else None
    if best:
        args.best_seed_file.parent.mkdir(parents=True, exist_ok=True)
        args.best_seed_file.write_text(f'{best["seed"]}\n', encoding="utf-8")

    print("\nIV-F summary")
    print("condition   setting       seed       status      eval_loss  triggers  max_grad_pre  param_diff")
    for row in rows:
        print(
            f'{row["condition"]:<11} {row["setting"]:<13} {row["seed"]:<10} '
            f'{row["status"]:<11} {str(row["final_eval_loss"]):<10} '
            f'{row["trigger_count"]:<9} {str(row["max_gradient_norm_pre"]):<13} '
            f'{row["final_parameter_difference"]}'
        )
    if best:
        print(f'\nBest seed: {best["seed"]} (lowest final baseline eval loss: {best["final_eval_loss"]})')
        print(f"Saved best seed to: {args.best_seed_file}")
    print(f"Saved summary CSV to: {args.output}")
    print(f"Saved aggregate mean/std CSV to: {args.aggregate_output}")


if __name__ == "__main__":
    main()
