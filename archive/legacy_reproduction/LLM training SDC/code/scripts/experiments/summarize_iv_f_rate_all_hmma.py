#!/usr/bin/env python3
"""Summarize IV-F single-seed rate reruns that use all-HMMA injection."""

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


def load_baseline_eval_loss(baseline_dir):
    records = load_jsonl(baseline_dir / "metrics.jsonl")
    if not records:
        return ""
    return last_value(records, "eval_loss")


def parse_run(run_dir, checkpoint_root, baseline_eval_loss, expected_step):
    relative = run_dir.relative_to(checkpoint_root)
    parts = relative.parts
    kernel_label = parts[0] if len(parts) >= 1 else ""
    setting = parts[2] if len(parts) >= 3 else ""
    seed_text = parts[-1].removeprefix("seed_")
    seed = int(seed_text) if seed_text.isdigit() else ""
    rate_text = setting.removeprefix("every_")
    trigger_rate = int(rate_text) if rate_text.isdigit() else ""

    records = load_jsonl(run_dir / "metrics.jsonl")
    core = [record for record in records if "loss" in record]
    trigger_steps = [
        int(record.get("_step", -1))
        for record in core
        if float(record.get("trigger_nvbit", 0.0)) == 1.0
    ]

    checkpoint = run_dir / f"model_{expected_step}"
    complete = all(
        path.is_file()
        for path in (
            checkpoint / "model.safetensors",
            checkpoint / "training_state.json",
            run_dir / "metrics.jsonl",
            run_dir / "summary.json",
            run_dir / "run_config.json",
        )
    )

    final_eval_loss = last_value(records, "eval_loss")
    eval_delta = ""
    if isinstance(final_eval_loss, float) and isinstance(baseline_eval_loss, float):
        if math.isfinite(final_eval_loss) and math.isfinite(baseline_eval_loss):
            eval_delta = final_eval_loss - baseline_eval_loss
        elif not math.isfinite(final_eval_loss):
            eval_delta = final_eval_loss

    return {
        "source": "iv_f_rate_all_hmma",
        "kernel_label": kernel_label,
        "setting": setting,
        "trigger_rate": trigger_rate,
        "seed": seed,
        "status": "complete" if complete else "incomplete",
        "baseline_eval_loss": baseline_eval_loss,
        "final_eval_loss": final_eval_loss,
        "eval_loss_delta": eval_delta,
        "final_train_loss": last_value(core, "loss"),
        "max_train_loss": max_finite(core, "loss"),
        "max_gradient_norm_pre": max_finite(core, "gradient_norm_pre"),
        "max_gradient_norm_post": max_finite(core, "gradient_norm_post"),
        "max_attention_logits": max_finite(records, "max_attention_logits"),
        "final_parameter_difference": last_value(records, "parameter_difference"),
        "max_rt": max_finite(records, "rt/rt"),
        "trigger_count": len(trigger_steps),
        "first_trigger_step": trigger_steps[0] if trigger_steps else "",
        "last_trigger_step": trigger_steps[-1] if trigger_steps else "",
        "detected_count": sum(float(record.get("detected_anomaly", 0.0)) == 1.0 for record in core),
        "nonfinite_metric_count": nonfinite_metric_count(records),
        "run_dir": str(run_dir),
    }


def parse_reused_ivb_every10(run_dir, baseline_eval_loss, expected_step):
    kernel_label = run_dir.name
    bit_text = run_dir.parent.name.removeprefix("bit_") if run_dir.parent else ""
    seed_text = run_dir.parent.parent.name.removeprefix("seed_") if run_dir.parent.parent else ""
    seed = int(seed_text) if seed_text.isdigit() else ""

    records = load_jsonl(run_dir / "metrics.jsonl")
    core = [record for record in records if "loss" in record]
    trigger_steps = [
        int(record.get("_step", -1))
        for record in core
        if float(record.get("trigger_nvbit", 0.0)) == 1.0
    ]

    checkpoint = run_dir / f"model_{expected_step}"
    complete = all(
        path.is_file()
        for path in (
            checkpoint / "model.safetensors",
            checkpoint / "training_state.json",
            run_dir / "metrics.jsonl",
            run_dir / "summary.json",
            run_dir / "run_config.json",
        )
    )

    final_eval_loss = last_value(records, "eval_loss")
    eval_delta = ""
    if isinstance(final_eval_loss, float) and isinstance(baseline_eval_loss, float):
        if math.isfinite(final_eval_loss) and math.isfinite(baseline_eval_loss):
            eval_delta = final_eval_loss - baseline_eval_loss
        elif not math.isfinite(final_eval_loss):
            eval_delta = final_eval_loss

    return {
        "source": "iv_b_reuse",
        "kernel_label": kernel_label,
        "setting": "every_10",
        "trigger_rate": 10,
        "seed": seed,
        "status": "complete" if complete else "incomplete",
        "baseline_eval_loss": baseline_eval_loss,
        "final_eval_loss": final_eval_loss,
        "eval_loss_delta": eval_delta,
        "final_train_loss": last_value(core, "loss"),
        "max_train_loss": max_finite(core, "loss"),
        "max_gradient_norm_pre": max_finite(core, "gradient_norm_pre"),
        "max_gradient_norm_post": max_finite(core, "gradient_norm_post"),
        "max_attention_logits": max_finite(records, "max_attention_logits"),
        "final_parameter_difference": last_value(records, "parameter_difference"),
        "max_rt": max_finite(records, "rt/rt"),
        "trigger_count": len(trigger_steps),
        "first_trigger_step": trigger_steps[0] if trigger_steps else "",
        "last_trigger_step": trigger_steps[-1] if trigger_steps else "",
        "detected_count": sum(float(record.get("detected_anomaly", 0.0)) == 1.0 for record in core),
        "nonfinite_metric_count": nonfinite_metric_count(records),
        "run_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/iv_f_rate_all_hmma_mb256"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("checkpoints/iv_f_mb256/baseline/seed_42"))
    parser.add_argument("--output", type=Path, default=Path("logs/iv_f_rate_all_hmma_mb256/iv_f_rate_all_hmma_summary.csv"))
    parser.add_argument("--expected-step", type=int, default=1000)
    parser.add_argument("--reuse-every10-dir", type=Path, action="append", default=[])
    args = parser.parse_args()

    baseline_eval_loss = load_baseline_eval_loss(args.baseline_dir)
    run_dirs = sorted({path.parent for path in args.checkpoint_root.rglob("metrics.jsonl")})
    rows = [parse_run(path, args.checkpoint_root, baseline_eval_loss, args.expected_step) for path in run_dirs]
    for reuse_dir in args.reuse_every10_dir:
        if (reuse_dir / "metrics.jsonl").is_file():
            rows.append(parse_reused_ivb_every10(reuse_dir, baseline_eval_loss, args.expected_step))
    rows.sort(key=lambda row: (row["kernel_label"], row["trigger_rate"] or 0, row["seed"] or 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\nIV-F all-HMMA rate summary")
    print("source              kernel setting    seed status     eval_loss       delta          triggers max_grad_pre param_diff")
    for row in rows:
        print(
            f'{row["source"]:<19} {row["kernel_label"]:<6} {row["setting"]:<10} {row["seed"]!s:<4} '
            f'{row["status"]:<10} {str(row["final_eval_loss"]):<15} '
            f'{str(row["eval_loss_delta"]):<14} {row["trigger_count"]:<8} '
            f'{str(row["max_gradient_norm_pre"]):<12} {row["final_parameter_difference"]}'
        )
    print(f"Saved summary CSV to: {args.output}")


if __name__ == "__main__":
    main()
