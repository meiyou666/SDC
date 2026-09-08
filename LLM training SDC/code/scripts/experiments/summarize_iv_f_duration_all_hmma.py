#!/usr/bin/env python3
"""Summarize IV-F all-HMMA duration reruns and export Fig.3 time series."""

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


def as_number(value):
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.lower() in {"nan", "inf", "+inf", "-inf"}:
        return float(value)
    return None


def last_value(records, key):
    for record in reversed(records):
        if key in record:
            value = as_number(record[key])
            if value is not None:
                return value
    return ""


def max_finite(records, key):
    values = []
    for record in records:
        if key in record:
            value = as_number(record[key])
            if value is not None and math.isfinite(value):
                values.append(value)
    return max(values) if values else ""


def nonfinite_count(records):
    count = 0
    for record in records:
        for value in record.values():
            number = as_number(value)
            if number is not None and not math.isfinite(number):
                count += 1
    return count


def safe_label(label):
    return "".join(char if char.isalnum() or char == "_" else "_" for char in label)


def model_l2_difference(model_path, baseline_path):
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Post-hoc parameter comparison requires torch and safetensors") from exc

    total_squared = 0.0
    with safe_open(model_path, framework="pt", device="cpu") as model_file:
        with safe_open(baseline_path, framework="pt", device="cpu") as baseline_file:
            model_keys = list(model_file.keys())
            baseline_keys = list(baseline_file.keys())
            if model_keys != baseline_keys:
                raise ValueError(f"Tensor keys differ between {model_path} and {baseline_path}")
            for key in model_keys:
                difference = model_file.get_tensor(key).float() - baseline_file.get_tensor(key).float()
                total_squared += torch.sum(difference * difference, dtype=torch.float64).item()
    return math.sqrt(total_squared)


def grouped_by_step(records):
    steps = {}
    for record in records:
        step = record.get("_step")
        if not isinstance(step, int):
            continue
        merged = steps.setdefault(step, {"_step": step})
        merged.update(record)
    return steps


def load_summary(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_eval_loss(baseline_root, seed):
    records = load_jsonl(baseline_root / f"seed_{seed}" / "metrics.jsonl")
    return last_value(records, "eval_loss") if records else ""


def parse_run(run_dir, checkpoint_root, baseline_root, expected_step, trigger_step, compute_parameter_difference):
    relative = run_dir.relative_to(checkpoint_root)
    parts = relative.parts
    if len(parts) < 4 or parts[1] != "duration":
        raise ValueError(f"Unexpected IV-F duration path layout: {run_dir}")

    kernel_label = parts[0]
    setting = parts[2]
    duration_text = setting.removeprefix("steps_")
    duration = int(duration_text) if duration_text.isdigit() else ""
    seed_text = parts[3].removeprefix("seed_")
    seed = int(seed_text) if seed_text.isdigit() else ""

    records = load_jsonl(run_dir / "metrics.jsonl")
    core = [record for record in records if "loss" in record]
    summary = load_summary(run_dir / "summary.json")
    run_config = load_summary(run_dir / "run_config.json")
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

    baseline_eval_loss = load_baseline_eval_loss(baseline_root, seed) if seed != "" else ""
    final_eval_loss = last_value(records, "eval_loss")
    eval_delta = ""
    if isinstance(final_eval_loss, float) and isinstance(baseline_eval_loss, float):
        if math.isfinite(final_eval_loss) and math.isfinite(baseline_eval_loss):
            eval_delta = final_eval_loss - baseline_eval_loss
        elif not math.isfinite(final_eval_loss):
            eval_delta = final_eval_loss

    parameter_difference = last_value(records, "parameter_difference")
    parameter_difference_source = "metrics" if parameter_difference != "" else ""
    baseline_model_path = baseline_root / f"seed_{seed}" / f"model_{expected_step}" / "model.safetensors"
    if (
        parameter_difference == ""
        and compute_parameter_difference
        and complete
        and baseline_model_path.is_file()
    ):
        parameter_difference = model_l2_difference(checkpoint / "model.safetensors", baseline_model_path)
        parameter_difference_source = "posthoc_safetensors_fp32"

    expected_trigger_steps = []
    if isinstance(duration, int):
        expected_trigger_steps = list(range(trigger_step, trigger_step + duration))
    schedule_ok = trigger_steps == expected_trigger_steps
    confusion = summary.get("confusion_count", {}) if isinstance(summary.get("confusion_count"), dict) else {}

    return {
        "source": "iv_f_duration_all_hmma",
        "kernel_label": kernel_label,
        "setting": setting,
        "duration": duration,
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
        "final_parameter_difference": parameter_difference,
        "parameter_difference_source": parameter_difference_source,
        "max_rt": max_finite(records, "rt/rt"),
        "max_rt_jump": max_finite(records, "rt/rt_jump"),
        "trigger_count": len(trigger_steps),
        "first_trigger_step": trigger_steps[0] if trigger_steps else "",
        "last_trigger_step": trigger_steps[-1] if trigger_steps else "",
        "expected_first_trigger_step": trigger_step if isinstance(duration, int) else "",
        "expected_last_trigger_step": trigger_step + duration - 1 if isinstance(duration, int) else "",
        "trigger_schedule_ok": schedule_ok,
        "detected_count": sum(float(record.get("detected_anomaly", 0.0)) == 1.0 for record in core),
        "tp": confusion.get("tp", ""),
        "fp": confusion.get("fp", ""),
        "tn": confusion.get("tn", ""),
        "fn": confusion.get("fn", ""),
        "nonfinite_metric_count": nonfinite_count(records),
        "target_func_contains": run_config.get("target_func_contains", ""),
        "target_instr": run_config.get("target_instr", ""),
        "target_bitmask": run_config.get("target_bitmask", ""),
        "fi_nvbit_duration": run_config.get("fi_nvbit_duration", ""),
        "fi_nvbit_steps": " ".join(str(step) for step in run_config.get("fi_nvbit_steps", [])),
        "run_dir": str(run_dir),
    }


def finite_values(rows, metric):
    values = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                values.append(value)
    return values


def aggregate_rows(rows):
    metrics = (
        "final_eval_loss",
        "eval_loss_delta",
        "max_train_loss",
        "max_gradient_norm_pre",
        "max_attention_logits",
        "final_parameter_difference",
        "max_rt",
        "max_rt_jump",
        "trigger_count",
        "detected_count",
        "nonfinite_metric_count",
    )
    groups = {}
    for row in rows:
        if row["status"] == "complete":
            groups.setdefault((row["kernel_label"], row["duration"]), []).append(row)

    aggregates = []
    for (kernel_label, duration), group_rows in sorted(groups.items()):
        aggregate = {
            "kernel_label": kernel_label,
            "duration": duration,
            "complete_runs": len(group_rows),
        }
        for metric in metrics:
            values = finite_values(group_rows, metric)
            aggregate[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            aggregate[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0 if values else ""
        aggregates.append(aggregate)
    return aggregates


def export_fig3_timeseries(checkpoint_root, output, kernel_label, duration, seed, trigger_step, window_before, window_after):
    run_dir = checkpoint_root / safe_label(kernel_label) / "duration" / f"steps_{duration}" / f"seed_{seed}"
    records = load_jsonl(run_dir / "metrics.jsonl")
    if not records:
        print(f"Fig.3 time-series source not found yet: {run_dir / 'metrics.jsonl'}")
        return

    steps = grouped_by_step(records)
    start = max(1, trigger_step - window_before)
    end = trigger_step + duration - 1 + window_after
    fieldnames = [
        "_step",
        "relative_step",
        "injection_active",
        "loss",
        "rt",
        "rt_jump",
        "gradient_norm_pre",
        "gradient_norm_post",
        "detected_anomaly",
        "trigger_nvbit",
        "max_attention_logits",
        "attn_entropy",
        "lr",
        "parameter_difference",
    ]
    rows = []
    for step in range(start, end + 1):
        record = steps.get(step, {})
        rows.append({
            "_step": step,
            "relative_step": step - trigger_step,
            "injection_active": int(trigger_step <= step < trigger_step + duration),
            "loss": record.get("loss", ""),
            "rt": record.get("rt/rt", ""),
            "rt_jump": record.get("rt/rt_jump", ""),
            "gradient_norm_pre": record.get("gradient_norm_pre", ""),
            "gradient_norm_post": record.get("gradient_norm_post", ""),
            "detected_anomaly": record.get("detected_anomaly", ""),
            "trigger_nvbit": record.get("trigger_nvbit", ""),
            "max_attention_logits": record.get("max_attention_logits", ""),
            "attn_entropy": record.get("attn_entropy", ""),
            "lr": record.get("lr", ""),
            "parameter_difference": record.get("parameter_difference", ""),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved Fig.3 time-series CSV to: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/iv_f_duration_all_hmma_mb256"))
    parser.add_argument("--baseline-root", type=Path, default=Path("checkpoints/iv_f_mb256/baseline"))
    parser.add_argument("--output", type=Path, default=Path("logs/iv_f_duration_all_hmma_mb256/iv_f_duration_all_hmma_summary.csv"))
    parser.add_argument("--aggregate-output", type=Path, default=Path("logs/iv_f_duration_all_hmma_mb256/iv_f_duration_all_hmma_aggregate.csv"))
    parser.add_argument("--fig3-output", type=Path, default=Path("logs/iv_f_duration_all_hmma_mb256/iv_f_duration_all_hmma_fig3_timeseries.csv"))
    parser.add_argument("--fig3-kernel", default="BP8")
    parser.add_argument("--fig3-duration", type=int, default=3)
    parser.add_argument("--fig3-seed", type=int, default=42)
    parser.add_argument("--window-before", type=int, default=100)
    parser.add_argument("--window-after", type=int, default=200)
    parser.add_argument("--trigger-step", type=int, default=500)
    parser.add_argument("--expected-step", type=int, default=1000)
    parser.add_argument("--compute-parameter-difference", action="store_true")
    args = parser.parse_args()

    run_dirs = sorted({path.parent for path in args.checkpoint_root.rglob("metrics.jsonl")})
    rows = [
        parse_run(
            path,
            args.checkpoint_root,
            args.baseline_root,
            args.expected_step,
            args.trigger_step,
            args.compute_parameter_difference,
        )
        for path in run_dirs
    ]
    rows.sort(key=lambda row: (row["kernel_label"], row["duration"] or 0, row["seed"] or 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    aggregates = aggregate_rows(rows)
    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    if aggregates:
        with args.aggregate_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(aggregates)

    export_fig3_timeseries(
        args.checkpoint_root,
        args.fig3_output,
        args.fig3_kernel,
        args.fig3_duration,
        args.fig3_seed,
        args.trigger_step,
        args.window_before,
        args.window_after,
    )

    print("\nIV-F all-HMMA duration summary")
    print("kernel duration seed status     eval_loss       delta          triggers schedule max_grad_pre max_rt       param_diff")
    for row in rows:
        print(
            f'{row["kernel_label"]:<6} {row["duration"]!s:<8} {row["seed"]!s:<4} '
            f'{row["status"]:<10} {str(row["final_eval_loss"]):<15} '
            f'{str(row["eval_loss_delta"]):<14} {row["trigger_count"]:<8} '
            f'{str(row["trigger_schedule_ok"]):<8} {str(row["max_gradient_norm_pre"]):<12} '
            f'{str(row["max_rt"]):<12} {row["final_parameter_difference"]}'
        )
    print(f"Saved summary CSV to: {args.output}")
    print(f"Saved aggregate CSV to: {args.aggregate_output}")


if __name__ == "__main__":
    main()
