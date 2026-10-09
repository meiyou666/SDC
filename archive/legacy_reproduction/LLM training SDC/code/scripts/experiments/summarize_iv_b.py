#!/usr/bin/env python3
"""Summarize one bit batch of the simplified Section IV-B kernel sweep."""

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


def bit_field(bit):
    local_bit = bit % 16
    packed_value = "low_bf16" if bit < 16 else "high_bf16"
    if local_bit <= 6:
        field = "mantissa"
    elif local_bit <= 14:
        field = "exponent"
    else:
        field = "sign"
    return packed_value, local_bit, field


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


def load_manifest(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bit", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--compute-parameter-difference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=1000)
    parser.add_argument("--expected-trigger-count", type=int)
    parser.add_argument("--expected-first-trigger", type=int)
    parser.add_argument("--expected-last-trigger", type=int)
    args = parser.parse_args()

    if not (0 <= args.bit <= 31):
        parser.error("Expected 0 <= --bit <= 31")

    baseline_eval_loss = ""
    baseline_model_path = None
    if args.baseline_dir is not None:
        baseline_records = load_jsonl(args.baseline_dir / "metrics.jsonl")
        if not baseline_records:
            raise ValueError(f"Baseline metrics not found: {args.baseline_dir / 'metrics.jsonl'}")
        baseline_eval_loss = last_value(baseline_records, "eval_loss")
        baseline_model_path = args.baseline_dir / f"model_{args.expected_step}" / "model.safetensors"
        if args.compute_parameter_difference and not baseline_model_path.is_file():
            raise ValueError(f"Baseline model not found: {baseline_model_path}")

    packed_value, local_bit, field = bit_field(args.bit)
    rows = []
    for manifest_row in load_manifest(args.manifest):
        kernel_label = manifest_row["kernel_label"]
        run_dir = args.checkpoint_root / safe_label(kernel_label)
        records = load_jsonl(run_dir / "metrics.jsonl")
        core = [record for record in records if "loss" in record]
        run_config = {}
        if (run_dir / "run_config.json").is_file():
            run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

        checkpoint = run_dir / f"model_{args.expected_step}"
        complete = all(
            path.is_file()
            for path in (
                checkpoint / "model.safetensors",
                checkpoint / "training_state.json",
                run_dir / "summary.json",
                run_dir / "run_config.json",
            )
        )
        status = "complete" if complete else "failed" if (run_dir / "FAILED").is_file() else "missing"
        final_eval_loss = last_value(records, "eval_loss")
        eval_delta = ""
        if isinstance(final_eval_loss, float) and isinstance(baseline_eval_loss, float):
            if math.isfinite(final_eval_loss) and math.isfinite(baseline_eval_loss):
                eval_delta = final_eval_loss - baseline_eval_loss
            elif not math.isfinite(final_eval_loss):
                eval_delta = final_eval_loss

        parameter_difference = last_value(records, "parameter_difference")
        parameter_difference_source = "metrics" if parameter_difference != "" else ""
        if parameter_difference == "" and args.compute_parameter_difference and complete and baseline_model_path is not None:
            parameter_difference = model_l2_difference(checkpoint / "model.safetensors", baseline_model_path)
            parameter_difference_source = "posthoc_safetensors_fp32"

        trigger_steps = [
            int(record.get("_step", -1))
            for record in core
            if float(record.get("trigger_nvbit", 0.0)) == 1.0
        ]
        schedule_checks = []
        if args.expected_trigger_count is not None:
            schedule_checks.append(len(trigger_steps) == args.expected_trigger_count)
        if args.expected_first_trigger is not None:
            schedule_checks.append(bool(trigger_steps) and trigger_steps[0] == args.expected_first_trigger)
        if args.expected_last_trigger is not None:
            schedule_checks.append(bool(trigger_steps) and trigger_steps[-1] == args.expected_last_trigger)

        rows.append({
            "bit_position": args.bit,
            "bitmask": 1 << args.bit,
            "packed_value": packed_value,
            "local_bf16_bit": local_bit,
            "field": field,
            "kernel_label": kernel_label,
            "location": manifest_row["location"],
            "status": status,
            "baseline_eval_loss": baseline_eval_loss,
            "final_eval_loss": final_eval_loss,
            "eval_loss_delta": eval_delta,
            "final_parameter_difference": parameter_difference,
            "parameter_difference_source": parameter_difference_source,
            "max_gradient_norm_pre": max_finite(core, "gradient_norm_pre"),
            "max_gradient_norm_post": max_finite(core, "gradient_norm_post"),
            "max_attention_logits": max_finite(records, "max_attention_logits"),
            "max_training_loss": max_finite(core, "loss"),
            "max_rt": max_finite(records, "rt/rt"),
            "trigger_count": len(trigger_steps),
            "first_trigger_step": trigger_steps[0] if trigger_steps else "",
            "last_trigger_step": trigger_steps[-1] if trigger_steps else "",
            "trigger_schedule_ok": all(schedule_checks) if schedule_checks else "",
            "detected_count": sum(float(record.get("detected_anomaly", 0.0)) == 1.0 for record in core),
            "nonfinite_metric_count": nonfinite_count(records),
            "target_instr": run_config.get("target_instr", ""),
            "manifest_target_instr": manifest_row.get("target_instr", ""),
            "target_bitmask_logged": run_config.get("target_bitmask", ""),
            "target_func_contains": run_config.get("target_func_contains", ""),
            "profile_function_id_first_seen": manifest_row.get("function_id_first_seen", ""),
            "profile_launch_count": manifest_row.get("launch_count", ""),
            "profile_hmma_instr_count": manifest_row.get("hmma_instr_count", ""),
            "profile_hmma_instr_indexes": manifest_row.get("hmma_instr_indexes", ""),
            "kernel_name": manifest_row["kernel_name"],
            "run_dir": str(run_dir),
        })

    if not rows:
        raise ValueError("No manifest rows selected for summary")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("\nIV-B kernel-sensitivity summary")
    print("bit kernel location status    eval_loss       delta           param_diff      triggers schedule nonfinite")
    for row in rows:
        print(
            f'{row["bit_position"]:>2}  {row["kernel_label"]:<6} {row["location"]:<8} {row["status"]:<9} '
            f'{str(row["final_eval_loss"]):<15} {str(row["eval_loss_delta"]):<15} '
            f'{str(row["final_parameter_difference"]):<15} '
            f'{row["trigger_count"]:<8} {str(row["trigger_schedule_ok"]):<8} '
            f'{row["nonfinite_metric_count"]}'
        )
    print(f"\nSaved IV-B summary to: {args.output}")


if __name__ == "__main__":
    main()
