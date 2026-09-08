#!/usr/bin/env python3
"""Summarize one or both independent Section IV-A bit-position groups."""

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
    value = "low_bf16" if bit < 16 else "high_bf16"
    if local_bit <= 6:
        field = "mantissa"
    elif local_bit <= 14:
        field = "exponent"
    else:
        field = "sign"
    return value, local_bit, field


def paper_expectation(bit):
    if bit in {10, 11, 26, 27}:
        return "measurable_eval_loss_increase"
    if bit in {12, 13, 14, 28, 29, 30}:
        return "nan_expected"
    return "no_noticeable_eval_loss_change"


def resolve_run_dir(checkpoint_roots, bit):
    candidates = [root / f"bit_{bit:02d}" for root in checkpoint_roots]
    existing = [path for path in candidates if path.exists()]
    if len(existing) > 1:
        raise ValueError(
            f"Bit {bit} exists under multiple checkpoint roots: "
            + ", ".join(str(path) for path in existing)
        )
    return existing[0] if existing else candidates[0]


def model_l2_difference(model_path, baseline_path):
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "Post-hoc parameter comparison requires torch and safetensors"
        ) from exc

    total_squared = 0.0
    with safe_open(model_path, framework="pt", device="cpu") as model_file:
        with safe_open(baseline_path, framework="pt", device="cpu") as baseline_file:
            model_keys = list(model_file.keys())
            baseline_keys = list(baseline_file.keys())
            if model_keys != baseline_keys:
                raise ValueError(
                    f"Tensor keys differ between {model_path} and {baseline_path}"
                )
            for key in model_keys:
                model_tensor = model_file.get_tensor(key)
                baseline_tensor = baseline_file.get_tensor(key)
                if model_tensor.shape != baseline_tensor.shape:
                    raise ValueError(f"Tensor shape differs for {key}")
                difference = model_tensor.float() - baseline_tensor.float()
                total_squared += torch.sum(
                    difference * difference, dtype=torch.float64
                ).item()
    return math.sqrt(total_squared)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        dest="checkpoint_roots",
        help="A directory containing bit_XX runs; repeat for both groups",
    )
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--compute-parameter-difference", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("logs/iv_a/iv_a_summary.csv"))
    parser.add_argument("--expected-step", type=int, default=1000)
    parser.add_argument("--bit-start", type=int, default=0)
    parser.add_argument("--bit-end", type=int, default=31)
    parser.add_argument("--skip-bit", type=int, action="append", default=[])
    parser.add_argument("--expected-trigger-count", type=int)
    parser.add_argument("--expected-first-trigger", type=int)
    parser.add_argument("--expected-last-trigger", type=int)
    args = parser.parse_args()

    checkpoint_roots = args.checkpoint_roots or [Path("checkpoints/iv_a")]
    if not (0 <= args.bit_start <= args.bit_end <= 31):
        parser.error("Expected 0 <= --bit-start <= --bit-end <= 31")
    if args.compute_parameter_difference and args.baseline_dir is None:
        parser.error("--compute-parameter-difference requires --baseline-dir")

    baseline_eval_loss = ""
    baseline_model_path = None
    if args.baseline_dir is not None:
        baseline_records = load_jsonl(args.baseline_dir / "metrics.jsonl")
        if not baseline_records:
            raise ValueError(f"Baseline metrics not found: {args.baseline_dir / 'metrics.jsonl'}")
        baseline_eval_loss = last_value(baseline_records, "eval_loss")
        baseline_model_path = (
            args.baseline_dir
            / f"model_{args.expected_step}"
            / "model.safetensors"
        )
        if args.compute_parameter_difference and not baseline_model_path.is_file():
            raise ValueError(f"Baseline model not found: {baseline_model_path}")

    rows = []
    skip_bits = set(args.skip_bit)
    for bit in range(args.bit_start, args.bit_end + 1):
        if bit in skip_bits:
            continue
        run_dir = resolve_run_dir(checkpoint_roots, bit)
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
        if (
            parameter_difference == ""
            and args.compute_parameter_difference
            and complete
            and baseline_model_path is not None
        ):
            parameter_difference = model_l2_difference(
                checkpoint / "model.safetensors", baseline_model_path
            )
            parameter_difference_source = "posthoc_safetensors_fp32"

        packed_value, local_bit, field = bit_field(bit)
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
        trigger_schedule_ok = all(schedule_checks) if schedule_checks else ""

        rows.append({
            "bit_position": bit,
            "bitmask": 1 << bit,
            "packed_value": packed_value,
            "local_bf16_bit": local_bit,
            "field": field,
            "paper_expectation": paper_expectation(bit),
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
            "trigger_schedule_ok": trigger_schedule_ok,
            "detected_count": sum(float(record.get("detected_anomaly", 0.0)) == 1.0 for record in core),
            "nonfinite_metric_count": nonfinite_count(records),
            "target_instr": run_config.get("target_instr", ""),
            "target_bitmask_logged": run_config.get("target_bitmask", ""),
            "run_dir": str(run_dir),
        })

    if not rows:
        raise ValueError("No bit rows selected for summary")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nIV-A bit-position summary")
    print("bit  field      status    eval_loss       delta           param_diff      triggers schedule nonfinite")
    for row in rows:
        print(
            f'{row["bit_position"]:>2}   {row["field"]:<10} {row["status"]:<9} '
            f'{str(row["final_eval_loss"]):<15} {str(row["eval_loss_delta"]):<15} '
            f'{str(row["final_parameter_difference"]):<15} '
            f'{row["trigger_count"]:<8} {str(row["trigger_schedule_ok"]):<8} '
            f'{row["nonfinite_metric_count"]}'
        )
    print(f"\nSaved IV-A summary to: {args.output}")


if __name__ == "__main__":
    main()
