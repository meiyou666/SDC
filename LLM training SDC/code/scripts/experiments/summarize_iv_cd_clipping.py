#!/usr/bin/env python3
"""Summarize IV-C/D clipped vs no-clipping comparison runs."""

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


def first_nonfinite_step(records):
    for record in records:
        for value in record.values():
            number = as_number(value)
            if number is not None and not math.isfinite(number):
                return record.get("_step", "")
    return ""


def nonfinite_count(records):
    count = 0
    for record in records:
        for value in record.values():
            number = as_number(value)
            if number is not None and not math.isfinite(number):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, help="Root containing clipped/ and no_clipping/ subdirectories")
    parser.add_argument("--clipped-dir", type=Path, help="Existing IV-B clipped run directory")
    parser.add_argument("--no-clipping-dir", type=Path, help="New no-clipping run directory")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--kernel-label", required=True)
    parser.add_argument("--bit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=1000)
    args = parser.parse_args()

    baseline_eval_loss = ""
    if args.baseline_dir is not None:
        baseline_eval_loss = last_value(load_jsonl(args.baseline_dir / "metrics.jsonl"), "eval_loss")

    rows = []
    if args.checkpoint_root is not None:
        condition_dirs = {
            "clipped": args.checkpoint_root / "clipped",
            "no_clipping": args.checkpoint_root / "no_clipping",
        }
    elif args.clipped_dir is not None and args.no_clipping_dir is not None:
        condition_dirs = {
            "clipped": args.clipped_dir,
            "no_clipping": args.no_clipping_dir,
        }
    else:
        parser.error("Use either --checkpoint-root or both --clipped-dir and --no-clipping-dir")

    for condition, run_dir in condition_dirs.items():
        records = load_jsonl(run_dir / "metrics.jsonl")
        core = [record for record in records if "loss" in record]
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
        run_config = {}
        if (run_dir / "run_config.json").is_file():
            run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

        final_eval_loss = last_value(records, "eval_loss")
        eval_delta = ""
        if isinstance(final_eval_loss, float) and isinstance(baseline_eval_loss, float):
            if math.isfinite(final_eval_loss) and math.isfinite(baseline_eval_loss):
                eval_delta = final_eval_loss - baseline_eval_loss
            elif not math.isfinite(final_eval_loss):
                eval_delta = final_eval_loss

        trigger_steps = [
            int(record.get("_step", -1))
            for record in core
            if float(record.get("trigger_nvbit", 0.0)) == 1.0
        ]
        rows.append({
            "kernel_label": args.kernel_label,
            "bit_position": args.bit,
            "condition": condition,
            "status": status,
            "grad_clipping": run_config.get("grad_clipping", ""),
            "baseline_eval_loss": baseline_eval_loss,
            "final_eval_loss": final_eval_loss,
            "eval_loss_delta": eval_delta,
            "final_train_loss": last_value(core, "loss"),
            "max_training_loss": max_finite(core, "loss"),
            "max_gradient_norm_pre": max_finite(core, "gradient_norm_pre"),
            "max_gradient_norm_post": max_finite(core, "gradient_norm_post"),
            "max_attention_logits": max_finite(records, "max_attention_logits"),
            "max_rt": max_finite(records, "rt/rt"),
            "trigger_count": len(trigger_steps),
            "first_trigger_step": trigger_steps[0] if trigger_steps else "",
            "last_trigger_step": trigger_steps[-1] if trigger_steps else "",
            "detected_count": sum(float(record.get("detected_anomaly", 0.0)) == 1.0 for record in core),
            "first_nonfinite_step": first_nonfinite_step(records),
            "nonfinite_metric_count": nonfinite_count(records),
            "target_instr": run_config.get("target_instr", ""),
            "target_func_contains": run_config.get("target_func_contains", ""),
            "run_dir": str(run_dir),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nIV-C/D clipping comparison summary")
    print("condition     status    final_eval      delta          max_grad_pre   max_grad_post  triggers nonfinite")
    for row in rows:
        print(
            f'{row["condition"]:<13} {row["status"]:<9} {str(row["final_eval_loss"]):<15} '
            f'{str(row["eval_loss_delta"]):<14} {str(row["max_gradient_norm_pre"]):<14} '
            f'{str(row["max_gradient_norm_post"]):<14} {row["trigger_count"]:<8} '
            f'{row["nonfinite_metric_count"]}'
        )
    print(f"\nSaved IV-C/D clipping summary to: {args.output}")


if __name__ == "__main__":
    main()
