#!/usr/bin/env python3
"""Validate the NVBit fast-path and injection-regression smoke tests."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def core_rows(records):
    return [record for record in records if "loss" in record]


def mean_step_seconds(records):
    rows = core_rows(records)
    if len(rows) < 2:
        raise ValueError("At least two training steps are required for timing")
    deltas = [
        float(current["_time"]) - float(previous["_time"])
        for previous, current in zip(rows, rows[1:])
    ]
    return math.fsum(deltas) / len(deltas)


def step_deltas(records):
    rows = core_rows(records)
    return {
        int(current["_step"]): float(current["_time"]) - float(previous["_time"])
        for previous, current in zip(rows, rows[1:])
    }


def merged_steps(records, expected_steps):
    merged = {step: {} for step in range(1, expected_steps + 1)}
    for record in records:
        step = int(record.get("_step", -1))
        if step not in merged:
            continue
        for key, value in record.items():
            if key != "_time":
                merged[step][key] = value
    return merged


def trigger_steps(records):
    return [
        int(record["_step"])
        for record in core_rows(records)
        if float(record.get("trigger_nvbit", 0.0)) == 1.0
    ]


def require_checkpoint(run_dir, step):
    checkpoint = run_dir / f"model_{step}"
    required = (
        checkpoint / "model.safetensors",
        checkpoint / "optimizer.pt",
        checkpoint / "training_state.json",
        run_dir / "metrics.jsonl",
        run_dir / "summary.json",
        run_dir / "run_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing test artifacts:\n  " + "\n  ".join(missing))
    return checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_pretrigger(args):
    require_checkpoint(args.baseline_dir, args.expected_steps)
    require_checkpoint(args.fi_dir, args.expected_steps)
    baseline = load_jsonl(args.baseline_dir / "metrics.jsonl")
    fi = load_jsonl(args.fi_dir / "metrics.jsonl")

    baseline_steps = merged_steps(baseline, args.expected_steps)
    fi_steps = merged_steps(fi, args.expected_steps)
    mismatches = []
    compared = 0
    for step in range(1, args.expected_steps + 1):
        keys = sorted(set(baseline_steps[step]) | set(fi_steps[step]))
        for key in keys:
            compared += 1
            left = baseline_steps[step].get(key, "<missing>")
            right = fi_steps[step].get(key, "<missing>")
            if left != right:
                mismatches.append((step, key, left, right))

    baseline_time = mean_step_seconds(baseline)
    fi_time = mean_step_seconds(fi)
    slowdown = fi_time / baseline_time
    fi_triggers = trigger_steps(fi)

    print("NVBit pre-trigger fast-path test")
    print(f"  steps compared       : {args.expected_steps}")
    print(f"  metric values        : {compared}")
    print(f"  metric mismatches    : {len(mismatches)}")
    print(f"  FI trigger steps     : {fi_triggers}")
    print(f"  baseline mean step   : {baseline_time:.4f} s")
    print(f"  FI mean step         : {fi_time:.4f} s")
    print(f"  pre-trigger slowdown : {slowdown:.3f}x")
    print(f"  allowed slowdown     : {args.max_slowdown:.3f}x")

    failed = False
    if mismatches:
        failed = True
        print("  first mismatches:")
        for step, key, left, right in mismatches[:10]:
            print(f"    step={step} metric={key}: baseline={left!r}, FI={right!r}")
    if fi_triggers:
        failed = True
        print("  ERROR: injection unexpectedly triggered during the pre-trigger test")
    if slowdown > args.max_slowdown:
        failed = True
        print("  ERROR: pre-trigger slowdown exceeds the configured limit")

    return 1 if failed else 0


def analyze_injection(args):
    control_checkpoint = require_checkpoint(args.control_dir, args.expected_steps)
    fi_checkpoint = require_checkpoint(args.fi_dir, args.expected_steps)
    control = load_jsonl(args.control_dir / "metrics.jsonl")
    fi = load_jsonl(args.fi_dir / "metrics.jsonl")
    control_triggers = trigger_steps(control)
    actual_triggers = trigger_steps(fi)

    control_steps = merged_steps(control, args.trigger_step - 1)
    fi_steps = merged_steps(fi, args.trigger_step - 1)
    pretrigger_mismatches = []
    for step in range(1, args.trigger_step):
        keys = sorted(set(control_steps[step]) | set(fi_steps[step]))
        for key in keys:
            left = control_steps[step].get(key, "<missing>")
            right = fi_steps[step].get(key, "<missing>")
            if left != right:
                pretrigger_mismatches.append((step, key, left, right))

    control_hash = sha256(control_checkpoint / "model.safetensors")
    fi_hash = sha256(fi_checkpoint / "model.safetensors")
    control_time = mean_step_seconds(control)
    fi_deltas = step_deltas(fi)
    injection_time = fi_deltas.get(args.trigger_step)
    post_times = [
        elapsed for step, elapsed in fi_deltas.items()
        if step > args.trigger_step
    ]
    post_time = math.fsum(post_times) / len(post_times) if post_times else None
    post_slowdown = post_time / control_time if post_time is not None else None

    print("NVBit injection regression test")
    print(f"  control triggers     : {control_triggers}")
    print(f"  expected trigger     : [{args.trigger_step}]")
    print(f"  actual trigger       : {actual_triggers}")
    print(f"  pre-trigger mismatch : {len(pretrigger_mismatches)}")
    print(f"  control mean step    : {control_time:.4f} s")
    print(
        "  injected Step time  : "
        + (f"{injection_time:.4f} s" if injection_time is not None else "unavailable")
    )
    print(
        "  post-trigger mean   : "
        + (f"{post_time:.4f} s" if post_time is not None else "unavailable")
    )
    print(
        "  post-trigger slowdown: "
        + (f"{post_slowdown:.3f}x" if post_slowdown is not None else "unavailable")
    )
    print(f"  control model SHA256 : {control_hash}")
    print(f"  injected model SHA256: {fi_hash}")
    print(f"  checkpoints differ   : {control_hash != fi_hash}")

    failed = False
    if control_triggers:
        failed = True
        print("  ERROR: the non-NVBit control unexpectedly contains trigger markers")
    if actual_triggers != [args.trigger_step]:
        failed = True
        print("  ERROR: the recorded injection schedule is incorrect")
    if pretrigger_mismatches:
        failed = True
        print("  ERROR: control and FI trajectories differ before injection")
        for step, key, left, right in pretrigger_mismatches[:10]:
            print(f"    step={step} metric={key}: control={left!r}, FI={right!r}")
    if post_slowdown is None or post_slowdown > args.max_posttrigger_slowdown:
        failed = True
        print(
            "  ERROR: post-trigger slowdown is unavailable or exceeds "
            f"{args.max_posttrigger_slowdown:.3f}x"
        )
    if control_hash == fi_hash:
        failed = True
        print("  ERROR: injection did not change the final model checkpoint")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    pretrigger = subparsers.add_parser("pretrigger")
    pretrigger.add_argument("--baseline-dir", type=Path, required=True)
    pretrigger.add_argument("--fi-dir", type=Path, required=True)
    pretrigger.add_argument("--expected-steps", type=int, default=30)
    pretrigger.add_argument("--max-slowdown", type=float, default=3.0)
    pretrigger.set_defaults(handler=analyze_pretrigger)

    injection = subparsers.add_parser("injection")
    injection.add_argument("--control-dir", type=Path, required=True)
    injection.add_argument("--fi-dir", type=Path, required=True)
    injection.add_argument("--expected-steps", type=int, default=15)
    injection.add_argument("--trigger-step", type=int, default=12)
    injection.add_argument("--max-posttrigger-slowdown", type=float, default=3.0)
    injection.set_defaults(handler=analyze_injection)

    args = parser.parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
