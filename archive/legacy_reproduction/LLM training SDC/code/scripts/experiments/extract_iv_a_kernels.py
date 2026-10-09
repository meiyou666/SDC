#!/usr/bin/env python3
"""Extract IV-A forward/backward kernel manifests from verbose NVBit logs."""

import argparse
import csv
import re
import random
from collections import OrderedDict
from pathlib import Path


LAUNCH_RE = re.compile(r"\[NVBit\] Launch function ID ([-0-9]+), .* instrumentation enabled: (.*)$")
INSPECT_RE = re.compile(r"\[NVBit Fault Injector\] inspecting (.*) - num instrs: .* - count: ([0-9]+)")
INSTR_RE = re.compile(r"\[NVBit Fault Injector\].*idx:\s*([0-9]+);\s*func:\s*([0-9]+)")


def infer_location(path: Path) -> str:
    name = path.name.lower()
    if "forward" in name:
        return "forward"
    if "backward" in name:
        return "backward"
    raise ValueError(f"Cannot infer location from log name: {path}")


def kernel_label(location: str, index: int) -> str:
    prefix = "FP" if location == "forward" else "BP"
    return f"{prefix}{index}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for log_path in args.log:
        location = infer_location(log_path)
        kernels = OrderedDict()
        inspected_kernels = OrderedDict()
        instrs_by_kernel = {}
        inspect_name_by_function_id = {}
        current_inspect_kernel = None
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\n")
                match = INSPECT_RE.search(line)
                if match:
                    kernel_name = match.group(1).strip()
                    function_id = match.group(2)
                    current_inspect_kernel = kernel_name
                    if kernel_name not in inspected_kernels:
                        inspected_kernels[kernel_name] = {"function_id": function_id}
                    inspect_name_by_function_id[function_id] = kernel_name
                    instrs_by_kernel.setdefault(kernel_name, [])
                    continue

                match = INSTR_RE.search(line)
                if match:
                    instr_idx = int(match.group(1))
                    func_id = match.group(2)
                    kernel_name = current_inspect_kernel
                    if kernel_name is None or inspected_kernels.get(kernel_name, {}).get("function_id") != func_id:
                        kernel_name = inspect_name_by_function_id.get(func_id)
                    if kernel_name is not None:
                        instrs_by_kernel.setdefault(kernel_name, []).append(instr_idx)
                    continue

                match = LAUNCH_RE.search(line)
                if match:
                    function_id = match.group(1)
                    kernel_name = match.group(2).strip()
                    if kernel_name not in kernels:
                        kernels[kernel_name] = {"function_id": function_id, "launch_count": 0}
                    kernels[kernel_name]["launch_count"] += 1

        for index, (kernel_name, info) in enumerate(kernels.items(), 1):
            hmma_instr_indexes = sorted(set(instrs_by_kernel.get(kernel_name, [])))
            selection_rng = random.Random(f"{args.seed}:{location}:{index}:{kernel_name}")
            target_instr = selection_rng.choice(hmma_instr_indexes) if hmma_instr_indexes else ""
            rows.append({
                "location": location,
                "kernel_label": kernel_label(location, index),
                "function_id_first_seen": info["function_id"],
                "launch_count": info["launch_count"],
                "hmma_instr_count": len(hmma_instr_indexes),
                "hmma_instr_indexes": ",".join(str(value) for value in hmma_instr_indexes),
                "target_instr": target_instr,
                "kernel_name": kernel_name,
                "target_func_contains": kernel_name,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "location",
                "kernel_label",
                "function_id_first_seen",
                "launch_count",
                "hmma_instr_count",
                "hmma_instr_indexes",
                "target_instr",
                "kernel_name",
                "target_func_contains",
            ),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["location"]] = counts.get(row["location"], 0) + 1
    print(f"Saved kernel manifest to: {args.output}")
    for location in sorted(counts):
        print(f"{location}: {counts[location]} kernel(s)")
    missing_instrs = [row for row in rows if not row["target_instr"]]
    if missing_instrs:
        print("WARNING: some launched kernels had no parsed HMMA instruction indexes:")
        for row in missing_instrs:
            print(f'  {row["kernel_label"]}: {row["kernel_name"]}')


if __name__ == "__main__":
    main()
