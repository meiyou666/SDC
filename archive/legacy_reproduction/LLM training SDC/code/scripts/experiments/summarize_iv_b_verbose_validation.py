#!/usr/bin/env python3
"""Summarize verbose NVBit evidence for the IV-B BP1/BP2 validation runs."""

import argparse
import hashlib
import re
from pathlib import Path


INSPECT_RE = re.compile(r"\[NVBit Fault Injector\] inspecting (.*) - num instrs: .* - count: ([0-9]+)")
INSTR_RE = re.compile(r"\[NVBit Fault Injector\].*idx:\s*([0-9]+);\s*func:\s*([0-9]+)")
TARGET_RE = re.compile(r"TARGET_FUNC_CONTAINS = (.*) - Instrument functions containing this substring")


def parse_log(path: Path):
    target_filter = ""
    kernels = {}
    current_kernel = None
    function_names = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            target_match = TARGET_RE.search(line)
            if target_match and not target_filter:
                target_filter = target_match.group(1).strip()
                continue

            inspect_match = INSPECT_RE.search(line)
            if inspect_match:
                kernel_name = inspect_match.group(1).strip()
                function_id = inspect_match.group(2)
                current_kernel = kernel_name
                function_names[function_id] = kernel_name
                kernels.setdefault(kernel_name, set())
                continue

            instr_match = INSTR_RE.search(line)
            if instr_match:
                instr_idx = int(instr_match.group(1))
                function_id = instr_match.group(2)
                kernel_name = function_names.get(function_id, current_kernel)
                if kernel_name is not None:
                    kernels.setdefault(kernel_name, set()).add(instr_idx)

    return target_filter, kernels


def fingerprint(indexes):
    payload = ",".join(str(value) for value in sorted(indexes)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for log_path in args.log:
        target_filter, kernels = parse_log(log_path)
        if not kernels:
            rows.append((log_path.stem, target_filter, "NO_VERBOSE_KERNELS", 0, "", ""))
            continue
        for kernel_name, indexes in sorted(kernels.items()):
            rows.append((
                log_path.stem,
                target_filter,
                kernel_name,
                len(indexes),
                fingerprint(indexes),
                ",".join(str(value) for value in sorted(indexes)),
            ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("run_label\ttarget_func_contains\tinspected_kernel\thmma_idx_count\thmma_idx_sha16\thmma_idx_indexes\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")

    print(f"Saved verbose validation summary to: {args.output}")
    for row in rows:
        print(f"{row[0]}: idx_count={row[3]} sha16={row[4]} kernel={row[2]}")


if __name__ == "__main__":
    main()
