"""Standalone evaluation; dataset/model training stays in gemm.train."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from algorithms import Algorithm

from .evaluate import evaluate_case
from .types import PROFILES, ExecutionContext


def main():
    parser = argparse.ArgumentParser(description="统一 GEMM 逐 bit 评测")
    parser.add_argument("--input", required=True, help="ProtectedLinear 采集的 .pt 操作数")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="mxfp8")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    from backends.nvidia.runtime import require_device

    require_device(torch.device("cuda"))
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    sample = torch.load(args.input, map_location="cpu", weights_only=True)
    a, b = sample["a"].cuda(), sample["b"].cuda()
    context = ExecutionContext(**sample["context"])
    method = Algorithm("classic", args.atol, args.rtol)
    options = dict(profile=args.profile, algorithm=method, phase=sample["phase"], context=context)
    with (directory / "events.jsonl").open("w") as stream:
        summary = evaluate_case(a, b, **options, writer=lambda r: stream.write(json.dumps(r, allow_nan=False)+"\n"))
    if args.timing:
        from .benchmark import benchmark

        summary["timing"] = benchmark(a, b, **options)
    if args.diagnostics:
        from .diagnostics import diagnose

        summary["numerical"] = diagnose(a, b, profile=args.profile, phase=sample["phase"], context=context)
    from backends.nvidia.build import build_key

    summary["settings"] = {"profile": args.profile, "algorithm": asdict(method), "context": asdict(context),
                           "phase": sample["phase"], "build": build_key(), "device": torch.cuda.get_device_name()}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(directory)


if __name__ == "__main__":
    main()
