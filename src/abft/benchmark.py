"""Paired CUDA-event benchmark of native GEMM and ABFT on identical prepared operands.

Run ``python -m abft.benchmark --config configs/gemm/abft_benchmark.yaml``.
Preparation/quantization and diagnostics are outside measurement. Reference
checks, native GEMM, reduction, and the device-side decision are inside it.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import statistics

import yaml

from .config import AbftConfig, parse_abft_config


SCALES = {"fp32": "none", "bf16": "none", "fp8": "tensorwise", "nvfp4": "block16"}


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str
    formats: list[dict]
    shapes: list[list[int]]
    seeds: list[int]
    input_std: float
    abft: AbftConfig
    warmup: int
    repeats: int
    profile_repeats: int
    output: str


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(BenchmarkConfig.__dataclass_fields__):
        raise ValueError("Benchmark YAML must declare exactly all BenchmarkConfig fields")
    raw["abft"] = parse_abft_config(raw["abft"])
    cfg = BenchmarkConfig(**raw)
    if not isinstance(cfg.device, str) or not cfg.device.startswith("cuda:") or not cfg.device[5:].isdigit():
        raise ValueError("device must explicitly select cuda:<index>; CPU timing is unsupported")
    if cfg.abft.method == "none" or cfg.abft.on_error != "report" or cfg.abft.recovery != "none":
        raise ValueError("Normal benchmark requires an ABFT method, on_error=report, recovery=none")
    if getattr(cfg.abft, "timing", False):
        raise ValueError("Use profile_repeats for separate stage timing; abft.timing must be false")
    for name, minimum in (("warmup", 1), ("repeats", 2), ("profile_repeats", 0)):
        value = getattr(cfg, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(cfg.input_std, (int, float)) or isinstance(cfg.input_std, bool) or not math.isfinite(cfg.input_std) or cfg.input_std <= 0:
        raise ValueError("input_std must be finite and positive")
    if not isinstance(cfg.output, str) or not cfg.output.strip():
        raise ValueError("output must be a nonempty path")
    if not isinstance(cfg.shapes, list) or not cfg.shapes:
        raise ValueError("shapes must be a nonempty list of [M,N,K]")
    for shape in cfg.shapes:
        if not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("Each shape must contain positive integer M,N,K")
    if not isinstance(cfg.seeds, list) or not cfg.seeds or any(type(v) is not int or not 0 <= v < 2**63 for v in cfg.seeds):
        raise ValueError("seeds must be nonempty integers in [0, 2**63)")
    if not isinstance(cfg.formats, list) or not cfg.formats:
        raise ValueError("formats must be a nonempty list")
    for fmt in cfg.formats:
        if not isinstance(fmt, dict) or set(fmt) != {"precision", "scaling", "global_scale"}:
            raise ValueError("Each format declares precision, scaling, global_scale")
        if fmt["precision"] not in SCALES or fmt["scaling"] != SCALES[fmt["precision"]]:
            raise ValueError("Incompatible precision/scaling")
        scale = fmt["global_scale"]
        if scale != "amax" and (isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
            raise ValueError("global_scale must be amax or finite positive")
        if fmt["precision"] == "nvfp4" and any(shape[2] % 32 for shape in cfg.shapes):
            raise ValueError("Native NVFP4 requires K divisible by 32")
    return cfg


def summarize(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Timing samples must be finite and positive")
    return {"median": statistics.median(values), "mean": statistics.mean(values),
            "min": min(values), "max": max(values), "samples": values}


def compare_outputs(native, output):
    """Synchronizing numerical/bitwise diagnostics, outside all timed regions."""
    import torch

    finite = torch.isfinite(native) & torch.isfinite(output)
    delta = (output.float() - native.float()).abs()
    bits = torch.int16 if native.dtype == torch.bfloat16 else torch.int32
    maximum = float(delta[finite].max().item()) if bool(finite.any().item()) else None
    return {"output_equal": torch.equal(native, output),
            "output_bits_equal": native.dtype == output.dtype and torch.equal(
                native.contiguous().view(bits), output.contiguous().view(bits)),
            "unequal_elements": int((native != output).sum().item()),
            "nonfinite_elements": int((~finite).sum().item()),
            "max_abs_finite_difference": maximum if maximum is None or math.isfinite(maximum) else None,
            "difference_overflow": maximum is not None and not math.isfinite(maximum)}


def measure_pairs(baseline, protected, *, warmup, repeats, measure, observe):
    """Scheduling seam for CPU tests; only run() supplies native GPU execution.

    measure(fn) returns (value, elapsed_ms); observe runs after both measurements.
    Alternate order, including warmup. Retain no device outputs between pairs.
    """
    functions = {"baseline": baseline, "protected": protected}
    for index in range(warmup):
        for name in (("baseline", "protected") if index % 2 == 0 else ("protected", "baseline")):
            measure(functions[name])
    times = {"baseline": [], "protected": []}
    observations = []
    for index in range(repeats):
        outputs = {}
        for name in (("baseline", "protected") if index % 2 == 0 else ("protected", "baseline")):
            outputs[name], elapsed = measure(functions[name])
            times[name].append(elapsed)
        observations.append(observe(outputs["baseline"], outputs["protected"]))
    baseline_stats, protected_stats = summarize(times["baseline"]), summarize(times["protected"])
    ratios = [p / b for b, p in zip(times["baseline"], times["protected"], strict=True)]
    return {"baseline_ms": baseline_stats, "protected_ms": protected_stats,
            "paired_ratio": summarize(ratios), "observations": observations}


def run(cfg):
    """Execute only on the repository's configured native CUDA architecture."""
    from dataclasses import replace

    import torch

    from cutlass_gemm import backend

    from .engine import execute_prepared
    from .operands import PreparedGemm

    device = torch.device(cfg.device)
    backend.require_device(device)
    results = []
    with torch.cuda.device(device), torch.no_grad():
        stream = torch.cuda.current_stream(device)

        def measure(fn):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            value = fn()
            end.record(stream)
            end.synchronize()
            return value, start.elapsed_time(end)

        def observe(native, protected):
            output, report = protected
            record = report.to_record()
            return {"detected": bool(record["detected"]), "report": record,
                    **compare_outputs(native, output)}

        for fmt in cfg.formats:
            for m, n, k in cfg.shapes:
                for seed in cfg.seeds:
                    generator = torch.Generator(device=device).manual_seed(seed)
                    a = torch.randn((m, k), device=device, dtype=torch.float32, generator=generator) * cfg.input_std
                    b = torch.randn((k, n), device=device, dtype=torch.float32, generator=generator) * cfg.input_std
                    prepared = PreparedGemm.prepare(a, b, **fmt)
                    # Includes all preparation work before the first timed event.
                    torch.cuda.synchronize(device)
                    def protected():
                        return execute_prepared(prepared, abft_config=cfg.abft,
                                                context={"execution_kind": "benchmark", "retry": False})
                    measured = measure_pairs(prepared.raw_native, protected, warmup=cfg.warmup,
                                             repeats=cfg.repeats, measure=measure, observe=observe)
                    observations = measured.pop("observations")
                    alarms = sum(item["detected"] for item in observations)
                    measured.update(format=fmt, shape=[m, n, k], seed=seed,
                                    normal_calls=cfg.repeats, normal_alarms=alarms,
                                    observed_normal_alarm_rate=alarms / cfg.repeats,
                                    observations=observations)
                    if cfg.profile_repeats:
                        profile = []
                        for _ in range(cfg.profile_repeats):
                            _, report = execute_prepared(prepared, abft_config=replace(cfg.abft, timing=True))
                            profile.append(report.to_record())
                        measured["separate_stage_profile"] = profile
                    results.append(measured)
    return {"config": asdict(cfg), "timing_scope": "prepared operands to native output and ABFT device decision; excludes quantization, logging and optional stage profiling",
            "alarm_denominator": "repeated normal calls per fixed operand pair; not independent production samples",
            "environment": {"device": str(device), "name": torch.cuda.get_device_name(device),
                            "capability": list(torch.cuda.get_device_capability(device)),
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "python": platform.python_version(), "build_key": backend.build_key(),
                            "gemm_settings": backend.settings()}, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
