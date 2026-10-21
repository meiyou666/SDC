"""Diagnostic ablations of the BF16 augmented kernel, never a training backend.

Run in the verified project environment. All masks use the same compiled kernel;
its fixed resource footprint and diagnostic sink differ from production. Timing
differences describe controlled ablations, not additive standalone stage times.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from abft.benchmark import compare_outputs, summarize
from abft.native import classic_weights
from abft.profiling import _cutlass_kernel_us
from cutlass_gemm import backend
from gemm_breakdown_native import load_diagnostic


MASKS = {"core": 0, "barrier": 8, "encode": 9, "propagate": 11,
         "visitor_only": 4, "full": 15}
LABELS = {"official": "无保护", "production": "生产 ABFT", "core": "诊断主计算",
          "barrier": "+ 同步", "encode": "+ WA 编码", "propagate": "+ qB 传播",
          "full": "+ 输出校验", "visitor_only": "主计算 + 输出校验"}
PARENTS = {"production": "official", "core": "official", "barrier": "core",
           "encode": "barrier", "propagate": "encode", "full": "propagate",
           "visitor_only": "core"}
RESOURCE_NAMES = ("registers_per_thread", "local_bytes_per_thread", "shared_bytes_per_cta",
                  "max_threads_per_cta", "active_blocks_per_sm", "sm_count")


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fields = {"device", "shapes", "seeds", "input_std", "warmup", "repeats", "output",
              "ncu_variant", "ncu_shape_index", "diagnostic_seed"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError(f"Configuration requires exactly: {sorted(fields)}")
    cfg = SimpleNamespace(**raw)
    if not isinstance(cfg.device, str) or not cfg.device.startswith("cuda:") or not cfg.device[5:].isdigit():
        raise ValueError("device must be cuda:<index>")
    if not isinstance(cfg.shapes, list) or not cfg.shapes:
        raise ValueError("shapes must be a nonempty list")
    for shape in cfg.shapes:
        if not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("Each shape must be positive [M,N,K]")
        if shape[1] % 8 or shape[2] % 8:
            raise ValueError("N and K must be multiples of 8")
    if not isinstance(cfg.seeds, list) or not cfg.seeds or any(type(v) is not int or not 0 <= v < 2**63 for v in cfg.seeds):
        raise ValueError("seeds must contain integers in [0,2**63)")
    for name, minimum in (("warmup", 1), ("repeats", 2)):
        if type(raw[name]) is not int or raw[name] < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    if type(cfg.input_std) not in {int, float} or not math.isfinite(cfg.input_std) or cfg.input_std <= 0:
        raise ValueError("input_std must be finite and positive")
    if not isinstance(cfg.output, str) or not cfg.output:
        raise ValueError("output must be a nonempty path")
    if raw["ncu_variant"] not in {"official", "production", *MASKS}:
        raise ValueError("Unknown ncu_variant")
    index = raw["ncu_shape_index"]
    if type(index) is not int or not 0 <= index < len(cfg.shapes):
        raise ValueError("Invalid ncu_shape_index")
    seed = raw["diagnostic_seed"]
    if type(seed) not in {int, float} or not 0 < seed <= 1:
        raise ValueError("diagnostic_seed must be in (0,1]")
    return cfg


def make_calls(cfg, shape, seed, diagnostic):
    device = torch.device(cfg.device)
    m, n, k = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    a = (torch.randn(m, k, generator=generator, device=device) * cfg.input_std).bfloat16()
    b = (torch.randn(k, n, generator=generator, device=device) * cfg.input_std).bfloat16()
    weights = classic_weights(m, device)
    calls = {
        "official": lambda: torch.ops.sdc_cutlass.mm(a, b),
        "production": lambda: torch.ops.sdc_cutlass.abft_augmented_compact(a, b, weights, -1, -1, -1),
    }
    if diagnostic is not None:
        for name, mask in MASKS.items():
            calls[name] = lambda mask=mask: diagnostic.run(a, b, weights, mask, cfg.diagnostic_seed)
    return a, calls


def same_bits(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def validate(calls):
    baseline = calls["official"]()
    production = calls["production"]()
    full = calls["full"]()
    evidence = {"output": same_bits(full[0], production[0]),
                "partials": same_bits(full[1], production[1]),
                "flags": same_bits(full[2], production[2])}
    if not all(evidence.values()):
        raise RuntimeError(f"Diagnostic full path differs from production: {evidence}")
    outputs = {}
    for name, call in calls.items():
        value = call()
        output = value if isinstance(value, torch.Tensor) else value[0]
        outputs[name] = compare_outputs(baseline, output)
        if not outputs[name]["output_bits_equal"]:
            raise RuntimeError(f"{name} changed the data GEMM output")
    return {"full_vs_production": evidence, "data_outputs": outputs}


def profile(calls, cfg):
    names = list(calls)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        for _ in range(cfg.warmup):
            for call in calls.values():
                call()
        torch.cuda.synchronize()
        for repeat in range(cfg.repeats):
            for name in names if repeat % 2 == 0 else names[::-1]:
                with torch.profiler.record_function("breakdown." + name):
                    value = calls[name]()
        torch.cuda.synchronize()
    del value
    samples = {name: [] for name in names}
    for event in profiler.events():
        if event.device_type.name != "CPU" or not event.name.startswith("breakdown."):
            continue
        name = event.name.removeprefix("breakdown.")
        samples[name].append(_cutlass_kernel_us(event))
    if any(len(values) != cfg.repeats for values in samples.values()):
        raise RuntimeError("Missing CUPTI samples; no timing result is reported")
    return {name: summarize(values) for name, values in samples.items()}


def run(cfg, *, ncu):
    device = torch.device(cfg.device)
    print("加载生产 CUTLASS 扩展...", flush=True)
    backend.require_device(device)
    diagnostic = None
    if not ncu or cfg.ncu_variant not in {"official", "production"}:
        print("加载诊断内核（首次需要编译）...", flush=True)
        diagnostic = load_diagnostic()
    report = {"config": vars(cfg),
              "build_key": backend.build_key(), "results": [],
              "timing": "CUPTI correlated CUTLASS kernel time in microseconds; no allocations, "
                        "input conversion, compilation, printing or host launch gaps",
              "interpretation": "All masks share one diagnostic kernel and resource footprint. "
                                "Differences include pipeline interactions and are not standalone stage timings. "
                                "Ablated paths do not provide valid ABFT protection."}
    with torch.cuda.device(device), torch.no_grad():
        shapes = [cfg.shapes[cfg.ncu_shape_index]] if ncu else cfg.shapes
        seeds = cfg.seeds[:1] if ncu else cfg.seeds
        for shape in shapes:
            for seed in seeds:
                a, calls = make_calls(cfg, shape, seed, diagnostic)
                torch.cuda.synchronize(device)
                if ncu:
                    call = calls[cfg.ncu_variant]
                    for _ in range(cfg.warmup):
                        call()
                    torch.cuda.synchronize(device)
                    torch.cuda.nvtx.range_push("sdc_breakdown_target")
                    output = call()
                    torch.cuda.synchronize(device)
                    torch.cuda.nvtx.range_pop()
                    del output
                    print("NCU target completed", flush=True)
                    return
                print(f"BF16 A[{shape[0]},{shape[2]}] × B[{shape[2]},{shape[1]}]：验证并采样...", flush=True)
                validation = validate(calls)
                resources = diagnostic.resources(a)
                if len(resources) != len(RESOURCE_NAMES):
                    raise RuntimeError("Unexpected diagnostic resource ABI")
                times = profile(calls, cfg)
                deltas = {name: times[name]["median"] - times[parent]["median"] for name, parent in PARENTS.items()}
                perturbation = times["full"]["median"] / times["production"]["median"] - 1
                result = {"shape": shape, "seed": seed, "validation": validation,
                          "diagnostic_resources": dict(zip(RESOURCE_NAMES, resources, strict=True)),
                          "kernel_us": times, "delta_vs_parent_us": deltas,
                          "parent": PARENTS, "full_diagnostic_relative_perturbation": perturbation}
                report["results"].append(result)
                print("模式                       内核 μs     对照增量 μs", flush=True)
                for name in ("official", "production", "core", "barrier", "encode", "propagate", "full", "visitor_only"):
                    label = LABELS[name]
                    suffix = "—" if name not in deltas else f"{deltas[name]:+.3f} ({LABELS[PARENTS[name]]})"
                    print(f"{label:<18} {times[name]['median']:>9.3f}     {suffix}", flush=True)
                print(f"诊断版相对生产版：{perturbation:+.1%}；增量不可当作独立组件耗时。", flush=True)
                output = Path(cfg.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                report["diagnostic_sources"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in Path(__file__).parent.glob("gemm_breakdown*") if p.is_file()}
                output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"完整数据：{cfg.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ncu", action="store_true", help="Launch the YAML-selected kernel inside a single NVTX range")
    arguments = parser.parse_args()
    run(load_config(arguments.config), ncu=arguments.ncu)
