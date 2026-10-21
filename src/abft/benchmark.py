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
    progress: bool = False
    preview_size: int = 0


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fields = set(BenchmarkConfig.__dataclass_fields__)
    optional = {"progress", "preview_size"}
    if not isinstance(raw, dict) or set(raw) - fields or fields - optional - set(raw):
        raise ValueError("Benchmark YAML contains unknown fields or is missing required fields")
    raw["abft"] = parse_abft_config(raw["abft"])
    cfg = BenchmarkConfig(**raw)
    if type(cfg.progress) is not bool:
        raise ValueError("progress must be a boolean")
    if type(cfg.preview_size) is not int or not 0 <= cfg.preview_size <= 16:
        raise ValueError("preview_size must be an integer from 0 to 16 (0 disables previews)")
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


def measure_pairs(baseline, protected, *, warmup, repeats, measure, observe, on_pair=None):
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
        observation = observe(outputs["baseline"], outputs["protected"])
        observations.append(observation)
        if on_pair is not None:
            on_pair(index + 1, times["baseline"][-1], times["protected"][-1], observation)
    baseline_stats, protected_stats = summarize(times["baseline"]), summarize(times["protected"])
    ratios = [p / b for b, p in zip(times["baseline"], times["protected"], strict=True)]
    return {"baseline_ms": baseline_stats, "protected_ms": protected_stats,
            "paired_ratio": summarize(ratios), "observations": observations}


def show_matrix(name, tensor, size):
    """Copy only a bounded preview to the host, outside timed operations."""
    if not size:
        return
    values = tensor[:size, :size].detach().float().cpu().tolist()
    print(f"[矩阵] {name} shape={tuple(tensor.shape)} dtype={tensor.dtype}，左上角：", flush=True)
    for row in values:
        print("  " + " ".join(f"{value: .6g}" for value in row), flush=True)


def run(cfg):
    """Execute only on the repository's configured native CUDA architecture."""
    import torch

    from cutlass_gemm import backend

    from .engine import execute_prepared
    from .operands import PreparedGemm

    def log(message):
        if cfg.progress:
            print(f"[GEMM] {message}", flush=True)

    device = torch.device(cfg.device)
    log(f"检查设备并加载 CUTLASS 扩展：{device}；首次运行可能需要编译")
    backend.require_device(device)
    total = len(cfg.formats) * len(cfg.shapes) * len(cfg.seeds)
    log(f"设备 {torch.cuda.get_device_name(device)}；共 {total} 组；量化与调试输出不计入耗时")
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
            nonlocal preview_pending, preview_outputs
            output, report = protected
            record = report.to_record()
            if preview_pending:
                preview_outputs = (native, output)
                preview_pending = False
            return {"detected": bool(record["detected"]), "report": record,
                    **compare_outputs(native, output)}

        def log_pair(index, baseline_ms, protected_ms, observation):
            log(f"测量 {index}/{cfg.repeats}：普通={baseline_ms:.6f} ms，"
                f"ABFT={protected_ms:.6f} ms，报警={observation['detected']}，"
                f"逐位一致={observation['output_bits_equal']}，"
                f"最大有限绝对差={observation['max_abs_finite_difference']}，"
                f"非有限元素={observation['nonfinite_elements']}")

        for fmt in cfg.formats:
            for m, n, k in cfg.shapes:
                for seed in cfg.seeds:
                    log(f"组 {len(results) + 1}/{total}：{fmt['precision']}，"
                        f"A[{m},{k}] × B[{k},{n}]，seed={seed}；生成并准备输入")
                    generator = torch.Generator(device=device).manual_seed(seed)
                    a = torch.randn((m, k), device=device, dtype=torch.float32, generator=generator) * cfg.input_std
                    b = torch.randn((k, n), device=device, dtype=torch.float32, generator=generator) * cfg.input_std
                    show_matrix("生成输入 A（类型转换和量化前）", a, cfg.preview_size)
                    show_matrix("生成输入 B（类型转换和量化前）", b, cfg.preview_size)
                    prepared = PreparedGemm.prepare(a, b, **fmt)
                    if fmt["precision"] in {"fp32", "bf16"}:
                        show_matrix("实际 GEMM 输入 A", prepared.a, cfg.preview_size)
                        show_matrix("实际 GEMM 输入 B", prepared.b, cfg.preview_size)
                    # Includes all preparation work before the first timed event.
                    torch.cuda.synchronize(device)
                    def protected():
                        return execute_prepared(prepared, abft_config=cfg.abft,
                                                context={"execution_kind": "benchmark", "retry": False})
                    preview_pending = bool(cfg.preview_size)
                    preview_outputs = None
                    pair_logs = []
                    log(f"输入准备完成；预热 {cfg.warmup} 组，然后测量 {cfg.repeats} 组")
                    measured = measure_pairs(prepared.raw_native, protected, warmup=cfg.warmup,
                                             repeats=cfg.repeats, measure=measure, observe=observe,
                                             on_pair=lambda *entry: pair_logs.append(entry) if cfg.progress else None)
                    observations = measured.pop("observations")
                    alarms = sum(item["detected"] for item in observations)
                    measured.update(format=fmt, shape=[m, n, k], seed=seed,
                                    normal_calls=cfg.repeats, normal_alarms=alarms,
                                    observed_normal_alarm_rate=alarms / cfg.repeats,
                                    observations=observations)
                    if cfg.profile_repeats:
                        from .profiling import print_profile, profile_prepared

                        log(f"单独测量各阶段：{cfg.profile_repeats} 次")
                        measured["stage_profile"] = profile_prepared(
                            prepared, cfg.abft, warmup=cfg.warmup, repeats=cfg.profile_repeats,
                        )
                    # All printing/readback of previews follows both measurement passes.
                    if preview_outputs is not None:
                        show_matrix("普通 GEMM 输出 C", preview_outputs[0], cfg.preview_size)
                        show_matrix("ABFT GEMM 输出 C", preview_outputs[1], cfg.preview_size)
                    for entry in pair_logs:
                        log_pair(*entry)
                    if cfg.profile_repeats:
                        print_profile(measured["stage_profile"])
                    log(f"本组完成：普通中位耗时={measured['baseline_ms']['median']:.6f} ms，"
                        f"ABFT中位耗时={measured['protected_ms']['median']:.6f} ms，"
                        f"配对耗时比例中位数={measured['paired_ratio']['median']:.3f}，"
                        f"正常调用报警={alarms}/{cfg.repeats}")
                    results.append(measured)
    return {"config": asdict(cfg), "timing_scope": "CUDA event interval around full calls, including possible host launch gaps; excludes quantization, logging, compilation, warmup and separate CUPTI profiling",
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
    print(f"[GEMM] 完成，结果已保存：{output}", flush=True)


if __name__ == "__main__":
    main()
