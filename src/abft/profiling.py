"""Separate CUPTI profiling of the existing ABFT execution path.

GPU activity sums exclude host launch gaps. CPU scopes include submission,
waiting, and profiler overhead; they are not pure CPU computation timings.
"""
import math
import statistics


STAGES = {
    "baseline.gemm": "无保护 GEMM",
    "abft.reference": "校验权重/参考准备",
    "abft.execute": "融合 GEMM（含片内 checksum）",
    "abft.merge_reference": "参考 checksum 与幅值合并",
    "abft.merge_observed": "输出 checksum 合并",
    "abft.detect": "阈值比较与非有限值检测",
    "abft.locate": "候选错误定位",
    "abft.verify": "插件检测与定位合计",
    "abft.report_flags": "报告状态张量",
    "abft.total": "ABFT 完整调用（含以上阶段，不重复相加）",
}


def _statistics(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Profile timings must be finite and nonnegative")
    return {"median": statistics.median(values), "mean": statistics.mean(values),
            "min": min(values), "max": max(values), "samples": values}


def summarize_profile(events, repeats, *, input_fused):
    """Use CPU scope owners to avoid counting duplicate CUDA annotations."""
    scopes = {name: [] for name in STAGES}
    for event in events:
        if event.name in scopes and event.device_type.name == "CPU":
            scopes[event.name].append(event)
    required = {"baseline.gemm", "abft.reference", "abft.execute", "abft.merge_observed",
                "abft.verify", "abft.report_flags", "abft.total"}
    if input_fused:
        required.add("abft.merge_reference")
    for name in required:
        if len(scopes[name]) != repeats:
            raise RuntimeError(f"Incomplete profile: {name}, expected {repeats} samples")
    if not sum(e.device_time_total for e in scopes["baseline.gemm"]):
        raise RuntimeError("CUDA activity timings are unavailable; check PyTorch profiler/CUPTI support")
    detailed_verify = bool(scopes["abft.detect"] and scopes["abft.locate"])
    rows = []
    for name, label in STAGES.items():
        samples = scopes[name]
        if not samples or (name == "abft.verify" and detailed_verify):
            continue
        if len(samples) != repeats:
            raise RuntimeError(f"Incomplete profile: {name}, expected {repeats} samples")
        rows.append({"stage": name, "label": label,
                     "gpu_active_ms": _statistics([e.device_time_total / 1000 for e in samples]),
                     "cpu_scope_ms": _statistics([e.cpu_time_total / 1000 for e in samples])})
    return {"repeats": repeats, "stages": rows,
            "gpu_scope": "sum of correlated GPU kernel/copy activities; excludes CPU launch gaps",
            "cpu_scope": "instrumented CPU scope, including submission, waits and profiler overhead; overlaps GPU work",
            "exclusions": ["compilation", "input preparation/quantization", "warmup", "printing", "report serialization"],
            "fused_scope": "GEMM and in-kernel checksums execute together and are not separately timed"}


def profile_prepared(prepared, config, *, warmup, repeats):
    """Warm up instrumentation, then profile without prints or host readback."""
    import torch

    from .engine import execute_prepared

    record = torch.profiler.record_function
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    torch.cuda.synchronize(prepared.a.device)
    with torch.profiler.profile(activities=activities) as profiler:
        for _ in range(warmup):
            prepared.raw_native()
            execute_prepared(prepared, abft_config=config,
                             profile_scope=lambda name: record("warmup." + name))
        torch.cuda.synchronize(prepared.a.device)
        for index in range(repeats):
            # Alternate order as in the uninstrumented paired measurements.
            order = ("baseline", "protected") if index % 2 == 0 else ("protected", "baseline")
            for name in order:
                if name == "baseline":
                    with record("baseline.gemm"):
                        native = prepared.raw_native()
                else:
                    with record("abft.total"):
                        output, report = execute_prepared(prepared, abft_config=config, profile_scope=record)
        torch.cuda.synchronize(prepared.a.device)
    result = summarize_profile(profiler.events(), repeats, input_fused=config.reference_mode == "input_fused")
    # Synchronizing diagnostics are deliberately after the profiler has stopped.
    from .benchmark import compare_outputs

    result["last_output_comparison"] = compare_outputs(native, output)
    result["last_report"] = report.to_record()
    return result


def print_profile(result):
    """Print only after all profiling, synchronization and aggregation finish."""
    print("[分阶段] 单位 ms；GPU 为活动累计时间，CPU 含提交/等待及 profiler 开销；两列不可相加", flush=True)
    for row in result["stages"]:
        gpu, cpu = row["gpu_active_ms"], row["cpu_scope_ms"]
        print(f"  {row['label']}：GPU 中位={gpu['median']:.6f}，均值={gpu['mean']:.6f}；"
              f"CPU 中位={cpu['median']:.6f}，均值={cpu['mean']:.6f}", flush=True)
    print("[分阶段] 融合内核中的主乘法与 checksum 合并计时；编译、预热、量化和打印均已排除", flush=True)
