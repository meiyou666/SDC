"""Separate CUPTI profiling of the existing ABFT execution path.

GPU activity sums exclude host launch gaps. CPU scopes include submission,
waiting, and profiler overhead; they are not pure CPU computation timings.
"""
import math
import statistics


STAGES = {
    "baseline.gemm": "无保护 GEMM",
    "abft.reference": "校验权重/参考准备",
    "abft.execute": "融合 GEMM 调用（含校验和缓冲初始化）",
    "abft.finalize": "单核收尾：全列合并、阈值、非有限值和状态",
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


def _cutlass_kernel_us(event):
    """Only correlated CUTLASS kernel activities, not their enclosing CPU scope."""
    own = sum(kernel.duration for kernel in getattr(event, "kernels", ())
              if "cutlass::Kernel<" in kernel.name or "cutlass::device_kernel<" in kernel.name
              or kernel.name.startswith(("_ZN7cutlass6KernelI", "_ZN7cutlass13device_kernelI")))
    return own + sum(_cutlass_kernel_us(child) for child in getattr(event, "cpu_children", ()))


def summarize_profile(events, repeats, *, input_fused, native=False):
    """Use CPU scope owners to avoid counting duplicate CUDA annotations."""
    scopes = {name: [] for name in STAGES}
    for event in events:
        if event.name in scopes and event.device_type.name == "CPU":
            scopes[event.name].append(event)
    required = {"baseline.gemm", "abft.reference", "abft.execute", "abft.total"}
    if native:
        required.add("abft.finalize")
    else:
        required.update(("abft.merge_observed", "abft.verify", "abft.report_flags"))
    if input_fused and not native:
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
        row = {"stage": name, "label": label,
               "gpu_active_ms": _statistics([e.device_time_total / 1000 for e in samples]),
               "cpu_scope_ms": _statistics([e.cpu_time_total / 1000 for e in samples])}
        if name in {"baseline.gemm", "abft.execute"}:
            kernel_times = [_cutlass_kernel_us(e) for e in samples]
            if all(value > 0 for value in kernel_times):
                row["cutlass_kernel_ms"] = _statistics([value / 1000 for value in kernel_times])
                row["other_gpu_activity_ms"] = _statistics([
                    max(0.0, e.device_time_total - value) / 1000
                    for e, value in zip(samples, kernel_times, strict=True)
                ])
        rows.append(row)
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
    result = summarize_profile(profiler.events(), repeats,
                               input_fused=config.reference_mode in {"input_fused", "augmented"},
                               native=config.implementation == "native")
    if config.reference_mode == "augmented":
        for row in result["stages"]:
            if row["stage"] == "abft.execute":
                row["label"] = ("BF16 增广 GEMM（无预清零）" if config.implementation == "native"
                                else "BF16 编码与增广 GEMM 调用（含缓冲初始化）")
            elif row["stage"] == "abft.locate":
                row["label"] = "定位状态（低精度编码路径未启用定位）"
        result["fused_scope"] = ("BF16 augmented GEMM plus one finalize kernel; no pre-zeroing, output reread or separate projection kernels"
                                 if config.implementation == "native" else
                                 "BF16 tile encoding and augmented GEMM share one kernel; native call additionally initializes buffers; no independent FP32 reference GEMM")
    # Synchronizing diagnostics are deliberately after the profiler has stopped.
    from .benchmark import compare_outputs

    result["last_output_comparison"] = compare_outputs(native, output)
    result["last_report"] = report.to_record()
    return result


def print_profile(result):
    """Print only after all profiling, synchronization and aggregation finish."""
    import unicodedata

    labels = {
        "baseline.gemm": "无保护 GEMM", "abft.reference": "参考准备",
        "abft.execute": "ABFT GEMM", "abft.finalize": "校验合并与判定",
        "abft.merge_reference": "参考校验合并", "abft.merge_observed": "输出校验合并",
        "abft.detect": "检测", "abft.locate": "定位状态", "abft.verify": "检测与定位",
        "abft.report_flags": "报告状态", "abft.total": "ABFT 总计",
    }
    if any(row["stage"] == "abft.finalize" for row in result["stages"]):
        labels["abft.reference"] = "权重准备"

    def line(label, milliseconds):
        width = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in label)
        print(f"  {label}{' ' * max(1, 24 - width)}{milliseconds * 1000:>10.3f}", flush=True)

    print("  阶段                    GPU 中位耗时 / μs", flush=True)
    for row in result["stages"]:
        label = labels[row["stage"]]
        if "cutlass_kernel_ms" in row:
            line(label + " 内核", row["cutlass_kernel_ms"]["median"])
            if row["other_gpu_activity_ms"]["median"] > 0:
                line("  初始化等", row["other_gpu_activity_ms"]["median"])
        else:
            line(label, row["gpu_active_ms"]["median"])
    stages = {row["stage"]: row for row in result["stages"]}
    baseline = stages["baseline.gemm"]["gpu_active_ms"]["median"]
    protected = stages["abft.total"]["gpu_active_ms"]["median"]
    if baseline > 0:
        print(f"  总耗时 / 无保护：{protected / baseline:.2f}×", flush=True)
