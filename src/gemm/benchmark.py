"""Device-only latency from CUDA Graph replay; compilation is warmed up first."""

import statistics

import torch

from algorithms import resolve_algorithm
from backends.nvidia.library import library_gemm
from backends.nvidia.runtime import multiply, pack_operands

from .api import gemm
from .quantization import hadamard_pair
from .types import PROFILES, ExecutionContext


def recovery_latency(output, expected, method, *, repeats=30):
    """Reset C outside the timed graph segment, then detect/correct/recheck."""
    scratch = output.clone()
    for _ in range(3):
        scratch.copy_(output)
        torch.ops.sdc.verify(scratch, expected, method.atol, method.rtol, True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    stop = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch.copy_(output)
        start.record()
        retained = torch.ops.sdc.verify(scratch, expected, method.atol, method.rtol, True)
        stop.record()
    times = []
    for _ in range(repeats):
        graph.replay()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    return {"median_ms": statistics.median(times), "samples_ms": times,
            "status": retained[0].cpu().tolist(), "scope": "output reduction + decision + correction + recheck"}


def device_latency(call, *, warmup=5, repeats=30, replay_count=10):
    if min(warmup, repeats, replay_count) <= 0:
        raise ValueError("计时次数必须为正整数。")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        retained = [call() for _ in range(replay_count)]
    values = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop) / replay_count)
    del retained
    ordered = sorted(values)
    return {"median_ms": statistics.median(values), "p95_ms": ordered[min(len(ordered)-1, int(.95*len(ordered)))],
            "samples_ms": values, "replay_count": replay_count}


def benchmark(a, b, *, profile, algorithm, phase="forward", context=None, repeats=30):
    if a.device.type != "cuda":
        raise ValueError("设备计时只在 NVIDIA 后端执行。")
    context = context or ExecutionContext()
    method = resolve_algorithm(algorithm)
    options = dict(profile=profile, backend="nvidia", phase=phase, context=context)
    base = device_latency(lambda: gemm(a, b, algorithm="none", **options), repeats=repeats)
    protected = device_latency(lambda: gemm(a, b, algorithm=algorithm, **options), repeats=repeats)
    left, right = a.float(), b.float()
    if profile == "nvfp4" and phase == "wgrad":
        left, right = hadamard_pair(left, right, context.quant_seed(phase) ^ 0x48A2)
    pa, pb = pack_operands(left, right, profile, phase, context)
    m, k = left.shape
    n = right.shape[1]
    output = multiply(pa, pb, m, n, k, profile)
    expected = torch.ops.sdc.reference(*pa, *pb, m, n, k, PROFILES[profile].code)
    stage = {
        "pack": lambda: pack_operands(left, right, profile, phase, context),
        "gemm": lambda: multiply(pa, pb, m, n, k, profile),
    }
    recovery = None
    if method.name == "classic":
        stage.update({
            "reference_generation_and_propagation": lambda: torch.ops.sdc.reference(*pa, *pb, m, n, k, PROFILES[profile].code),
            "output_reduction_and_decision": lambda: torch.ops.sdc.verify(output, expected, method.atol, method.rtol, False),
        })
        mask = torch.ops.sdc.mask(m, n, 0, 0, 31, True, a)
        damaged = multiply(pa, pb, m, n, k, profile, mask)
        recovery = recovery_latency(damaged, expected, method, repeats=repeats)
        recovery["same_source_retry_ms"] = protected["median_ms"]
    stages = {name: device_latency(call, repeats=repeats) for name, call in stage.items()}
    library = {}
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        def library_total():
            la, lb = a.float(), b.float()
            if profile == "nvfp4" and phase == "wgrad":
                la, lb = hadamard_pair(la, lb, context.quant_seed(phase) ^ 0x48A2)
            qa, qb = pack_operands(la, lb, profile, phase, context)
            return library_gemm(qa, qb, m, n, k, profile)

        value = library_total()
        library["output_delta_l2"] = float((value.double()-output.double()).norm())
        library["core"] = device_latency(lambda: library_gemm(pa, pb, m, n, k, profile), repeats=repeats)
        library["total"] = device_latency(library_total, repeats=repeats)
        library["protected_overhead_fraction"] = protected["median_ms"] / library["total"]["median_ms"] - 1
    except (RuntimeError, AttributeError, NotImplementedError) as exc:
        library = {"total": None, "reason": str(exc)}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_bf16
    return {"scope": "quantization + GEMM + reference generation/propagation + verification",
            "timer": "CUDA events around CUDA Graph replay", "same_source": base, "protected": protected,
            "stages": stages, "native_library": library,
            "recovery": recovery,
            "overhead_fraction": protected["median_ms"] / base["median_ms"] - 1}
