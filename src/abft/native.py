"""Compiled ABFT execution plans with device-side merge and decision.

Only explicitly immutable builtin weights are cached. Scratch and report tensors remain
owned by each call, so concurrent streams and retained reports never alias a
mutable global workspace.
"""
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from threading import Lock
import time

import torch

from cutlass_gemm import backend

from .methods import Classic, Sum, get_method


_weights = OrderedDict()
_weights_lock = Lock()
_MAX_WEIGHT_ENTRIES = 32
_MAX_WEIGHT_BYTES = 16 * 1024 * 1024


def classic_weights(m, device):
    """Cache BF16 row weights with explicit producer/consumer stream ordering."""
    device = torch.device(device)
    stream = torch.cuda.current_stream(device)
    key = (stream.device.index, m)
    with _weights_lock:
        entry = _weights.pop(key, None)
        if entry is None:
            rows = torch.arange(1, m + 1, device=device, dtype=torch.float32)
            weights = torch.stack((torch.ones_like(rows), rows)).to(torch.bfloat16)
            ready = torch.cuda.Event()
            ready.record(stream)
            entry = (weights, ready, stream.cuda_stream)
        weights, ready, owner = entry
        if weights.numel() * weights.element_size() <= _MAX_WEIGHT_BYTES:
            _weights[key] = entry
            while (len(_weights) > _MAX_WEIGHT_ENTRIES or
                   sum(item[0].numel() * item[0].element_size() for item in _weights.values()) > _MAX_WEIGHT_BYTES):
                _weights.popitem(last=False)
        if owner != stream.cuda_stream:
            stream.wait_event(ready)
            weights.record_stream(stream)
    return weights


def _builtin_weights(method, prepared, cfg, policy):
    """Cache only builtin Classic/Sum formulas, never plugin/adaptive evidence."""
    custom_preparation = any(name in vars(method) for name in (
        "projection_weights", "prepare_fused", "prepare_augmented",
    ))
    if type(method) not in {Classic, Sum} or method.stateful or custom_preparation:
        prepare = method.prepare_augmented if cfg.reference_mode == "augmented" else method.prepare_fused
        return prepare(prepared, cfg)
    if type(method) is Classic and cfg.reference_mode == "augmented":
        weights = classic_weights(prepared.m, prepared.a.device)
    else:
        device = torch.device(prepared.a.device)
        dtype = torch.bfloat16 if cfg.reference_mode == "augmented" else torch.float32
        stream = torch.cuda.current_stream(device)
        key = ("compiled", policy.algorithm, stream.device.index, prepared.m, dtype)
        with _weights_lock:
            entry = _weights.pop(key, None)
            if entry is None:
                weights = method.projection_weights(prepared).to(dtype)
                ready = torch.cuda.Event()
                ready.record(stream)
                entry = (weights, ready, stream.cuda_stream)
            weights, ready, owner = entry
            if weights.numel() * weights.element_size() <= _MAX_WEIGHT_BYTES:
                _weights[key] = entry
                while (len(_weights) > _MAX_WEIGHT_ENTRIES or
                       sum(item[0].numel() * item[0].element_size() for item in _weights.values()) > _MAX_WEIGHT_BYTES):
                    _weights.popitem(last=False)
            if owner != stream.cuda_stream:
                stream.wait_event(ready)
                weights.record_stream(stream)
    return {"weights": weights, "m": prepared.m, "n": prepared.n,
            **({"augmented": True} if cfg.reference_mode == "augmented" else {})}


def _tensors(finalized, diagnostics):
    reference, residual, threshold, flags, hit, location = finalized
    detected, nonfinite, located, recovered, recheck_failed = flags.unbind()
    tensors = dict(detected=detected, nonfinite=nonfinite, located=located,
                   recovered=recovered, recheck_failed=recheck_failed,
                   fault_hit=hit, location=location)
    if diagnostics:
        tensors.update(residual=residual, threshold=threshold)
    return reference, tensors


def _execute_native(prepared, cfg, fault, context, *, method=None, policy=None, profile_scope=None):
    """Resolve a method host plan once; kernels never call Python method hooks."""
    from .engine import AbftError, AbftReport, _commit_method, _report_context, _selected

    if method is None:
        method = get_method(cfg.method)
        if method.stateful:
            raise ValueError("Stateful native methods require a retained method_instance or MethodSession")
    policy = method.validate_execution(prepared, cfg) if policy is None else policy
    augmented = cfg.reference_mode == "augmented"
    need_statistics = cfg.diagnostics or method.needs_diagnostics
    algorithm = policy.algorithm
    policy_args = () if algorithm == 0 else (algorithm,)

    backend.require_device(prepared.a.device)
    selected = _selected(fault, context)
    injection = (-1, -1, -1)
    if selected:
        if fault.location != "accumulator":
            raise ValueError("native fused GEMM accepts only accumulator faults")
        if fault.row >= prepared.m or fault.col >= prepared.n or fault.bit >= 32:
            raise ValueError("Accumulator fault coordinate/bit outside FP32 output")
        injection = (fault.row, fault.col, fault.bit)
    started = time.perf_counter()
    events = []
    scope = profile_scope or nullcontext

    @contextmanager
    def phase(name):
        with scope("abft." + name):
            mark = None
            if cfg.timing and prepared.a.is_cuda:
                mark = torch.cuda.Event(enable_timing=True)
                mark.record(torch.cuda.current_stream(prepared.a.device))
            yield
            if mark is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record(torch.cuda.current_stream(prepared.a.device))
                events.append((name, mark, end))

    with phase("reference"):
        state = _builtin_weights(method, prepared, cfg, policy)
        weights = state["weights"]
        weight_dtype = torch.bfloat16 if augmented else torch.float32
        if (weights.ndim != 2 or weights.shape != (policy.channels, prepared.m) or
                weights.dtype != weight_dtype or weights.device != prepared.a.device or
                not weights.is_contiguous()):
            raise ValueError("Native method weights do not match compiled policy channels/format/device")
        # prepare owns snapshotting adaptive state. Retain this exact tensor
        # through both executions; never recalibrate it from current evidence.
        thresholds = state.get("thresholds")
        if thresholds is not None and (
            not isinstance(thresholds, torch.Tensor) or thresholds.dtype != torch.float32 or
            thresholds.device != prepared.a.device or not thresholds.is_contiguous() or
            thresholds.shape not in {(1,), (policy.channels, prepared.n)}
        ):
            raise ValueError("Native method thresholds must be contiguous FP32 [1] or [channels,N] on the operand device")
        finalize_args = (algorithm, thresholds) if thresholds is not None else policy_args

    def run(injection):
        if augmented:
            output, partials, tile_flags = torch.ops.sdc_cutlass.abft_augmented_compact(
                prepared.a, prepared.b, weights, *injection, *policy_args,
            )
            return output, (partials, tile_flags)
        # The C++ entry zeros this per-call flag once and the epilogue records
        # actual post-cast output nonfinites. There is no global C scan here.
        nonfinite = torch.empty(1, device=prepared.a.device, dtype=torch.int32)
        output, observed, hit, predicted, magnitude = prepared.collected_reference_native(
            weights, injection, algorithm=algorithm, nonfinite=nonfinite,
        )
        return output, (observed, predicted, magnitude, hit, nonfinite)

    def finalize(evidence, first_reference):
        if augmented:
            return torch.ops.sdc_cutlass.abft_finalize(
                *evidence, prepared.n, cfg.atol, cfg.rtol, first_reference, need_statistics, *finalize_args,
            )
        return torch.ops.sdc_cutlass.abft_finalize_reference(
            *evidence, prepared.n, cfg.atol, cfg.rtol, first_reference, need_statistics, *finalize_args,
        )

    with phase("execute"):
        output, evidence = run(injection)
    with phase("finalize"):
        reference, tensors = _tensors(finalize(evidence, None), need_statistics)
    state = {**state, "reference": reference}
    # Scratch is no longer part of the report. The allocator handles ordered
    # reuse; result tensors and first reference remain owned by this call.
    del evidence
    report_context = _report_context(prepared, cfg, fault, context, selected, output, method)
    report_context["reference_operand_reuse"] = "main_B_register_fragments" if augmented else "mainloop_shared_operands"
    report_context["threshold_source"] = "method_state" if thresholds is not None else "config_atol_rtol"
    report = AbftReport(tensors, report_context, cfg.method,
                        "augmented_native" if augmented else "input_fused_native", events)
    if cfg.recovery == "recompute" and bool(tensors["detected"].item()):
        with phase("recover_recheck"):
            replacement, retry_evidence = run((-1, -1, -1))
            _, checked = _tensors(finalize(retry_evidence, reference), need_statistics)
            tensors["recheck_failed"] = checked["detected"]
            tensors["recovered"] = ~checked["detected"]
            if need_statistics:
                tensors["retry_residual"] = checked["residual"]
            if not bool(checked["detected"].item()):
                output = replacement
    with phase("state_update") if method.stateful or method.needs_diagnostics else nullcontext():
        _commit_method(method, state, tensors, context, prepared)
    if not cfg.diagnostics:
        for key in ("residual", "threshold", "retry_residual"):
            tensors.pop(key, None)
    should_raise = (cfg.on_error == "raise" and bool(tensors["detected"].item())
                    and not bool(tensors["recovered"].item()))
    report.host_seconds = time.perf_counter() - started
    if should_raise:
        raise AbftError(report)
    return output, report


def execute_augmented(prepared, cfg, fault, context, *, method=None, policy=None, profile_scope=None):
    return _execute_native(prepared, cfg, fault, context, method=method, policy=policy,
                           profile_scope=profile_scope)


def execute_input_fused(prepared, cfg, fault, context, *, method=None, policy=None, profile_scope=None):
    return _execute_native(prepared, cfg, fault, context, method=method, policy=policy,
                           profile_scope=profile_scope)
