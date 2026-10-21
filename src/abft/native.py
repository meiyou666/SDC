"""Compact augmented GEMM followed by one device-side merge and decision.

Only immutable classic weights are cached. Scratch and report tensors remain
owned by each call, so concurrent streams and retained reports never alias a
mutable global workspace.
"""
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from threading import Lock
import time

import torch

from cutlass_gemm import backend


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


def _tensors(finalized, diagnostics):
    reference, residual, threshold, flags, hit, location = finalized
    detected, nonfinite, located, recovered, recheck_failed = flags.unbind()
    tensors = dict(detected=detected, nonfinite=nonfinite, located=located,
                   recovered=recovered, recheck_failed=recheck_failed,
                   fault_hit=hit, location=location)
    if diagnostics:
        tensors.update(residual=residual, threshold=threshold)
    return reference, tensors


def execute_augmented(prepared, cfg, fault, context, *, profile_scope=None):
    """Use the same encoding/thresholds as reference; retain first retry evidence."""
    from .engine import AbftError, AbftReport, _report_context, _selected

    backend.require_device(prepared.a.device)
    selected = _selected(fault, context)
    injection = (-1, -1, -1)
    if selected:
        if fault.location != "accumulator":
            raise ValueError("native augmented GEMM accepts only accumulator faults")
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
        weights = classic_weights(prepared.m, prepared.a.device)
    with phase("execute"):
        output, partials, tile_flags = torch.ops.sdc_cutlass.abft_augmented_compact(
            prepared.a, prepared.b, weights, *injection,
        )
    with phase("finalize"):
        reference, tensors = _tensors(torch.ops.sdc_cutlass.abft_finalize(
            partials, tile_flags, prepared.n, cfg.atol, cfg.rtol, None, cfg.diagnostics,
        ), cfg.diagnostics)
    # Scratch is no longer part of the report. The allocator handles ordered
    # reuse; result tensors and first reference remain owned by this call.
    del partials, tile_flags
    report_context = _report_context(prepared, cfg, fault, context, selected, output)
    report_context["reference_operand_reuse"] = "main_B_register_fragments"
    report = AbftReport(tensors, report_context, cfg.method, "augmented_native", events)
    if cfg.recovery == "recompute" and bool(tensors["detected"].item()):
        with phase("recover_recheck"):
            replacement, retry_partials, retry_flags = torch.ops.sdc_cutlass.abft_augmented_compact(
                prepared.a, prepared.b, weights, -1, -1, -1,
            )
            _, checked = _tensors(torch.ops.sdc_cutlass.abft_finalize(
                retry_partials, retry_flags, prepared.n, cfg.atol, cfg.rtol, reference, cfg.diagnostics,
            ), cfg.diagnostics)
            tensors["recheck_failed"] = checked["detected"]
            tensors["recovered"] = ~checked["detected"]
            if cfg.diagnostics:
                tensors["retry_residual"] = checked["residual"]
            if not bool(checked["detected"].item()):
                output = replacement
    should_raise = (cfg.on_error == "raise" and bool(tensors["detected"].item())
                    and not bool(tensors["recovered"].item()))
    report.host_seconds = time.perf_counter() - started
    if should_raise:
        raise AbftError(report)
    return output, report
