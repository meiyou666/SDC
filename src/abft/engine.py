"""One preparation, optional fault, fused evidence, and clean-operand retry."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import math
import time

import torch

from cutlass_gemm import backend

from .config import parse_abft_config, parse_fault_config
from .methods import get_method
from .operands import INPUT_REFERENCE_PRECISIONS, PreparedGemm


@dataclass
class AbftReport:
    tensors: dict
    context: dict
    method: str
    execution_path: str
    timing_events: list = field(default_factory=list)
    host_seconds: float = 0.0

    def __getattr__(self, name):
        if name in self.tensors:
            return self.tensors[name]
        raise AttributeError(name)

    def to_record(self):
        """The explicit CPU synchronization boundary for experiment logging."""
        record = {name: value.detach().cpu().tolist() for name, value in self.tensors.items()}
        record.update(context=self.context, method=self.method, execution_path=self.execution_path,
                      host_seconds=self.host_seconds)
        if self.timing_events:
            self.timing_events[-1][2].synchronize()
            record["timing_ms"] = {name: start.elapsed_time(end) for name, start, end in self.timing_events}
        nonfinite = {}
        def clean(value, path=""):
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite[path] = str(value)
                return None
            if isinstance(value, dict):
                return {key: clean(item, f"{path}.{key}" if path else str(key))
                        for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item, f"{path}[{index}]") for index, item in enumerate(value)]
            return value
        record = clean(record)
        if nonfinite:
            record["nonfinite_metrics"] = nonfinite
        return record


class AbftError(RuntimeError):
    def __init__(self, report):
        super().__init__("ABFT detected an unrecovered anomaly")
        self.report = report


def _selected(fault, context):
    return (fault.location != "none" and not context.get("retry", False)
            and context.get("call_index", 1) == fault.call_index
            and (fault.layer is None or context.get("layer") == fault.layer)
            and (fault.role is None or context.get("role") == fault.role))


def _report_context(prepared, cfg, fault, context, selected, output):
    augmented = cfg.reference_mode == "augmented"
    fused = cfg.reference_mode in {"input_fused", "augmented"}
    precision = prepared.precision
    return {**context, "fault": asdict(fault), "fault_selected": selected,
            "precision": precision,
            "scaling": {"fp32": "none", "bf16": "none", "fp8": "tensorwise", "nvfp4": "block16"}[precision],
            "reference_mode": cfg.reference_mode, "implementation": cfg.implementation,
            "threshold_scale": cfg.threshold_scale,
            "checksum_input_precision": "bf16" if augmented else "fp32",
            "reference_algorithm": "augmented_gemm" if augmented else "separate_projection_arithmetic",
            "augmented_rows_per_data_tile": 2 if augmented else 0,
            "protection_scope": "computation_after_input_load" if fused else "prepared_inputs_and_computation",
            "m": prepared.m, "n": prepared.n, "k": prepared.k,
            "input_precision": precision, "accumulator_precision": "fp32",
            "output_precision": str(output.dtype), "check_precision": "fp32",
            "injection_point": ("none" if not selected else
                                "post_alpha_fp32_epilogue_before_output_cast"
                                if fault.location == "accumulator" else "prepared_input_codes"),
            "fault_hit_meaning": "injection_executed_not_output_damage",
            "location_assumption": "disabled_for_low_precision_encoding" if augmented else "single_output_element"}


@torch.no_grad()
def execute(a, b, precision, scaling, global_scale="amax", abft_config=None,
            fault_config=None, context=None, *, raw_fn=None, reference_mm=None, collector_fn=None,
            reference_collector_fn=None):
    """Training convenience entry: prepare once, then execute prepared GEMM.

    Quantization/input preparation is outside the returned report's execution
    timings. Unprotected calls preserve the original backend.gemm fast path.
    """
    cfg, fault = parse_abft_config(abft_config), parse_fault_config(fault_config)
    context = {"layer": None, "role": None, "call_index": 1,
               "execution_kind": "forward", "retry": False, **dict(context or {})}
    selected = _selected(fault, context)
    if cfg.method == "none" and not selected:
        if raw_fn is None:
            return backend.gemm(a, b, precision, scaling, global_scale), None
    prepared = PreparedGemm.prepare(a, b, precision, scaling, global_scale)
    return execute_prepared(prepared, cfg, fault, context, raw_fn=raw_fn,
                            reference_mm=reference_mm, collector_fn=collector_fn,
                            reference_collector_fn=reference_collector_fn)


@torch.no_grad()
def execute_prepared(prepared, abft_config=None, fault_config=None, context=None,
                     *, raw_fn=None, reference_mm=None, collector_fn=None, reference_collector_fn=None,
                     profile_scope=None):
    """Execute existing native operands without preparing or quantizing again.

    Reference decoding and construction are protection costs inside this call.
    raw_fn(prepared), collector_fn(prepared, weights, fault_tuple), and
    reference_mm(left, right) are CPU-test dependency injection hooks.
    reference_collector_fn(prepared, weights, fault_tuple) explicitly returns
    (output, observed_partials, hit, predicted_partials, magnitude_partials).
    In input_fused mode the reference stage prepares projection metadata;
    input-side reference arithmetic belongs to the execute stage, and binding
    its tile partials belongs to verify. Retry retains the first reference.
    Optional stage events exclude input preparation; external benchmark events
    should encompass this entire call. host_seconds measures host execution,
    including policy-required synchronization but not asynchronous GPU work.
    """
    cfg, fault = parse_abft_config(abft_config), parse_fault_config(fault_config)
    scope = profile_scope or nullcontext
    augmented = cfg.reference_mode == "augmented"
    input_fused = cfg.reference_mode in {"input_fused", "augmented"}
    if augmented and prepared.precision != "bf16":
        raise ValueError("augmented currently requires BF16 GEMM")
    if input_fused:
        if prepared.precision not in INPUT_REFERENCE_PRECISIONS:
            raise NotImplementedError(f"input_fused reference is not implemented for {prepared.precision}")
        if fault.location in {"a", "b"}:
            raise ValueError("input_fused protects computation only; input memory faults require separate reference")
    context = {"layer": None, "role": None, "call_index": 1,
               "execution_kind": "forward", "retry": False, **dict(context or {})}
    selected = _selected(fault, context)
    if cfg.implementation == "native":
        if any(fn is not None for fn in (raw_fn, reference_mm, collector_fn, reference_collector_fn)):
            raise ValueError("native implementation does not accept reference collector overrides")
        from .native import execute_augmented

        return execute_augmented(prepared, cfg, fault, context, profile_scope=profile_scope)
    if cfg.method == "none" and not selected:
        return (raw_fn(prepared) if raw_fn else prepared.raw_native()), None
    a = prepared.a
    started = time.perf_counter()
    events = []
    def begin():
        if cfg.timing and a.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(a.device))
            return event
        return None
    def end(name, event):
        if event is not None:
            finish = begin()
            events.append((name, event, finish))
    mark = begin()
    method = get_method(cfg.method) if cfg.method != "none" else None
    if method:
        method.validate_config(cfg)
        method._profile_scope = scope
    with scope("abft.reference"):
        if method is None:
            state = None
        elif augmented:
            state = method.prepare_augmented(prepared, cfg)
        elif input_fused:
            state = method.prepare_fused(prepared, cfg)
        else:
            state = method.prepare(prepared, cfg, reference_mm)
    end("reference", mark)
    mark = begin()
    active = prepared.injected(fault) if selected and fault.location in {"a", "b"} else prepared
    injection = (-1, -1, -1)
    if selected and fault.location == "accumulator":
        if fault.row >= prepared.m or fault.col >= prepared.n or fault.bit >= 32:
            raise ValueError("Accumulator fault coordinate/bit outside FP32 output")
        injection = (fault.row, fault.col, fault.bit)
    end("fault_prepare", mark)
    fused = method is not None or injection[0] >= 0
    weights = state["weights"] if state else torch.zeros((2, prepared.m), device=a.device, dtype=torch.float32)
    def collect(operands, fault_tuple):
        if collector_fn is not None:
            return collector_fn(operands, weights, fault_tuple)
        return operands.collected_native(weights, fault_tuple)
    def collect_reference(operands, fault_tuple):
        if reference_collector_fn is not None:
            return reference_collector_fn(operands, weights, fault_tuple)
        if augmented:
            return operands.augmented_native(weights, fault_tuple)
        return operands.collected_reference_native(weights, fault_tuple)
    mark = begin()
    with scope("abft.execute"):
        if input_fused:
            output, partials, hit, predicted, magnitude = collect_reference(active, injection)
        elif fused:
            output, partials, hit = collect(active, injection)
        else:
            output = raw_fn(active) if raw_fn else active.raw_native()
            hit = torch.zeros(1, device=a.device, dtype=torch.int32)
    if selected and fault.location in {"a", "b"}:
        hit = torch.ones(1, device=a.device, dtype=torch.int32)
    end("execute", mark)
    mark = begin()
    if input_fused:
        with scope("abft.merge_reference"):
            state = method.bind_reference(predicted, magnitude, state)
    if method:
        with scope("abft.merge_observed"):
            observed = method.propagate(partials, state)
        with scope("abft.verify"):
            result = method.verify(observed, output, state, cfg)
    else:
        false = torch.zeros((), device=a.device, dtype=torch.bool)
        result = {"detected": false, "nonfinite": (~torch.isfinite(output)).any(),
                  "located": false, "location": torch.full((2,), -1, device=a.device, dtype=torch.int64)}
    with scope("abft.report_flags"):
        result.update(fault_hit=hit, recovered=torch.zeros((), device=a.device, dtype=torch.bool),
                      recheck_failed=torch.zeros((), device=a.device, dtype=torch.bool))
    end("verify", mark)
    report_context = _report_context(prepared, cfg, fault, context, selected, output)
    report = AbftReport(result, report_context, cfg.method, "instrumented" if fused else "raw", events)
    if method and cfg.recovery == "recompute" and bool(result["detected"].item()):
        mark = begin()
        # Reuse state and clean buffers. The same collector gives a pre-cast
        # recheck, which cannot be replaced by summing a rounded BF16 output.
        if input_fused:
            # Never replace the original reference with retry evidence: a
            # co-varying retry reference must not authorize corrupted output.
            replacement, retry_partials, _, _, _ = collect_reference(prepared, (-1, -1, -1))
        elif collector_fn is None:
            replacement, retry_partials, _ = method.recover(prepared, state)
        else:
            replacement, retry_partials, _ = collect(prepared, (-1, -1, -1))
        checked = method.verify(method.propagate(retry_partials, state), replacement, state, cfg)
        result["recheck_failed"] = checked["detected"]
        result["recovered"] = ~checked["detected"]
        if "residual" in checked:
            result["retry_residual"] = checked["residual"]
        if not bool(checked["detected"].item()):
            output = replacement
        end("recover_recheck", mark)
    if not cfg.diagnostics:
        for key in ("residual", "threshold", "retry_residual"):
            result.pop(key, None)
    should_raise = method and cfg.on_error == "raise" and bool((result["detected"] & ~result["recovered"]).item())
    report.host_seconds = time.perf_counter() - started
    if should_raise:
        raise AbftError(report)
    return output, report
