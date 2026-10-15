"""One preparation, optional fault, fused evidence, and clean-operand retry."""
from dataclasses import asdict, dataclass, field
import math
import time

import torch

from cutlass_gemm import backend

from .config import parse_abft_config, parse_fault_config
from .methods import get_method
from .operands import PreparedGemm


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


@torch.no_grad()
def execute(a, b, precision, scaling, global_scale="amax", abft_config=None,
            fault_config=None, context=None, *, raw_fn=None, reference_mm=None, collector_fn=None):
    """Return output and optional report; injected callables support CPU tests.

    raw_fn(prepared), collector_fn(prepared, weights, fault_tuple), and
    reference_mm(left, right) are dependency injection hooks. Production uses
    existing native CUTLASS operations. No autograd graph is created here.
    """
    cfg, fault = parse_abft_config(abft_config), parse_fault_config(fault_config)
    context = {"layer": None, "role": None, "call_index": 1,
               "execution_kind": "forward", "retry": False, **dict(context or {})}
    selected = _selected(fault, context)
    if cfg.method == "none" and not selected:
        if raw_fn is None:
            return backend.gemm(a, b, precision, scaling, global_scale), None
        return raw_fn(PreparedGemm.prepare(a, b, precision, scaling, global_scale)), None
    started = time.perf_counter()
    events = []
    def begin():
        if a.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(a.device))
            return event
        return None
    def end(name, event):
        if event is not None:
            finish = begin()
            events.append((name, event, finish))
    mark = begin()
    prepared = PreparedGemm.prepare(a, b, precision, scaling, global_scale)
    method = get_method(cfg.method) if cfg.method != "none" else None
    state = method.prepare(prepared, cfg, reference_mm) if method else None
    end("prepare_reference", mark)
    mark = begin()
    active = prepared.injected(fault) if selected and fault.location in {"a", "b"} else prepared
    injection = (-1, -1, -1)
    if selected and fault.location == "accumulator":
        if fault.row >= prepared.m or fault.col >= prepared.n or fault.bit >= 32:
            raise ValueError("Accumulator fault coordinate/bit outside FP32 output")
        injection = (fault.row, fault.col, fault.bit)
    end("fault_prepare", mark)
    fused = method is not None or injection[0] >= 0
    weights = state["weights"] if state else torch.zeros((2, prepared.m), device=a.device)
    def collect(operands, fault_tuple):
        if collector_fn is not None:
            return collector_fn(operands, weights, fault_tuple)
        return operands.collected_native(weights, fault_tuple)
    mark = begin()
    if fused:
        output, partials, hit = collect(active, injection)
    else:
        output = raw_fn(active) if raw_fn else active.raw_native()
        hit = torch.zeros(1, device=a.device, dtype=torch.int32)
    if selected and fault.location in {"a", "b"}:
        hit = torch.ones(1, device=a.device, dtype=torch.int32)
    end("execute", mark)
    mark = begin()
    if method:
        result = method.verify(method.propagate(partials, state), output, state, cfg)
    else:
        false = torch.zeros((), device=a.device, dtype=torch.bool)
        result = {"detected": false, "nonfinite": (~torch.isfinite(output)).any(),
                  "located": false, "location": torch.full((2,), -1, device=a.device, dtype=torch.int64)}
    result.update(fault_hit=hit, recovered=torch.zeros((), device=a.device, dtype=torch.bool),
                  recheck_failed=torch.zeros((), device=a.device, dtype=torch.bool))
    end("verify", mark)
    report_context = {**context, "fault": asdict(fault), "fault_selected": selected,
                      "precision": precision, "scaling": scaling,
                      "m": prepared.m, "n": prepared.n, "k": prepared.k,
                      "input_precision": precision, "accumulator_precision": "fp32",
                      "output_precision": str(output.dtype), "check_precision": "fp32",
                      "injection_point": ("none" if not selected else
                                          "post_alpha_fp32_epilogue_before_output_cast"
                                          if fault.location == "accumulator" else "prepared_input_codes"),
                      "fault_hit_meaning": "injection_executed_not_output_damage",
                      "location_assumption": "single_output_element"}
    report = AbftReport(result, report_context, cfg.method, "instrumented" if fused else "raw", events)
    if method and cfg.recovery == "recompute" and bool(result["detected"].item()):
        mark = begin()
        # Reuse state and clean buffers. The same collector gives a pre-cast
        # recheck, which cannot be replaced by summing a rounded BF16 output.
        if collector_fn is None:
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
    report.host_seconds = time.perf_counter() - started
    if not cfg.diagnostics:
        for key in ("residual", "threshold", "retry_residual"):
            result.pop(key, None)
    if method and cfg.on_error == "raise" and bool((result["detected"] & ~result["recovered"]).item()):
        raise AbftError(report)
    return output, report
