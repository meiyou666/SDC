"""Paired whole-call detection and recovery metrics on saved training operands."""

import copy
from dataclasses import asdict, replace
import math

import torch

from algorithms import detector_for, resolve_algorithm

from .api import RecoveryError, gemm
from .types import PROFILES, ExecutionContext, Fault


def rate(successes, total):
    if not 0 <= successes <= total:
        raise ValueError("统计分子必须位于 0 与分母之间。")
    if not total:
        return {"numerator": successes, "denominator": 0, "value": None, "ci95": None}
    p, z = successes / total, 1.959963984540054
    denom = 1 + z*z/total
    center = (p + z*z/(2*total))/denom
    delta = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return {"numerator": successes, "denominator": total, "value": p,
            "ci95": [max(0., center-delta), min(1., center+delta)]}


def changed_elements(a, b):
    return a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)


def bit_events(m, n, k, profile, *, seed=2026, sites=("a", "b", "accumulator")):
    generator = torch.Generator().manual_seed(seed)
    shapes = {"a": (m, k), "b": (k, n), "accumulator": (m, n)}
    for site in sites:
        rows, cols = shapes[site]
        row, col = int(torch.randint(rows, (), generator=generator)), int(torch.randint(cols, (), generator=generator))
        for bit in range(32 if site == "accumulator" else PROFILES[profile].bits):
            yield Fault(site, row, col, bit)


def evaluate_case(a, b, *, profile, algorithm, backend="nvidia", phase="forward", context=None,
                  events=None, writer=None, detector=None):
    context = context or ExecutionContext()
    algorithm = resolve_algorithm(algorithm)
    if algorithm.name == "none":
        raise ValueError("故障检出评测需要检测算法。无保护 GEMM 用于性能参照。")
    detector = detector if detector is not None else detector_for(algorithm, backend)
    options = dict(profile=profile, algorithm=algorithm, backend=backend, phase=phase, context=context, detector=detector)
    snapshot = copy.deepcopy(detector.state_dict()) if detector is not None else None

    def run(**kwargs):
        if snapshot is not None:
            detector.load_state_dict(copy.deepcopy(snapshot))
        return gemm(a, b, **options, **kwargs)

    clean = run()
    encoded_k = (a.shape[1]+15)//16*16 if profile == "nvfp4" and phase == "wgrad" else a.shape[1]
    events = bit_events(a.shape[0], b.shape[1], encoded_k, profile, seed=context.stream_seed("fault", phase)) if events is None else events
    records = []
    for event in events:
        control = run(fault=replace(event, enabled=False))
        if bool(changed_elements(control.output, clean.output).any()):
            raise RuntimeError("零掩码调用改变了主计算结果。")
        result = run(fault=event)
        changed = changed_elements(result.output, clean.output)
        count = int(changed.sum())
        status = result.status.cpu().tolist()
        native = run(fault=event, recovery="native")
        try:
            recompute = run(fault=event, recovery="recompute")
            recompute_ok = recompute.recomputed and not bool(changed_elements(recompute.output, clean.output).any())
            recompute_failed = False
        except RecoveryError:
            recompute_ok, recompute_failed = False, True
        bits = result.fault_bits.cpu().tolist()
        values = result.fault_values.cpu().tolist() if result.fault_values is not None else [None, None]
        finite_values = all(value is not None and math.isfinite(value) for value in values)
        finite = bool(torch.isfinite(result.output).all())
        delta = (result.output.double()-clean.output.double()).norm().item() if finite and bool(torch.isfinite(clean.output).all()) else None
        native_equal = not bool(changed_elements(native.output, clean.output).any())
        location_ok = count == 1 and bool(status[1]) and 0 <= status[2] < a.shape[0] and 0 <= status[3] < b.shape[1] and bool(changed[status[2], status[3]])
        clean_max = float(clean.residual.abs().max()) if clean.residual.numel() else None
        record = {"profile": profile, "phase": phase, "algorithm": clean.algorithm, "backend": backend,
                  "context": asdict(context), "shape": [a.shape[0], b.shape[1], a.shape[1]],
                  "encoded_k": encoded_k,
                  "event": asdict(event), "bits_before_after": bits,
                  "fault_values": [value if value is None or math.isfinite(value) else str(value) for value in values],
                  "fault_delta_abs": abs(values[1]-values[0]) if finite_values else None,
                  "fault_delta_relative": abs(values[1]-values[0])/abs(values[0]) if finite_values and values[0] != 0 else None,
                  "direction": None if bits[0] < 0 else f"{(bits[0] >> event.bit) & 1}->{(bits[1] >> event.bit) & 1}",
                  "affected_elements": count, "finite_output": finite, "detected": bool(status[0]),
                  "located_correctly": location_ok, "native_attempted": bool(native.status[4]),
                  "native_postcheck_pass": not bool(native.status[5]), "native_equal": native_equal,
                  "recomputed": recompute_ok, "recompute_failed": recompute_failed,
                  "output_delta_l2": delta, "clean_residual_max": clean_max if clean_max is None or math.isfinite(clean_max) else str(clean_max),
                  "status": status}
        records.append(record)
        if writer:
            writer(record)
    affected = [r for r in records if r["affected_elements"]]
    singles = [r for r in records if r["affected_elements"] == 1]
    attempts = [r for r in records if r["native_attempted"]]
    if snapshot is not None:
        detector.load_state_dict(snapshot)
    return {"clean_calls": 1, "false_alarms": int(clean.detected), "events": len(records),
            "affected_events": len(affected), "nonfinite_events": sum(not r["finite_output"] for r in records),
            "fpr_call": rate(int(clean.detected), 1),
            "detection": rate(sum(r["detected"] for r in affected), len(affected)),
            "localization": rate(sum(r["located_correctly"] for r in singles), len(singles)),
            "native_recovery": rate(sum(r["native_attempted"] and r["native_postcheck_pass"] and r["native_equal"] for r in singles), len(singles)),
            "misrepair": rate(sum(r["native_postcheck_pass"] and not r["native_equal"] for r in attempts), len(attempts)),
            "recomputation": rate(sum(r["recomputed"] for r in affected), len(affected))}


def aggregate(summaries):
    result = {key: sum(s[key] for s in summaries) for key in (
        "clean_calls", "false_alarms", "events", "affected_events", "nonfinite_events")}
    for key in ("fpr_call", "detection", "localization", "native_recovery", "misrepair", "recomputation"):
        result[key] = rate(sum(s[key]["numerator"] for s in summaries), sum(s[key]["denominator"] for s in summaries))
    return result
