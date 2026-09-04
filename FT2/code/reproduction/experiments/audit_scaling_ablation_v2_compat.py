#!/usr/bin/env python3
"""Run the scaling-v2 auditor while adapting legacy main_18k run records.

Legacy reused fault-run artifacts do not duplicate ``references``; their paired
clean controls do.  Return a shallow in-memory view with those references so the
original, fingerprinted experiment script can audit and summarize all modes.
No raw artifact is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "reproduction" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import run_scaling_ablation_v2 as experiment


_effective_run = experiment.effective_run


def effective_run_with_legacy_references(
    context: Mapping[str, Any], spec_id: str, label: str
) -> dict[str, Any]:
    raw = _effective_run(context, spec_id, label)
    if raw.get("references"):
        return raw
    item = next(item for item in context["specs"] if item.spec_id == spec_id)
    source = context["entry"]["source_positions"][item.sample_position]
    control = experiment.effective_control(context, source, label)
    adapted = dict(raw)
    adapted["references"] = list(control["references"])
    return adapted


experiment.effective_run = effective_run_with_legacy_references

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--audit-only"]
    raise SystemExit(experiment.main())
