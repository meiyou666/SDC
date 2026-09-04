from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODE_ORDER = (
    "no_protection",
    "paper_clamp_first_token_bounds",
    "paper_clamp_offline_bounds",
)
MODE_LABELS = {
    "no_protection": "No Protection",
    "paper_clamp_first_token_bounds": "FT2 First-token",
    "paper_clamp_offline_bounds": "Offline Bounds",
}
MODE_COLORS = {
    "no_protection": "#6B7280",
    "paper_clamp_first_token_bounds": "#2563EB",
    "paper_clamp_offline_bounds": "#059669",
}
PAIR_ORDER = (
    ("opt_2_7b", "squad_v2"),
    ("opt_2_7b", "xtreme_mlqa_en_en"),
    ("qwen2_math_7b", "squad_v2"),
    ("qwen2_math_7b", "xtreme_mlqa_en_en"),
    ("qwen2_math_7b", "gsm8k"),
)
PAIR_LABELS = {
    ("opt_2_7b", "squad_v2"): "OPT\nSQuAD 2.0",
    ("opt_2_7b", "xtreme_mlqa_en_en"): "OPT\nMLQA en-en",
    ("qwen2_math_7b", "squad_v2"): "Qwen2\nSQuAD 2.0",
    ("qwen2_math_7b", "xtreme_mlqa_en_en"): "Qwen2\nMLQA en-en",
    ("qwen2_math_7b", "gsm8k"): "Qwen2\nGSM8K",
}
CRITICAL_PROJECTIONS = {
    "opt_2_7b": frozenset(("v_proj", "out_proj", "fc2")),
    "qwen2_math_7b": frozenset(
        ("v_proj", "o_proj", "up_proj", "down_proj")
    ),
}
OUTCOME_ORDER = ("MASKED_IDENTICAL", "MASKED_SEMANTIC", "SDC", "DUE")
OUTCOME_LABELS = {
    "MASKED_IDENTICAL": "Masked: identical",
    "MASKED_SEMANTIC": "Masked: semantic",
    "SDC": "SDC",
    "DUE": "DUE",
}
OUTCOME_COLORS = {
    "MASKED_IDENTICAL": "#9CA3AF",
    "MASKED_SEMANTIC": "#60A5FA",
    "SDC": "#DC2626",
    "DUE": "#7C3AED",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return float(center - half), float(center + half)


def _site_classes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted((root / "manifests").glob("*.json")):
        manifest = _read_json(path)
        for spec in manifest["specs"]:
            projections = CRITICAL_PROJECTIONS[spec["model_key"]]
            result[spec["spec_id"]] = (
                "critical"
                if spec["projection"] in projections
                else "non_critical"
            )
    if len(result) != 60:
        raise AssertionError(f"Expected 60 manifest specs, got {len(result)}")
    return result


def _load_runs(root: Path) -> list[dict[str, Any]]:
    runs = [
        _read_json(path)
        for path in sorted((root / "runs").glob("*/*.json"))
    ]
    if len(runs) != 180:
        raise AssertionError(f"Expected 180 run files, got {len(runs)}")
    return runs


def _save_figure(fig: plt.Figure, figures: Path, stem: str) -> list[str]:
    outputs = []
    for suffix in ("png", "svg"):
        path = figures / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _mode_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["mode_id"]].append(run)
    rows = []
    for mode in MODE_ORDER:
        records = grouped[mode]
        outcomes = Counter(
            record["classification"]["outcome"] for record in records
        )
        corrected = sum(
            record["engine"]["injection_trace"]["correction_action"] != "none"
            for record in records
        )
        total = len(records)
        low, high = _wilson(outcomes["SDC"], total)
        rows.append(
            {
                "mode": mode,
                "label": MODE_LABELS[mode],
                "total": total,
                "masked_identical": outcomes["MASKED_IDENTICAL"],
                "masked_semantic": outcomes["MASKED_SEMANTIC"],
                "sdc": outcomes["SDC"],
                "due": outcomes["DUE"],
                "sdc_rate": outcomes["SDC"] / total,
                "sdc_wilson95_low": low,
                "sdc_wilson95_high": high,
                "target_corrected": corrected,
                "target_correction_rate": corrected / total,
            }
        )
    return rows


def _pair_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[
            (run["model_key"], run["dataset_key"], run["mode_id"])
        ].append(run)
    rows = []
    for model, dataset in PAIR_ORDER:
        for mode in MODE_ORDER:
            records = grouped[(model, dataset, mode)]
            outcomes = Counter(
                record["classification"]["outcome"] for record in records
            )
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "mode": mode,
                    "total": len(records),
                    "sdc": outcomes["SDC"],
                    "sdc_rate": outcomes["SDC"] / len(records),
                    "masked_identical": outcomes["MASKED_IDENTICAL"],
                    "masked_semantic": outcomes["MASKED_SEMANTIC"],
                    "due": outcomes["DUE"],
                }
            )
    return rows


def _site_rows(
    runs: Iterable[Mapping[str, Any]],
    site_classes: Mapping[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["mode_id"], site_classes[run["spec_id"]])].append(run)
    rows = []
    for mode in MODE_ORDER:
        for site_class in ("critical", "non_critical"):
            records = grouped[(mode, site_class)]
            outcomes = Counter(
                record["classification"]["outcome"] for record in records
            )
            corrected = sum(
                record["engine"]["injection_trace"]["correction_action"] != "none"
                for record in records
            )
            low, high = _wilson(outcomes["SDC"], len(records))
            rows.append(
                {
                    "mode": mode,
                    "site_class": site_class,
                    "total": len(records),
                    "sdc": outcomes["SDC"],
                    "sdc_rate": outcomes["SDC"] / len(records),
                    "sdc_wilson95_low": low,
                    "sdc_wilson95_high": high,
                    "target_corrected": corrected,
                    "target_correction_rate": corrected / len(records),
                }
            )
    return rows


def _plot_mode_outcomes(
    rows: Sequence[Mapping[str, Any]], figures: Path
) -> list[str]:
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for outcome in OUTCOME_ORDER:
        values = np.array(
            [
                row[
                    {
                        "MASKED_IDENTICAL": "masked_identical",
                        "MASKED_SEMANTIC": "masked_semantic",
                        "SDC": "sdc",
                        "DUE": "due",
                    }[outcome]
                ]
                for row in rows
            ]
        )
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=OUTCOME_COLORS[outcome],
            label=OUTCOME_LABELS[outcome],
            width=0.68,
        )
        bottom += values
    for index, row in enumerate(rows):
        ax.text(
            index,
            row["total"] + 1.0,
            f"SDC {row['sdc']}/{row['total']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x, [row["label"] for row in rows])
    ax.set_ylabel("Faulted inferences")
    ax.set_ylim(0, 68)
    ax.set_title("RTX 4090 reduced pilot: outcome counts")
    ax.legend(ncol=2, frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    return _save_figure(fig, figures, "pilot_outcomes_by_mode")


def _plot_sdc_rates(
    rows: Sequence[Mapping[str, Any]], figures: Path
) -> list[str]:
    x = np.arange(len(rows))
    rates = np.array([row["sdc_rate"] for row in rows])
    lows = np.array([row["sdc_wilson95_low"] for row in rows])
    highs = np.array([row["sdc_wilson95_high"] for row in rows])
    yerr = np.vstack((rates - lows, highs - rates))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(
        x,
        rates * 100,
        color=[MODE_COLORS[row["mode"]] for row in rows],
        width=0.62,
        yerr=yerr * 100,
        capsize=5,
    )
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0.15) + 0.45,
            f"{row['sdc']}/{row['total']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x, [row["label"] for row in rows])
    ax.set_ylabel("SDC rate (%)")
    ax.set_title("SDC rate with 95% Wilson intervals")
    ax.grid(axis="y", alpha=0.2)
    return _save_figure(fig, figures, "pilot_sdc_rate_by_mode")


def _plot_pair_sdc(
    rows: Sequence[Mapping[str, Any]], figures: Path
) -> list[str]:
    lookup = {
        (row["model"], row["dataset"], row["mode"]): row for row in rows
    }
    x = np.arange(len(PAIR_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    for offset, mode in enumerate(MODE_ORDER):
        values = [
            lookup[(model, dataset, mode)]["sdc"]
            for model, dataset in PAIR_ORDER
        ]
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
    ax.set_xticks(x, [PAIR_LABELS[pair] for pair in PAIR_ORDER])
    ax.set_ylabel("SDC count (12 specs per mode/pair)")
    ax.set_yticks(range(0, 4))
    ax.set_title("Pilot SDC counts by model and dataset")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    return _save_figure(fig, figures, "pilot_sdc_by_pair")


def _plot_site_class(
    rows: Sequence[Mapping[str, Any]], figures: Path
) -> list[str]:
    lookup = {
        (row["mode"], row["site_class"]): row for row in rows
    }
    x = np.arange(len(MODE_ORDER))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    class_colors = {"critical": "#F97316", "non_critical": "#64748B"}
    for position, site_class in enumerate(("critical", "non_critical")):
        offset = (position - 0.5) * width
        sdc_rates = [
            100 * lookup[(mode, site_class)]["sdc_rate"]
            for mode in MODE_ORDER
        ]
        correction_rates = [
            100 * lookup[(mode, site_class)]["target_correction_rate"]
            for mode in MODE_ORDER
        ]
        axes[0].bar(
            x + offset,
            sdc_rates,
            width,
            color=class_colors[site_class],
            label=site_class.replace("_", "-"),
        )
        axes[1].bar(
            x + offset,
            correction_rates,
            width,
            color=class_colors[site_class],
            label=site_class.replace("_", "-"),
        )
    labels = [MODE_LABELS[mode] for mode in MODE_ORDER]
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("SDC rate (%)")
    axes[0].set_title("Outcome by sampled site class")
    axes[1].set_ylabel("Target correction rate (%)")
    axes[1].set_title("Direct FT2 correction at target")
    fig.suptitle("Critical (n=33) vs non-critical (n=27) FaultSpecs")
    return _save_figure(fig, figures, "pilot_critical_noncritical")


def _plot_transitions(
    audit: Mapping[str, Any], figures: Path
) -> list[str]:
    transition_keys = (
        ("first_token_vs_no_protection", "FT2 First-token"),
        ("offline_vs_no_protection", "Offline Bounds"),
    )
    recovered = []
    persistent = []
    for key, _ in transition_keys:
        transitions = audit["paired_transitions"][key]
        recovered.append(
            sum(
                count
                for transition, count in transitions.items()
                if transition.startswith("SDC->")
                and not transition.endswith("->SDC")
            )
        )
        persistent.append(transitions.get("SDC->SDC", 0))
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(x, recovered, color="#059669", label="Recovered to Masked")
    ax.bar(
        x,
        persistent,
        bottom=recovered,
        color="#DC2626",
        label="Persistent SDC",
    )
    for index in range(2):
        ax.text(
            index,
            3.08,
            f"{recovered[index]}/3 recovered",
            ha="center",
            va="bottom",
        )
    ax.set_xticks(x, [label for _, label in transition_keys])
    ax.set_ylabel("Initially-SDC paired FaultSpecs")
    ax.set_ylim(0, 3.7)
    ax.set_yticks(range(0, 4))
    ax.set_title("Paired transitions from No Protection (n=3 SDC)")
    ax.legend(frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    return _save_figure(fig, figures, "pilot_paired_sdc_transitions")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1] / "results" / "pilot"
    parser = argparse.ArgumentParser(
        description="Create audited figures for the FT2 reduced pilot"
    )
    parser.add_argument("--output-root", type=Path, default=default_root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    audit = _read_json(root / "audit.json")
    if audit["status"] != "passed":
        raise RuntimeError("Refuse to plot a campaign whose audit did not pass")
    runs = _load_runs(root)
    site_classes = _site_classes(root)
    mode_rows = _mode_rows(runs)
    pair_rows = _pair_rows(runs)
    site_rows = _site_rows(runs, site_classes)

    summaries = root / "summaries"
    figures = root / "figures"
    summaries.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    _write_csv(summaries / "pilot_mode_summary.csv", mode_rows)
    _write_csv(summaries / "pilot_pair_summary.csv", pair_rows)
    _write_csv(summaries / "pilot_site_class_summary.csv", site_rows)

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    outputs = []
    outputs.extend(_plot_mode_outcomes(mode_rows, figures))
    outputs.extend(_plot_sdc_rates(mode_rows, figures))
    outputs.extend(_plot_pair_sdc(pair_rows, figures))
    outputs.extend(_plot_site_class(site_rows, figures))
    outputs.extend(_plot_transitions(audit, figures))

    script_bytes = Path(__file__).read_bytes()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_fingerprint": audit["campaign_fingerprint"],
        "audit_artifact_sha256": audit["artifact_sha256"],
        "plot_script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "source_run_count": len(runs),
        "figures": [
            str(Path(path).relative_to(root))
            for path in outputs
        ],
        "summaries": [
            "summaries/pilot_mode_summary.csv",
            "summaries/pilot_pair_summary.csv",
            "summaries/pilot_site_class_summary.csv",
        ],
    }
    with (root / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

