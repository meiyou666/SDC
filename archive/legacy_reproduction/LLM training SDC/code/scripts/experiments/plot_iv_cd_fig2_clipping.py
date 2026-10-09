#!/usr/bin/env python3
"""Render a Figure-2-style IV-C/D clipping comparison plot as PDF."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BASELINE = Path("checkpoints/iv_f_mb256/baseline/seed_42/metrics.jsonl")
DEFAULT_CLIPPED = Path("checkpoints/iv_b_mb256/seed_42/bit_13/BP9/metrics.jsonl")
DEFAULT_NO_CLIPPING = Path(
    "checkpoints/iv_cd_mb256/clipping/seed_42/bit_13/BP9/no_clipping/metrics.jsonl"
)
DEFAULT_OUTPUT = Path("figures/iv_cd_fig2_clipping_comparison.pdf")


def parse_number(value):
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return math.nan
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return math.nan
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if lowered in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(text)
    except ValueError:
        return math.nan


def load_loss_series(path):
    series = []
    triggers = []
    detected = []
    grad_pre = []
    grad_post = []
    final_eval_loss = math.nan

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("_step")
            if not isinstance(step, int):
                continue
            if "loss" in row:
                value = parse_number(row["loss"])
                if math.isfinite(value):
                    series.append((step, value))
            if parse_number(row.get("trigger_nvbit")) == 1.0:
                triggers.append(step)
            if parse_number(row.get("detected_anomaly")) == 1.0:
                detected.append(step)
            if "gradient_norm_pre" in row:
                grad_pre.append((step, parse_number(row["gradient_norm_pre"])))
            if "gradient_norm_post" in row:
                grad_post.append((step, parse_number(row["gradient_norm_post"])))
            if "eval_loss" in row:
                final_eval_loss = parse_number(row["eval_loss"])

    return {
        "series": series,
        "triggers": triggers,
        "detected": detected,
        "grad_pre": grad_pre,
        "grad_post": grad_post,
        "final_eval_loss": final_eval_loss,
    }


def finite_max(values):
    finite = [(step, value) for step, value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    return max(finite, key=lambda item: item[1])


def format_number(value, digits=3):
    value = parse_number(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if abs(value) >= 10000 or (0 < abs(value) < 0.001):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def render(args):
    root = args.root.resolve()
    baseline = load_loss_series(root / args.baseline_metrics)
    clipped = load_loss_series(root / args.clipped_metrics)
    no_clipping = load_loss_series(root / args.no_clipping_metrics)

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {
        "baseline": "#2563eb",
        "clipped": "#f97316",
        "no_clipping": "#16a34a",
    }
    labels = {
        "baseline": "Baseline Run",
        "clipped": "Fault Injection Run - w Gradient Norm Clipping",
        "no_clipping": "Fault Injection Run - w/o Gradient Norm Clipping",
    }
    datasets = [
        ("baseline", baseline),
        ("clipped", clipped),
        ("no_clipping", no_clipping),
    ]

    fig, ax = plt.subplots(figsize=(3.65, 2.95))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.73, bottom=0.17)

    for key, data in datasets:
        raw = data["series"]
        ax.plot(
            [step for step, _ in raw],
            [value for _, value in raw],
            color=colors[key],
            linewidth=1.25,
            label=labels[key],
            zorder=2,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([4, 5, 6, 7, 8, 9, 10])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.2, width=0.8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.43),
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#d1d5db",
        ncol=1,
        borderpad=0.35,
        handlelength=2.2,
    )

    fig.savefig(output, format="pdf", bbox_inches="tight")
    if args.preview_png:
        fig.savefig(root / args.preview_png, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for key, data in datasets:
        max_grad_step, max_grad_value = finite_max(data["grad_pre"])
        print(
            f"{key}: {len(data['series'])} loss points, "
            f"final_train_loss={format_number(data['series'][-1][1])}, "
            f"final_eval_loss={format_number(data['final_eval_loss'])}, "
            f"triggers={len(data['triggers'])}, detected={len(data['detected'])}, "
            f"max_finite_grad_pre={format_number(max_grad_value)}@step{max_grad_step:.0f}"
        )
    print(f"wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--clipped-metrics", type=Path, default=DEFAULT_CLIPPED)
    parser.add_argument("--no-clipping-metrics", type=Path, default=DEFAULT_NO_CLIPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-png", type=Path, default=Path("figures/iv_cd_fig2_clipping_comparison_preview.png"))
    parser.add_argument("--xmin", type=float, default=0.0)
    parser.add_argument("--xmax", type=float, default=1000.0)
    parser.add_argument("--ymin", type=float, default=4.0)
    parser.add_argument("--ymax", type=float, default=10.6)
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
