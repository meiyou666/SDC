#!/usr/bin/env python3
"""Render the paper-style Figure 3 reproduction for IV-F duration=3."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


DEFAULT_METRICS = Path("checkpoints/iv_f_duration_all_hmma_mb256/BP8/duration/steps_3/seed_42/metrics.jsonl")
DEFAULT_OUTPUT = Path("figures/iv_f_fig3_paper_style.pdf")
DEFAULT_PREVIEW = Path("figures/iv_f_fig3_paper_style_preview.png")
DEFAULT_COMPAT_OUTPUT = Path("figures/iv_f_fig3_timeseries_only.pdf")
DEFAULT_COMPAT_PREVIEW = Path("figures/iv_f_fig3_timeseries_only_preview.png")


def parse_number(value):
    if value is None or isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return math.nan


def load_step_records(path):
    steps = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            step = record.get("_step")
            if not isinstance(step, int):
                continue
            merged = steps.setdefault(step, {"_step": step})
            merged.update(record)
    return [steps[step] for step in sorted(steps)]


def series(records, key):
    points = []
    for record in records:
        value = parse_number(record.get(key))
        if math.isfinite(value):
            points.append((record["_step"], value))
    return points


def window(points, start, end):
    return [(x_value, y_value) for x_value, y_value in points if start <= x_value <= end]


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.2, width=0.8)
    ax.grid(False)


def configure_rt_axis(ax):
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=False)
    ax.yaxis.get_offset_text().set_size(8)


def draw_panel(ax, points, title, ylabel, color, ylim, yticks, inset_ylim, inset_yticks, args, is_rt=False):
    x_values = [x_value for x_value, _ in points]
    y_values = [y_value for _, y_value in points]
    ax.plot(x_values, y_values, color=color, linewidth=1.25)
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9.5, pad=3)
    style_axis(ax)
    if is_rt:
        configure_rt_axis(ax)

    inset = inset_axes(ax, width="38%", height="42%", loc="upper right", borderpad=0.75)
    inset_points = window(points, args.inset_xmin, args.inset_xmax)
    inset.plot([x_value for x_value, _ in inset_points], [y_value for _, y_value in inset_points], color=color, linewidth=1.15)
    inset.set_xlim(args.inset_xmin, args.inset_xmax)
    inset.set_ylim(*inset_ylim)
    inset.set_xticks([480, 500, 520, 540])
    inset.set_yticks(inset_yticks)
    inset.tick_params(direction="out", length=2.6, width=0.7, labelsize=7)
    inset.grid(False)
    if is_rt:
        configure_rt_axis(inset)
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)

    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.55)
    return inset


def render(args):
    root = args.root.resolve()
    metrics_path = root / args.metrics
    output_paths = [root / args.output]
    preview_paths = [root / args.preview_png] if args.preview_png else []
    if args.compat_output:
        output_paths.append(root / args.compat_output)
    if args.compat_preview_png:
        preview_paths.append(root / args.compat_preview_png)

    records = load_step_records(metrics_path)
    loss_points = series(records, "loss")
    rt_points = series(records, "rt/rt")
    if not loss_points or not rt_points:
        raise RuntimeError(f"Missing loss or rt/rt points in {metrics_path}")

    for path in output_paths + preview_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.35), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.965, bottom=0.11, hspace=0.22)

    draw_panel(
        axes[0],
        loss_points,
        "Training Loss",
        "Training Loss",
        "#1f77b4",
        ylim=(4.0, 10.8),
        yticks=[4, 6, 8, 10],
        inset_ylim=(5.0, 5.95),
        inset_yticks=[5.25, 5.50, 5.75],
        args=args,
        is_rt=False,
    )
    draw_panel(
        axes[1],
        rt_points,
        r"$R_t$",
        r"$R_t$",
        "#1f77b4",
        ylim=(0.0, 0.0020),
        yticks=[0.0, 0.0005, 0.0010, 0.0015, 0.0020],
        inset_ylim=(0.00025, 0.00125),
        inset_yticks=[0.0005, 0.0010],
        args=args,
        is_rt=True,
    )

    axes[1].set_xlabel("Step")
    axes[1].set_xticks([0, 200, 400, 600, 800, 1000])
    for ax in axes:
        ax.set_xlim(args.xmin, args.xmax)

    for output_path in output_paths:
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"wrote {output_path}")
    for preview_path in preview_paths:
        fig.savefig(preview_path, format="png", dpi=240, bbox_inches="tight")
        print(f"wrote {preview_path}")
    plt.close(fig)

    loss_window = window(loss_points, args.inset_xmin, args.inset_xmax)
    rt_window = window(rt_points, args.inset_xmin, args.inset_xmax)
    print(f"loss_points={len(loss_points)} rt_points={len(rt_points)}")
    print(f"inset_window={args.inset_xmin:.0f}-{args.inset_xmax:.0f}")
    print(f"inset_loss_max={max(value for _, value in loss_window):.6f}")
    print(f"inset_rt_max={max(value for _, value in rt_window):.6e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-png", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--compat-output", type=Path, default=DEFAULT_COMPAT_OUTPUT)
    parser.add_argument("--compat-preview-png", type=Path, default=DEFAULT_COMPAT_PREVIEW)
    parser.add_argument("--xmin", type=float, default=0.0)
    parser.add_argument("--xmax", type=float, default=1000.0)
    parser.add_argument("--inset-xmin", type=float, default=480.0)
    parser.add_argument("--inset-xmax", type=float, default=540.0)
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
