#!/usr/bin/env python3
"""Render a Figure-1-style IV-B bit/kernel sensitivity table as PDF."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


BITS = [10, 13, 14]
KERNELS = ["FP1", "FP2", "FP3", "FP4", "BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7", "BP8", "BP9"]


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


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def metric_extrema(run_dir):
    records = load_jsonl(run_dir / "metrics.jsonl")
    result = {
        "max_gradient_norm_pre": math.nan,
        "has_nonfinite_gradient_norm_pre": False,
        "max_attention_logits": math.nan,
        "has_nonfinite_attention_logits": False,
    }
    grad_values = []
    attn_values = []
    for record in records:
        if "gradient_norm_pre" in record:
            value = parse_number(record["gradient_norm_pre"])
            if math.isfinite(value):
                grad_values.append(value)
            else:
                result["has_nonfinite_gradient_norm_pre"] = True
        if "max_attention_logits" in record:
            value = parse_number(record["max_attention_logits"])
            if math.isfinite(value):
                attn_values.append(value)
            else:
                result["has_nonfinite_attention_logits"] = True
    if grad_values:
        result["max_gradient_norm_pre"] = max(grad_values)
    if attn_values:
        result["max_attention_logits"] = max(attn_values)
    return result


def load_rows(root):
    rows = {}
    for bit in BITS:
        csv_path = root / "logs" / "iv_b_mb256" / "seed_42" / f"bit_{bit}" / "iv_b_summary_raw.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                run_dir = root / row["run_dir"]
                extrema = metric_extrema(run_dir)
                merged = dict(row)
                merged.update(extrema)
                rows[(int(row["bit_position"]), row["kernel_label"])] = merged
    return rows


def positive_delta(row):
    delta = parse_number(row["eval_loss_delta"])
    if not math.isfinite(delta):
        return math.nan
    return max(delta, 0.0)


def log_value(value):
    if not math.isfinite(value) or value <= 0:
        return math.nan
    return math.log10(value)


def format_delta(row):
    final_loss = parse_number(row["final_eval_loss"])
    delta = parse_number(row["eval_loss_delta"])
    if not math.isfinite(final_loss):
        return "NaN"
    return f"{delta:+.3f}"


def format_param(row):
    value = parse_number(row["final_parameter_difference"])
    if not math.isfinite(value):
        return "NaN"
    return f"{value:.0f}"


def format_scientific(value):
    if not math.isfinite(value):
        return "Inf"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-2):
        return f"{value:.1e}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.2g}"


def build_panel_specs(rows):
    max_eval = max(positive_delta(row) for row in rows.values() if math.isfinite(positive_delta(row)))
    finite_params = [parse_number(row["final_parameter_difference"]) for row in rows.values()]
    finite_params = [value for value in finite_params if math.isfinite(value)]
    grad_logs = []
    attn_logs = []
    for row in rows.values():
        if not row["has_nonfinite_gradient_norm_pre"]:
            value = log_value(row["max_gradient_norm_pre"])
            if math.isfinite(value):
                grad_logs.append(value)
        if not row["has_nonfinite_attention_logits"]:
            value = log_value(row["max_attention_logits"])
            if math.isfinite(value):
                attn_logs.append(value)

    return [
        {
            "title": "Final evaluation loss delta vs baseline",
            "subtitle": "cell text: final_loss - baseline; NaN = final evaluation produced NaN",
            "cmap": LinearSegmentedColormap.from_list("eval", ["#f7f7f7", "#ffe08a", "#f59e0b", "#b91c1c"]),
            "norm": Normalize(vmin=0.0, vmax=max_eval),
            "value": lambda row: positive_delta(row),
            "text": format_delta,
            "nan_if": lambda row: not math.isfinite(parse_number(row["final_eval_loss"])),
        },
        {
            "title": "Final parameter difference",
            "subtitle": "cell text: L2 distance to paired baseline checkpoint",
            "cmap": LinearSegmentedColormap.from_list("param", ["#f7f7f7", "#bfdbfe", "#3b82f6", "#1e3a8a"]),
            "norm": Normalize(vmin=min(finite_params), vmax=max(finite_params)),
            "value": lambda row: parse_number(row["final_parameter_difference"]),
            "text": format_param,
            "nan_if": lambda row: not math.isfinite(parse_number(row["final_parameter_difference"])),
        },
        {
            "title": "Maximum gradient norm before clipping",
            "subtitle": "cell text: max over training span; Inf marks non-finite gradient norm observed",
            "cmap": LinearSegmentedColormap.from_list("grad", ["#f7f7f7", "#bbf7d0", "#22c55e", "#166534"]),
            "norm": Normalize(vmin=min(grad_logs), vmax=max(grad_logs)),
            "value": lambda row: log_value(row["max_gradient_norm_pre"]),
            "text": lambda row: "Inf" if row["has_nonfinite_gradient_norm_pre"] else format_scientific(row["max_gradient_norm_pre"]),
            "nan_if": lambda row: row["has_nonfinite_gradient_norm_pre"],
        },
        {
            "title": "Maximum attention logits",
            "subtitle": "cell text: max over training span; Inf marks non-finite attention logits observed",
            "cmap": LinearSegmentedColormap.from_list("attn", ["#f7f7f7", "#ddd6fe", "#8b5cf6", "#4c1d95"]),
            "norm": Normalize(vmin=min(attn_logs), vmax=max(attn_logs)),
            "value": lambda row: log_value(row["max_attention_logits"]),
            "text": lambda row: "Inf" if row["has_nonfinite_attention_logits"] else format_scientific(row["max_attention_logits"]),
            "nan_if": lambda row: row["has_nonfinite_attention_logits"],
        },
    ]


def text_color(facecolor):
    red, green, blue, _ = facecolor
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.42 else "#111827"


def draw_panel(ax, rows, spec, show_x_labels):
    ax.set_xlim(0, len(KERNELS))
    ax.set_ylim(0, len(BITS))
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(len(KERNELS))])
    ax.set_xticklabels(KERNELS if show_x_labels else [], fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(BITS))])
    ax.set_yticklabels([f"bit {bit}" for bit in BITS], fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for col, kernel in enumerate(KERNELS):
        for row_idx, bit in enumerate(BITS):
            row = rows[(bit, kernel)]
            is_special = spec["nan_if"](row)
            if is_special:
                facecolor = "#7f1d1d" if "NaN" in spec["text"](row) or "Inf" in spec["text"](row) else "#374151"
            else:
                value = spec["value"](row)
                facecolor = spec["cmap"](spec["norm"](value)) if math.isfinite(value) else "#e5e7eb"
            rect = Rectangle((col, row_idx), 1, 1, facecolor=facecolor, edgecolor="white", linewidth=1.2)
            ax.add_patch(rect)
            text = spec["text"](row)
            ax.text(
                col + 0.5,
                row_idx + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color(matplotlib.colors.to_rgba(facecolor)),
                fontweight="bold" if text in {"NaN", "Inf"} else "normal",
            )

    ax.axvline(4, color="#111827", linewidth=1.4)
    ax.text(2, -0.2, "Forward kernels", ha="center", va="bottom", fontsize=8, color="#374151")
    ax.text(8.5, -0.2, "Backward kernels", ha="center", va="bottom", fontsize=8, color="#374151")
    ax.set_title(spec["title"], loc="left", fontsize=11, fontweight="bold", pad=48)
    ax.text(0, 1.32, spec["subtitle"], transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color="#4b5563")


def render_pdf(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    specs = build_panel_specs(rows)
    with PdfPages(output) as pdf:
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(13.2, 11.2), constrained_layout=False)
        fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.1, hspace=1.28)
        fig.suptitle("Bit / Kernel Sensitivity Table (IV-B Local Reproduction)", fontsize=15, fontweight="bold", y=0.975)
        fig.text(
            0.065,
            0.94,
            "Source: logs/iv_b_mb256/seed_42/bit_{10,13,14}/iv_b_summary_raw.csv and matching checkpoints metrics.jsonl. Baseline final eval loss = 4.317315.",
            fontsize=8.5,
            color="#374151",
        )
        for index, (ax, spec) in enumerate(zip(axes, specs)):
            draw_panel(ax, rows, spec, show_x_labels=index == len(specs) - 1)
        fig.text(
            0.065,
            0.025,
            "Notes: FP = forward-pass profiled GEMM kernel; BP = backward-pass profiled GEMM kernel. "
            "Bit 14 is a strong saturation case; NaN/Inf cells should not be interpreted as fine-grained kernel ranking.",
            fontsize=8,
            color="#374151",
        )
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("figures/iv_b_bit_kernel_sensitivity_fig1.pdf"))
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    rows = load_rows(root)
    render_pdf(rows, output)
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
