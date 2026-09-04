#!/usr/bin/env python3
"""Generate the FT2 report figures from the checked-in analysis tables.

The script intentionally reads only the three FT2 result directories listed in
``SOURCE_FILES``.  It uses rates already present in the JSON/CSV tables whenever
an aggregate rate is available.  The one pooled view (fault-model results) is
computed exactly as sum(SDC) / sum(planned) from ``per_cell.csv``.

Run from any directory inside the submitted report bundle:

    python code/analysis/make_report_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
except ImportError as exc:  # pragma: no cover - actionable error for other hosts
    raise SystemExit(
        "Pillow is required. Run this script with the Codex workspace Python "
        "runtime or install Pillow in the selected Python environment."
    ) from exc


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BUNDLE_ROOT
AUTHOR_DIR = BUNDLE_ROOT / "results" / "raw" / "main_18k_author_logic"
SCALING_DIR = BUNDLE_ROOT / "results" / "raw" / "scaling_factor"
CRITICAL_DIR = BUNDLE_ROOT / "results" / "raw" / "critical_layers"
DEFAULT_OUTPUT_DIR = BUNDLE_ROOT / "results" / "figures"

SOURCE_FILES = (
    AUTHOR_DIR / "per_cell.csv",
    AUTHOR_DIR / "summary.json",
    SCALING_DIR / "per_factor.csv",
    SCALING_DIR / "summary.json",
    CRITICAL_DIR / "summary.json",
    CRITICAL_DIR / "per_mode.csv",
    CRITICAL_DIR / "criticality.csv",
)

MODE_ORDER = (
    "no_protection",
    "paper_clamp_first_token_bounds",
    "paper_clamp_offline_bounds",
)
MODE_LABELS = {
    "no_protection": "No protection",
    "paper_clamp_first_token_bounds": "First-token\nbounds",
    "paper_clamp_offline_bounds": "Offline\nbounds",
}
MODE_SHORT_LABELS = {
    "no_protection": "No protection",
    "paper_clamp_first_token_bounds": "First-token bounds",
    "paper_clamp_offline_bounds": "Offline bounds",
}
MODE_COLORS = {
    "no_protection": "#D55E00",
    "paper_clamp_first_token_bounds": "#0072B2",
    "paper_clamp_offline_bounds": "#009E73",
}

PAIR_ORDER = (
    "opt_2_7b__squad_v2",
    "opt_2_7b__xtreme_mlqa_en_en",
    "qwen2_math_7b__gsm8k",
    "qwen2_math_7b__squad_v2",
    "qwen2_math_7b__xtreme_mlqa_en_en",
)
PAIR_LABELS = {
    "opt_2_7b__squad_v2": "OPT-2.7B · SQuAD v2",
    "opt_2_7b__xtreme_mlqa_en_en": "OPT-2.7B · MLQA en-en",
    "qwen2_math_7b__gsm8k": "Qwen2-Math-7B · GSM8K",
    "qwen2_math_7b__squad_v2": "Qwen2-Math-7B · SQuAD v2",
    "qwen2_math_7b__xtreme_mlqa_en_en": "Qwen2-Math-7B · MLQA en-en",
}

FAULT_ORDER = ("fp16_1bit", "fp16_2bit", "fp16_exponent_bit")
FAULT_LABELS = {
    "fp16_1bit": "FP16\n1-bit flip",
    "fp16_2bit": "FP16\n2-bit flip",
    "fp16_exponent_bit": "FP16\nexponent-bit flip",
}

INK = "#1F2937"
MUTED = "#5F6B7A"
GRID = "#D9E0E8"
LIGHT_GRID = "#EDF1F5"
BACKGROUND = "#FFFFFF"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#8E5EA2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_int(value: str | int | float) -> int:
    return int(float(value))


def as_float(value: str | int | float) -> float:
    return float(value)


def check_close(actual: float, expected: float, label: str, tol: float = 5e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise ValueError(f"Source-table mismatch for {label}: {actual} != {expected}")


def validate_sources(
    author_rows: Sequence[Mapping[str, str]],
    author_summary: Mapping,
    scaling_rows: Sequence[Mapping[str, str]],
    scaling_summary: Mapping,
    critical_rows: Sequence[Mapping[str, str]],
    critical_summary: Mapping,
    criticality_rows: Sequence[Mapping[str, str]],
) -> None:
    """Fail early if tables are incomplete or disagree with their summaries."""

    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing))

    expected_author_cells = {
        (pair, mode, fault)
        for pair in PAIR_ORDER
        for mode in MODE_ORDER
        for fault in FAULT_ORDER
    }
    actual_author_cells = {
        (row["pair_id"], row["mode_id"], row["fault_type"])
        for row in author_rows
    }
    if actual_author_cells != expected_author_cells:
        absent = sorted(expected_author_cells - actual_author_cells)
        extra = sorted(actual_author_cells - expected_author_cells)
        raise ValueError(f"Unexpected author per-cell coverage; missing={absent}, extra={extra}")

    for row in author_rows:
        planned = as_int(row["planned"])
        if planned <= 0:
            raise ValueError(f"Non-positive planned count in author row: {row}")
        check_close(
            as_float(row["sdc_rate"]),
            as_int(row["sdc"]) / planned,
            f"author per-cell rate {row['pair_id']}/{row['mode_id']}/{row['fault_type']}",
        )

    for mode in MODE_ORDER:
        rows = [row for row in author_rows if row["mode_id"] == mode]
        sdc = sum(as_int(row["sdc"]) for row in rows)
        planned = sum(as_int(row["planned"]) for row in rows)
        summary = author_summary["mode_totals"][mode]
        if sdc != int(summary["sdc"]) or planned != int(summary["evaluable"]):
            raise ValueError(f"Author mode totals disagree for {mode}")
        check_close(sdc / planned, float(summary["sdc_rate"]), f"author mode {mode}")

    for pair in PAIR_ORDER:
        for mode in MODE_ORDER:
            rows = [
                row
                for row in author_rows
                if row["pair_id"] == pair and row["mode_id"] == mode
            ]
            sdc = sum(as_int(row["sdc"]) for row in rows)
            planned = sum(as_int(row["planned"]) for row in rows)
            summary = author_summary["pair_mode_totals"][pair][mode]
            if sdc != int(summary["sdc"]) or planned != int(summary["evaluable"]):
                raise ValueError(f"Author pair-mode totals disagree for {pair}/{mode}")
            check_close(
                sdc / planned,
                float(summary["sdc_rate"]),
                f"author pair-mode {pair}/{mode}",
            )

    scaling_by_mode = {row["mode"]: row for row in scaling_rows}
    if set(scaling_by_mode) != set(scaling_summary["factor_results"]):
        raise ValueError("Scaling CSV modes do not match summary.json factor_results")
    for mode, summary in scaling_summary["factor_results"].items():
        row = scaling_by_mode[mode]
        check_close(as_float(row["sdc_rate"]), float(summary["sdc_rate"]), f"scaling {mode}")
        if as_int(row["sdc"]) != int(summary["sdc"]):
            raise ValueError(f"Scaling SDC count disagrees for {mode}")

    critical_by_mode = {row["mode"]: row for row in critical_rows}
    if set(critical_by_mode) != set(critical_summary["mode_results"]):
        raise ValueError("Critical-layer CSV modes do not match summary.json mode_results")
    for mode, summary in critical_summary["mode_results"].items():
        row = critical_by_mode[mode]
        check_close(as_float(row["sdc_rate"]), float(summary["sdc_rate"]), f"critical {mode}")
        if as_int(row["sdc"]) != int(summary["sdc"]):
            raise ValueError(f"Critical-layer SDC count disagrees for {mode}")

    expected_projections = {
        row["omitted_projection"] for row in critical_rows if row["omitted_projection"]
    }
    actual_projections = {row["projection"] for row in criticality_rows}
    if actual_projections != expected_projections:
        raise ValueError("criticality.csv projections do not match per_mode.csv omissions")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    windows_font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = (
        ("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf")
        if bold
        else ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf")
    )
    candidates: list[Path | str] = [windows_font_dir / name for name in names]
    candidates.extend(names)
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(50, bold=True)
F_SUBTITLE = font(27)
F_PANEL = font(30, bold=True)
F_AXIS = font(24)
F_TICK = font(22)
F_LABEL = font(25)
F_VALUE = font(24, bold=True)
F_VALUE_SMALL = font(20, bold=True)
F_FOOT = font(21)
F_LEGEND = font(22)


def text_size(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=face, spacing=4, align="center")
    return box[2] - box[0], box[3] - box[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    face: ImageFont.ImageFont,
    fill: str = INK,
    spacing: int = 4,
) -> None:
    width, height = text_size(draw, text, face)
    draw.multiline_text(
        (x - width / 2, y - height / 2),
        text,
        font=face,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def draw_right(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    face: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    width, height = text_size(draw, text, face)
    draw.text((x - width, y - height / 2), text, font=face, fill=fill)


def draw_header(
    draw: ImageDraw.ImageDraw,
    width: int,
    title: str,
    subtitle: str,
) -> None:
    draw_centered(draw, width / 2, 62, title, F_TITLE)
    draw_centered(draw, width / 2, 119, subtitle, F_SUBTITLE, fill=MUTED)


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def blend(start: str, end: str, amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    a = hex_rgb(start)
    b = hex_rgb(end)
    return tuple(round(x + (y - x) * amount) for x, y in zip(a, b))


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    unit = 10.0**exponent
    fraction = value / unit
    for step in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if fraction <= step:
            return step * unit
    return 10.0 * unit


def y_map(value: float, y_max: float, top: int, bottom: int) -> float:
    return bottom - (value / y_max) * (bottom - top)


def x_map(value: float, x_min: float, x_max: float, left: int, right: int) -> float:
    if math.isclose(x_min, x_max):
        return (left + right) / 2
    return left + (value - x_min) / (x_max - x_min) * (right - left)


def draw_y_grid(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    y_max: float,
    formatter: Callable[[float], str],
    tick_count: int = 5,
) -> None:
    left, top, right, bottom = box
    for index in range(tick_count + 1):
        value = y_max * index / tick_count
        y = y_map(value, y_max, top, bottom)
        draw.line((left, y, right, y), fill=GRID if index == 0 else LIGHT_GRID, width=2)
        draw_right(draw, left - 16, y, formatter(value), F_TICK, MUTED)
    draw.line((left, top, left, bottom), fill=GRID, width=2)
    draw.line((left, bottom, right, bottom), fill=GRID, width=2)


def draw_x_ticks(
    draw: ImageDraw.ImageDraw,
    values: Sequence[float],
    labels: Sequence[str],
    left: int,
    right: int,
    y: int,
    x_min: float | None = None,
    x_max: float | None = None,
) -> list[float]:
    if x_min is None:
        x_min = min(values)
    if x_max is None:
        x_max = max(values)
    positions = [x_map(value, x_min, x_max, left, right) for value in values]
    for x, label in zip(positions, labels):
        draw.line((x, y, x, y + 9), fill=GRID, width=2)
        draw_centered(draw, x, y + 28, label, F_TICK, MUTED)
    return positions


def draw_dashed_vertical(
    draw: ImageDraw.ImageDraw,
    x: float,
    top: float,
    bottom: float,
    fill: str,
    width: int = 3,
    dash: int = 11,
    gap: int = 8,
) -> None:
    y = top
    while y < bottom:
        draw.line((x, y, x, min(y + dash, bottom)), fill=fill, width=width)
        y += dash + gap


def save_figure(
    image: Image.Image,
    output_dir: Path,
    filename: str,
    source_files: Iterable[Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Title", filename.removesuffix(".png").replace("_", " "))
    metadata.add_text("Generator", str(Path(__file__).resolve()))
    metadata.add_text(
        "Data sources",
        "; ".join(str(path.relative_to(PROJECT_ROOT)) for path in source_files),
    )
    image.save(destination, format="PNG", dpi=(180, 180), pnginfo=metadata, optimize=True)
    return destination


def make_overall_sdc(author_summary: Mapping, output_dir: Path) -> Path:
    width, height = 1600, 920
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    total = int(author_summary["overall"]["evaluable"])
    draw_header(
        draw,
        width,
        "Overall SDC rate by protection mode",
        f"Author-logic rescoring · {total:,} evaluable fault runs",
    )

    rows = [author_summary["mode_totals"][mode] for mode in MODE_ORDER]
    rates = [float(row["sdc_rate"]) for row in rows]
    y_max = nice_upper(max(rates) * 1.18)
    plot = (180, 225, 1500, 735)
    draw_y_grid(draw, plot, y_max, lambda value: fmt_pct(value, 1), tick_count=6)

    left, top, right, bottom = plot
    centers = [left + (index + 0.5) * (right - left) / len(MODE_ORDER) for index in range(3)]
    bar_width = 245
    for x, mode, row, rate in zip(centers, MODE_ORDER, rows, rates):
        bar_top = y_map(rate, y_max, top, bottom)
        draw.rounded_rectangle(
            (x - bar_width / 2, bar_top, x + bar_width / 2, bottom),
            radius=14,
            fill=MODE_COLORS[mode],
        )
        value = f"{fmt_pct(rate)}\n{int(row['sdc']):,} / {int(row['evaluable']):,}"
        draw_centered(draw, x, bar_top - 44, value, F_VALUE, INK)
        draw_centered(draw, x, bottom + 58, MODE_LABELS[mode], F_LABEL, INK, spacing=2)

    draw_centered(draw, 55, (top + bottom) / 2, "SDC\nrate", F_AXIS, MUTED, spacing=1)
    draw_centered(
        draw,
        width / 2,
        875,
        "Bar labels show SDC count / evaluable runs; rates are read from summary.json.",
        F_FOOT,
        MUTED,
    )
    return save_figure(
        image,
        output_dir,
        "overall_sdc_by_mode.png",
        (AUTHOR_DIR / "summary.json",),
    )


def make_pair_heatmap(author_summary: Mapping, output_dir: Path) -> Path:
    width, height = 1800, 1040
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        width,
        "SDC rate by model–task pair",
        "Protection modes compared on identical fault-model coverage",
    )

    matrix = [
        [float(author_summary["pair_mode_totals"][pair][mode]["sdc_rate"]) for mode in MODE_ORDER]
        for pair in PAIR_ORDER
    ]
    maximum = max(max(row) for row in matrix)
    label_left = 70
    grid_left, grid_top = 465, 240
    grid_right, grid_bottom = 1710, 815
    cell_width = (grid_right - grid_left) / len(MODE_ORDER)
    cell_height = (grid_bottom - grid_top) / len(PAIR_ORDER)

    for column, mode in enumerate(MODE_ORDER):
        draw_centered(
            draw,
            grid_left + (column + 0.5) * cell_width,
            grid_top - 58,
            MODE_LABELS[mode],
            F_LABEL,
            INK,
            spacing=2,
        )

    for row_index, pair in enumerate(PAIR_ORDER):
        y0 = grid_top + row_index * cell_height
        y1 = y0 + cell_height
        _, label_height = text_size(draw, PAIR_LABELS[pair], F_LABEL)
        draw.text(
            (label_left, (y0 + y1) / 2 - label_height / 2),
            PAIR_LABELS[pair],
            font=F_LABEL,
            fill=INK,
        )
        for column, mode in enumerate(MODE_ORDER):
            x0 = grid_left + column * cell_width
            x1 = x0 + cell_width
            entry = author_summary["pair_mode_totals"][pair][mode]
            rate = float(entry["sdc_rate"])
            intensity = 0.0 if maximum == 0 else rate / maximum
            fill = blend("#F2F7FC", "#07568A", intensity)
            draw.rectangle((x0 + 3, y0 + 3, x1 - 3, y1 - 3), fill=fill)
            text_fill = BACKGROUND if intensity >= 0.57 else INK
            label = f"{fmt_pct(rate)}\n{int(entry['sdc'])}/{int(entry['evaluable'])}"
            draw_centered(draw, (x0 + x1) / 2, (y0 + y1) / 2, label, F_VALUE, text_fill)

    # Compact continuous legend.
    legend_left, legend_right, legend_y = 670, 1490, 900
    steps = 164
    for step in range(steps):
        x0 = legend_left + step * (legend_right - legend_left) / steps
        x1 = legend_left + (step + 1) * (legend_right - legend_left) / steps
        draw.rectangle(
            (x0, legend_y, x1 + 1, legend_y + 24),
            fill=blend("#F2F7FC", "#07568A", step / (steps - 1)),
        )
    draw_right(draw, legend_left - 16, legend_y + 12, "SDC rate", F_LEGEND, MUTED)
    draw_centered(draw, legend_left, legend_y + 53, fmt_pct(0.0), F_TICK, MUTED)
    draw_centered(draw, (legend_left + legend_right) / 2, legend_y + 53, fmt_pct(maximum / 2), F_TICK, MUTED)
    draw_centered(draw, legend_right, legend_y + 53, fmt_pct(maximum), F_TICK, MUTED)
    draw_centered(
        draw,
        width / 2,
        1002,
        "Each cell pools the three listed FP16 fault models; labels show SDC / evaluable runs.",
        F_FOOT,
        MUTED,
    )
    return save_figure(
        image,
        output_dir,
        "model_task_sdc_heatmap.png",
        (AUTHOR_DIR / "summary.json",),
    )


def make_scaling_dual(
    scaling_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> Path:
    width, height = 1900, 1030
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        width,
        "Scaling-factor ablation: SDC and correction workload",
        "Qwen2-Math-7B · GSM8K · FP16 exponent-bit faults · 400 runs per condition",
    )

    factor_rows = sorted(
        (row for row in scaling_rows if row["scaling_factor"].strip()),
        key=lambda row: as_float(row["scaling_factor"]),
    )
    baseline = next(row for row in scaling_rows if row["mode"] == "no_protection")
    factors = [as_float(row["scaling_factor"]) for row in factor_rows]
    factor_labels = [f"{factor:.2f}" for factor in factors]

    left_plot = (150, 270, 870, 790)
    right_plot = (1110, 270, 1830, 790)
    draw_centered(draw, 510, 185, "A · SDC rate (Wilson 95% CI)", F_PANEL)
    draw_centered(
        draw,
        510,
        226,
        f"No-protection reference: {fmt_pct(as_float(baseline['sdc_rate']))} "
        f"({baseline['sdc']}/{baseline['runs']}; outside panel scale)",
        F_FOOT,
        MUTED,
    )
    draw_centered(draw, 1470, 185, "B · Mean corrected elements per run", F_PANEL)

    ci_highs = [as_float(row["wilson95_high"]) for row in factor_rows]
    sdc_ymax = nice_upper(max(ci_highs) * 1.12)
    draw_y_grid(draw, left_plot, sdc_ymax, lambda value: fmt_pct(value, 1), tick_count=5)
    x_positions = draw_x_ticks(
        draw,
        factors,
        factor_labels,
        left_plot[0],
        left_plot[2],
        left_plot[3],
    )
    draw_centered(draw, (left_plot[0] + left_plot[2]) / 2, 880, "Scaling factor", F_AXIS, MUTED)

    rate_points: list[tuple[float, float]] = []
    for x, row in zip(x_positions, factor_rows):
        rate = as_float(row["sdc_rate"])
        low = as_float(row["wilson95_low"])
        high = as_float(row["wilson95_high"])
        y = y_map(rate, sdc_ymax, left_plot[1], left_plot[3])
        y_low = y_map(low, sdc_ymax, left_plot[1], left_plot[3])
        y_high = y_map(high, sdc_ymax, left_plot[1], left_plot[3])
        rate_points.append((x, y))
        draw.line((x, y_high, x, y_low), fill=BLUE, width=4)
        draw.line((x - 10, y_high, x + 10, y_high), fill=BLUE, width=4)
        draw.line((x - 10, y_low, x + 10, y_low), fill=BLUE, width=4)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=BLUE)
        draw_centered(draw, x, max(left_plot[1] + 20, y_high - 28), fmt_pct(rate), F_VALUE_SMALL, INK)
    if len(rate_points) > 1:
        draw.line(rate_points, fill=BLUE, width=4, joint="curve")

    fault_values = [as_float(row["mean_fault_corrections"]) for row in factor_rows]
    clean_values = [as_float(row["mean_clean_corrections"]) for row in factor_rows]
    correction_ymax = nice_upper(max(fault_values + clean_values) * 1.08)
    draw_y_grid(
        draw,
        right_plot,
        correction_ymax,
        lambda value: f"{value:,.0f}",
        tick_count=5,
    )
    right_positions = draw_x_ticks(
        draw,
        factors,
        factor_labels,
        right_plot[0],
        right_plot[2],
        right_plot[3],
    )
    draw_centered(draw, (right_plot[0] + right_plot[2]) / 2, 880, "Scaling factor", F_AXIS, MUTED)

    fault_points = [
        (x, y_map(value, correction_ymax, right_plot[1], right_plot[3]))
        for x, value in zip(right_positions, fault_values)
    ]
    clean_points = [
        (x, y_map(value, correction_ymax, right_plot[1], right_plot[3]))
        for x, value in zip(right_positions, clean_values)
    ]
    draw.line(fault_points, fill=BLUE, width=5, joint="curve")
    draw.line(clean_points, fill=GREEN, width=5, joint="curve")
    for (x, fault_y), (_, clean_y), fault_value, clean_value in zip(
        fault_points, clean_points, fault_values, clean_values
    ):
        draw.ellipse((x - 8, fault_y - 8, x + 8, fault_y + 8), fill=BLUE)
        draw.ellipse((x - 8, clean_y - 8, x + 8, clean_y + 8), fill=GREEN)
        draw_centered(draw, x, fault_y - 23, f"{fault_value:,.0f}", F_VALUE_SMALL, BLUE)
        draw_centered(draw, x, clean_y + 25, f"{clean_value:,.0f}", F_VALUE_SMALL, GREEN)

    legend_y = 232
    draw.line((1260, legend_y, 1300, legend_y), fill=BLUE, width=5)
    draw.ellipse((1272, legend_y - 8, 1288, legend_y + 8), fill=BLUE)
    draw.text((1312, legend_y - 15), "Fault run", font=F_LEGEND, fill=INK)
    draw.line((1510, legend_y, 1550, legend_y), fill=GREEN, width=5)
    draw.ellipse((1522, legend_y - 8, 1538, legend_y + 8), fill=GREEN)
    draw.text((1562, legend_y - 15), "Clean control", font=F_LEGEND, fill=INK)

    draw_centered(
        draw,
        width / 2,
        985,
        "All rates, confidence limits, and correction means are read directly from per_factor.csv.",
        F_FOOT,
        MUTED,
    )
    return save_figure(
        image,
        output_dir,
        "scaling_factor_sdc_and_corrections.png",
        (SCALING_DIR / "per_factor.csv", SCALING_DIR / "summary.json"),
    )


def make_critical_omission(
    critical_rows: Sequence[Mapping[str, str]],
    criticality_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> Path:
    width, height = 1900, 1070
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        width,
        "SDC after omitting each projection from protection",
        "Qwen2-Math-7B · GSM8K · FP16 exponent-bit faults · 400 paired runs per mode",
    )

    by_mode = {row["mode"]: row for row in critical_rows}
    criticality = {row["projection"]: row for row in criticality_rows}
    ordered = [by_mode["protect_all_linear"]]
    ordered.extend(by_mode[f"leave_{row['projection']}_unprotected"] for row in criticality_rows)

    plot_left, plot_top, plot_right, plot_bottom = 470, 250, 1540, 865
    maximum_ci = max(as_float(row["wilson95_high"]) for row in ordered)
    x_max = nice_upper(maximum_ci * 1.10)
    tick_count = 4
    for index in range(tick_count + 1):
        value = x_max * index / tick_count
        x = x_map(value, 0.0, x_max, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill=GRID if index == 0 else LIGHT_GRID, width=2)
        draw_centered(draw, x, plot_bottom + 33, fmt_pct(value, 1), F_TICK, MUTED)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRID, width=2)

    baseline_rate = as_float(by_mode["protect_all_linear"]["sdc_rate"])
    baseline_x = x_map(baseline_rate, 0.0, x_max, plot_left, plot_right)
    draw_dashed_vertical(draw, baseline_x, plot_top, plot_bottom, MUTED, width=3)
    draw.text(
        (baseline_x + 10, plot_top - 36),
        f"Protect-all rate: {fmt_pct(baseline_rate)}",
        font=F_FOOT,
        fill=MUTED,
    )

    row_gap = (plot_bottom - plot_top) / len(ordered)
    for index, row in enumerate(ordered):
        y = plot_top + (index + 0.5) * row_gap
        projection = row["omitted_projection"]
        if projection:
            label = projection
            is_paper_expected = criticality[projection]["paper_expected_critical"].lower() == "true"
            color = ORANGE if is_paper_expected else BLUE
            marker = "circle"
        else:
            label = "Protect all linear"
            color = INK
            marker = "diamond"
        draw_right(draw, plot_left - 30, y, label, F_LABEL, INK)
        low = as_float(row["wilson95_low"])
        high = as_float(row["wilson95_high"])
        rate = as_float(row["sdc_rate"])
        x_low = x_map(low, 0.0, x_max, plot_left, plot_right)
        x_high = x_map(high, 0.0, x_max, plot_left, plot_right)
        x = x_map(rate, 0.0, x_max, plot_left, plot_right)
        draw.line((x_low, y, x_high, y), fill=color, width=5)
        draw.line((x_low, y - 10, x_low, y + 10), fill=color, width=4)
        draw.line((x_high, y - 10, x_high, y + 10), fill=color, width=4)
        if marker == "diamond":
            draw.polygon(((x, y - 11), (x + 11, y), (x, y + 11), (x - 11, y)), fill=color)
        else:
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)
        draw.text(
            (plot_right + 28, y - 18),
            f"{fmt_pct(rate)}  ({row['sdc']}/{row['runs']})",
            font=F_VALUE_SMALL,
            fill=INK,
        )

    draw_centered(draw, (plot_left + plot_right) / 2, 942, "SDC rate (Wilson 95% CI)", F_AXIS, MUTED)

    legend_y = 992
    draw.ellipse((420, legend_y - 9, 438, legend_y + 9), fill=ORANGE)
    draw.text((451, legend_y - 14), "Paper-expected critical omission", font=F_LEGEND, fill=INK)
    draw.ellipse((850, legend_y - 9, 868, legend_y + 9), fill=BLUE)
    draw.text((881, legend_y - 14), "Other omission", font=F_LEGEND, fill=INK)
    draw.polygon(
        ((1160, legend_y - 10), (1170, legend_y), (1160, legend_y + 10), (1150, legend_y)),
        fill=INK,
    )
    draw.text((1182, legend_y - 14), "Protect-all reference", font=F_LEGEND, fill=INK)

    return save_figure(
        image,
        output_dir,
        "critical_layer_omission_sdc.png",
        (
            CRITICAL_DIR / "per_mode.csv",
            CRITICAL_DIR / "criticality.csv",
            CRITICAL_DIR / "summary.json",
        ),
    )


def aggregate_fault_models(
    author_rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str], dict[str, float | int]]:
    """Pool exact SDC/planned counts across the five model-task pairs."""

    pooled: dict[tuple[str, str], dict[str, float | int]] = {}
    for fault in FAULT_ORDER:
        for mode in MODE_ORDER:
            rows = [
                row
                for row in author_rows
                if row["fault_type"] == fault and row["mode_id"] == mode
            ]
            if len(rows) != len(PAIR_ORDER):
                raise ValueError(f"Incomplete fault-model aggregation for {fault}/{mode}")
            sdc = sum(as_int(row["sdc"]) for row in rows)
            planned = sum(as_int(row["planned"]) for row in rows)
            pooled[(fault, mode)] = {
                "sdc": sdc,
                "planned": planned,
                "sdc_rate": sdc / planned,
            }
    return pooled


def make_fault_model_results(
    author_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> Path:
    width, height = 1850, 1000
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    pooled = aggregate_fault_models(author_rows)
    denominator = int(pooled[(FAULT_ORDER[0], MODE_ORDER[0])]["planned"])
    draw_header(
        draw,
        width,
        "SDC rate by injected fault model",
        f"Exact pooling across five model–task pairs · {denominator:,} planned runs per bar",
    )

    rates = [float(pooled[(fault, mode)]["sdc_rate"]) for fault in FAULT_ORDER for mode in MODE_ORDER]
    y_max = nice_upper(max(rates) * 1.17)
    plot = (175, 265, 1765, 760)
    draw_y_grid(draw, plot, y_max, lambda value: fmt_pct(value, 1), tick_count=6)
    left, top, right, bottom = plot
    group_width = (right - left) / len(FAULT_ORDER)
    bar_width = 115
    within_gap = 18
    cluster_width = len(MODE_ORDER) * bar_width + (len(MODE_ORDER) - 1) * within_gap

    for fault_index, fault in enumerate(FAULT_ORDER):
        group_center = left + (fault_index + 0.5) * group_width
        first_x = group_center - cluster_width / 2 + bar_width / 2
        for mode_index, mode in enumerate(MODE_ORDER):
            entry = pooled[(fault, mode)]
            rate = float(entry["sdc_rate"])
            x = first_x + mode_index * (bar_width + within_gap)
            bar_top = y_map(rate, y_max, top, bottom)
            draw.rounded_rectangle(
                (x - bar_width / 2, bar_top, x + bar_width / 2, bottom),
                radius=10,
                fill=MODE_COLORS[mode],
            )
            label_y = max(top + 35, bar_top - 40)
            draw_centered(
                draw,
                x,
                label_y,
                f"{fmt_pct(rate)}\n{int(entry['sdc'])}/{int(entry['planned'])}",
                F_VALUE_SMALL,
                INK,
            )
        draw_centered(draw, group_center, bottom + 61, FAULT_LABELS[fault], F_LABEL, INK, spacing=2)

    legend_y = 205
    legend_x = 470
    for mode in MODE_ORDER:
        draw.rounded_rectangle((legend_x, legend_y - 10, legend_x + 34, legend_y + 10), radius=5, fill=MODE_COLORS[mode])
        draw.text((legend_x + 46, legend_y - 15), MODE_SHORT_LABELS[mode], font=F_LEGEND, fill=INK)
        legend_x += 355 if mode == "no_protection" else 400

    draw_centered(draw, 55, (top + bottom) / 2, "SDC\nrate", F_AXIS, MUTED, spacing=1)
    draw_centered(
        draw,
        width / 2,
        955,
        "Pooled rates are computed only from per_cell.csv as sum(SDC) / sum(planned); no values are imputed.",
        F_FOOT,
        MUTED,
    )
    return save_figure(
        image,
        output_dir,
        "fault_model_sdc_results.png",
        (AUTHOR_DIR / "per_cell.csv",),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for source in SOURCE_FILES:
        if not source.is_file():
            raise FileNotFoundError(f"Required source file is missing: {source}")

    author_rows = read_csv(AUTHOR_DIR / "per_cell.csv")
    author_summary = read_json(AUTHOR_DIR / "summary.json")
    scaling_rows = read_csv(SCALING_DIR / "per_factor.csv")
    scaling_summary = read_json(SCALING_DIR / "summary.json")
    critical_rows = read_csv(CRITICAL_DIR / "per_mode.csv")
    critical_summary = read_json(CRITICAL_DIR / "summary.json")
    criticality_rows = read_csv(CRITICAL_DIR / "criticality.csv")

    validate_sources(
        author_rows,
        author_summary,
        scaling_rows,
        scaling_summary,
        critical_rows,
        critical_summary,
        criticality_rows,
    )

    output_dir = args.output_dir.resolve()
    outputs = [
        make_overall_sdc(author_summary, output_dir),
        make_pair_heatmap(author_summary, output_dir),
        make_scaling_dual(scaling_rows, output_dir),
        make_critical_omission(critical_rows, criticality_rows, output_dir),
        make_fault_model_results(author_rows, output_dir),
    ]
    print("Generated FT2 report figures:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
