#!/usr/bin/env python3
"""Paired FT2 hook-overhead benchmark for Qwen2-Math-7B on GSM8K.

This intentionally compares two modes *inside* the reproduction hook framework:
``no_protection`` and paper-clamp first-token bounds at scaling factor 2.0.  It
does not measure a native Transformers/no-hook baseline, so its result must not
be described as native end-to-end overhead.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import io
import os
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
MODEL_KEY = "qwen2_math_7b"
DATASET_KEY = "gsm8k"
PAIR_ID = f"{MODEL_KEY}__{DATASET_KEY}"
EXPECTED_PROMPTS = 10
MAX_INPUT_TOKENS = 1024
GENERATION_STEPS = 180
SEED = 196
NO_PROTECTION = "no_protection"
FIRST_TOKEN = "first_token_s2p00"
MODE_IDS = (NO_PROTECTION, FIRST_TOKEN)
SCOPE_STATEMENT = (
    "Incremental cost of paper-clamp first-token bounds (factor 2.0) relative "
    "to FT2HookEngine no_protection, with the reproduction hooks installed in "
    "both conditions. This is not a native Transformers/no-hook or native "
    "end-to-end overhead measurement."
)

# Keep the benchmark offline, matching the frozen reproduction campaign.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _candidate_reproduction_sources() -> Iterable[Path]:
    configured = os.environ.get("FT2_REPRODUCTION_ROOT")
    if configured:
        root = Path(configured).expanduser()
        yield root if root.name == "src" else root / "src"
    # Report bundle layout: code/experiments (this file), code/reproduction/src.
    yield SCRIPT_PATH.parents[1] / "reproduction" / "src"
    # Repository layout: reproduction/experiments, reproduction/src.
    yield SCRIPT_PATH.parents[1] / "src"
    yield SCRIPT_PATH.parents[2] / "reproduction" / "src"
    yield Path.cwd() / "reproduction" / "src"
    yield Path.cwd() / "src"


def _find_reproduction_source() -> Path:
    checked: list[str] = []
    for candidate in _candidate_reproduction_sources():
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if (resolved / "ft2_formal" / "engine.py").is_file():
            return resolved
    raise RuntimeError(
        "Cannot find reproduction/src. Set FT2_REPRODUCTION_ROOT to the "
        "reproduction directory. Checked: " + ", ".join(checked)
    )


REPRODUCTION_SRC = _find_reproduction_source()
REPRODUCTION_ROOT = REPRODUCTION_SRC.parent
sys.path.insert(0, str(REPRODUCTION_SRC))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ft2_formal.adapters import discover_adapter
from ft2_formal.artifacts import (
    atomic_write_artifact,
    atomic_write_text,
    load_artifact,
    sha256_file,
    utc_now,
)
from ft2_formal.decoding import PreparedPrompt, fixed_greedy_generate, prepare_prompt
from ft2_formal.engine import FT2HookEngine
from ft2_formal.schema import BoundsSource, Correction, ProtectionSpec
from ft2_formal.tasks import DATASETS, MODELS, load_task_dataset, prompt_and_references


def _default_main_root() -> Path:
    configured = os.environ.get("FT2_MAIN_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            # Report bundle containing the archived main-18k selection.
            SCRIPT_PATH.parents[2] / "results" / "raw" / "main_18k_v1",
            # Original reproduction repository on the RTX 4090 server.
            REPRODUCTION_ROOT / "results" / "main_18k_v1",
            Path.cwd() / "reproduction" / "results" / "main_18k_v1",
        )
    )
    for candidate in candidates:
        if (candidate / "selection.json").is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired RTX 4090 timing/VRAM benchmark inside the FT2 reproduction "
            "hook framework (no fault, fixed 180-token greedy decoding)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for overhead_benchmark.json and CSV outputs.",
    )
    parser.add_argument(
        "--main-root",
        type=Path,
        default=_default_main_root(),
        help="main_18k_v1 directory containing the frozen selection.json.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Measured passes over all 10 prompts (default: 2).",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="Excluded warmup pairs before measurement (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device, for example cuda:0 (default: cuda:0).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local Qwen2-Math-7B snapshot override.",
    )
    parser.add_argument(
        "--allow-non-4090",
        action="store_true",
        help="Permit a smoke run on another CUDA GPU; results remain labeled by GPU.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace this script's three output files if they exist.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmup_rounds < 1:
        parser.error("--warmup-rounds must be at least 1")
    return args


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _install_generation_update_compat(model: Any) -> bool:
    """Mirror the compatibility shim used by run_scaling_ablation_v2.py."""
    update = model._update_model_kwargs_for_generation
    if "standardize_cache_format" in inspect.signature(update).parameters:
        return False

    def compatible_update(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("standardize_cache_format", None)
        return update(*args, **kwargs)

    model._update_model_kwargs_for_generation = compatible_update
    return True


def _load_frozen_prompt_rows(main_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection_path = main_root / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Frozen main-18k selection not found: {selection_path}. "
            "Pass --main-root explicitly."
        )
    selection = load_artifact(selection_path)
    try:
        entry = selection["pairs"][PAIR_ID]
    except KeyError as exc:
        raise RuntimeError(f"selection.json has no {PAIR_ID} entry") from exc
    if entry.get("model_key") != MODEL_KEY or entry.get("dataset_key") != DATASET_KEY:
        raise RuntimeError("Frozen selection pair metadata changed")
    fields = ("source_positions", "dataset_indices", "target_steps")
    if any(len(entry.get(field, ())) != EXPECTED_PROMPTS for field in fields):
        raise RuntimeError("Expected exactly 10 frozen Qwen2-Math-7B/GSM8K prompts")
    if len(set(int(value) for value in entry["source_positions"])) != EXPECTED_PROMPTS:
        raise RuntimeError("Frozen source positions contain duplicates")
    screening = entry.get("screening_artifacts", [{}] * EXPECTED_PROMPTS)
    if len(screening) != EXPECTED_PROMPTS:
        raise RuntimeError("Frozen screening-artifact list is not length 10")

    dataset_spec = DATASETS[DATASET_KEY]
    dataset = load_task_dataset(DATASET_KEY)
    rows: list[dict[str, Any]] = []
    for prompt_index, (source, dataset_index, target_step, screen) in enumerate(
        zip(
            entry["source_positions"],
            entry["dataset_indices"],
            entry["target_steps"],
            screening,
        )
    ):
        example = dataset[dataset_spec.evaluation_split][int(dataset_index)]
        prompt, references = prompt_and_references(DATASET_KEY, example)
        rows.append(
            {
                "prompt_index": prompt_index,
                "source_position": int(source),
                "dataset_index": int(dataset_index),
                "frozen_target_step_unused_no_fault": int(target_step),
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "references": list(references),
                "screening_artifact": dict(screen),
            }
        )
    return selection, rows


def _prepare_prompts(
    tokenizer: Any,
    prompt_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> list[PreparedPrompt]:
    prepared: list[PreparedPrompt] = []
    for row in prompt_rows:
        item = prepare_prompt(
            tokenizer,
            str(row["prompt"]),
            device=device,
            max_input_tokens=MAX_INPUT_TOKENS,
        )
        if item.prompt_sha256 != row["prompt_sha256"]:
            raise AssertionError("Prepared prompt hash mismatch")
        if item.input_ids.shape != (1, MAX_INPUT_TOKENS):
            raise AssertionError("Benchmark requires batch 1 and padded length 1024")
        prepared.append(item)
    return prepared


def _protection(mode: str) -> ProtectionSpec:
    if mode == NO_PROTECTION:
        return ProtectionSpec(BoundsSource.NONE, Correction.NONE, 2.0)
    if mode == FIRST_TOKEN:
        return ProtectionSpec(
            BoundsSource.FIRST_TOKEN,
            Correction.PAPER_CLAMP,
            2.0,
        )
    raise KeyError(mode)


def _run_once(
    *,
    model: Any,
    tokenizer: Any,
    prepared: PreparedPrompt,
    engine: FT2HookEngine,
    adapter: Any,
    device: torch.device,
    mode: str,
    phase: str,
    repeat_index: int | None,
    warmup_round: int | None,
    prompt_row: Mapping[str, Any],
    mode_position_in_pair: int,
    sequence_index: int,
) -> dict[str, Any]:
    engine.start_inference(
        fault=None,
        protection=_protection(mode),
        offline_bounds=None,
    )
    try:
        # No pending work may leak into this sample. Resetting peak statistics
        # here makes every row independently auditable.
        torch.cuda.synchronize(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        reserved_before = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        started_ns = time.perf_counter_ns()
        output = fixed_greedy_generate(
            model,
            tokenizer,
            prepared,
            engine,
            num_new_tokens=GENERATION_STEPS,
        )
        torch.cuda.synchronize(device)
        elapsed_ns = time.perf_counter_ns() - started_ns
    except Exception:
        try:
            engine.abort()
        except RuntimeError:
            pass
        raise

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    allocated_after = int(torch.cuda.memory_allocated(device))
    reserved_after = int(torch.cuda.memory_reserved(device))
    record = output.engine_record
    engine_dict = record.to_dict()
    hook_counts = tuple(int(value) for value in record.hook_calls.values())
    expected_bounds = len(adapter.critical_sites) if mode == FIRST_TOKEN else 0
    expected_engine_mode = (
        "no_protection"
        if mode == NO_PROTECTION
        else "paper_clamp_first_token_bounds"
    )
    if (
        len(output.token_ids) != GENERATION_STEPS
        or record.injection_count != 0
        or record.injection_trace is not None
        or record.final_step != GENERATION_STEPS - 1
        or not record.complete
        or record.mode_id != expected_engine_mode
        or len(record.online_bounds) != expected_bounds
        or len(record.hook_calls) != len(adapter.sites)
        or min(hook_counts, default=-1) != GENERATION_STEPS
        or max(hook_counts, default=-1) != GENERATION_STEPS
    ):
        raise AssertionError(f"FT2 benchmark invariant failed for {mode}")
    if mode == NO_PROTECTION and record.correction_elements != 0:
        raise AssertionError("no_protection unexpectedly corrected activations")
    if mode == FIRST_TOKEN and any(
        bounds.scaling_factor != 2.0 for bounds in record.online_bounds.values()
    ):
        raise AssertionError("first-token bounds did not use scaling factor 2.0")
    if output.forward_input_lengths != (MAX_INPUT_TOKENS,) + (1,) * (
        GENERATION_STEPS - 1
    ):
        raise AssertionError("Fixed decoder forward lengths changed")

    elapsed_seconds = elapsed_ns / 1_000_000_000.0
    return {
        "phase": phase,
        "included_in_summary": phase == "measurement",
        "sequence_index": sequence_index,
        "repeat_index": repeat_index,
        "warmup_round": warmup_round,
        "prompt_index": int(prompt_row["prompt_index"]),
        "source_position": int(prompt_row["source_position"]),
        "dataset_index": int(prompt_row["dataset_index"]),
        "mode": mode,
        "engine_mode_id": record.mode_id,
        "mode_position_in_pair": mode_position_in_pair,
        "elapsed_ns": elapsed_ns,
        "elapsed_seconds": elapsed_seconds,
        "decoder_elapsed_seconds": float(output.elapsed_seconds),
        "tokens_per_second": GENERATION_STEPS / float(output.elapsed_seconds),
        "outer_tokens_per_second": GENERATION_STEPS / elapsed_seconds,
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": peak_allocated,
        "peak_extra_allocated_bytes": max(0, peak_allocated - allocated_before),
        "allocated_after_bytes": allocated_after,
        "reserved_before_bytes": reserved_before,
        "peak_reserved_bytes": peak_reserved,
        "peak_extra_reserved_bytes": max(0, peak_reserved - reserved_before),
        "reserved_after_bytes": reserved_after,
        "unpadded_input_tokens": int(prepared.unpadded_token_count),
        "padded_input_tokens": int(prepared.input_ids.shape[1]),
        "generated_tokens": len(output.token_ids),
        "first_eos_step": output.first_eos_step,
        "token_sha256": output.token_sha256,
        "text_sha256": _sha256_text(output.text),
        "token_ids": list(output.token_ids),
        "correction_elements": int(record.correction_elements),
        "online_bounds_key_count": len(record.online_bounds),
        "hook_site_count": len(record.hook_calls),
        "hook_calls_min": min(hook_counts),
        "hook_calls_max": max(hook_counts),
        "engine": engine_dict,
    }


def _describe(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "sample_stdev": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
    }


SUMMARY_METRICS = (
    ("elapsed_seconds", "seconds"),
    ("decoder_elapsed_seconds", "seconds"),
    ("peak_allocated_bytes", "bytes"),
    ("peak_extra_allocated_bytes", "bytes"),
    ("peak_reserved_bytes", "bytes"),
    ("peak_extra_reserved_bytes", "bytes"),
)


def _summarize_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row["phase"] == "measurement"]
    by_mode = {
        mode: [row for row in measured if row["mode"] == mode]
        for mode in MODE_IDS
    }
    if len(by_mode[NO_PROTECTION]) != len(by_mode[FIRST_TOKEN]):
        raise AssertionError("Unbalanced mode sample counts")

    mode_summary: dict[str, Any] = {}
    for mode, mode_rows in by_mode.items():
        total_generated_tokens = sum(int(row["generated_tokens"]) for row in mode_rows)
        total_decoder_seconds = sum(
            float(row["decoder_elapsed_seconds"]) for row in mode_rows
        )
        mode_summary[mode] = {
            "runs": len(mode_rows),
            "primary_timing": {
                "metric": "decoder_elapsed_seconds",
                "total_generated_tokens": total_generated_tokens,
                "total_decoder_elapsed_seconds": total_decoder_seconds,
                "milliseconds_per_generated_token": (
                    1000.0 * total_decoder_seconds / total_generated_tokens
                ),
                "generated_tokens_per_second": (
                    total_generated_tokens / total_decoder_seconds
                ),
            },
            "metrics": {
                metric: _describe([row[metric] for row in mode_rows])
                for metric, _unit in SUMMARY_METRICS
            },
            "correction_elements": _describe(
                [row["correction_elements"] for row in mode_rows]
            ),
        }

    paired_lookup: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = {}
    for row in measured:
        key = (int(row["repeat_index"]), int(row["prompt_index"]))
        target = paired_lookup.setdefault(key, {})
        if row["mode"] in target:
            raise AssertionError(f"Duplicate paired measurement: {key}, {row['mode']}")
        target[str(row["mode"])] = row

    paired_rows: list[dict[str, Any]] = []
    for (repeat_index, prompt_index), pair in sorted(paired_lookup.items()):
        if set(pair) != set(MODE_IDS):
            raise AssertionError(f"Incomplete measurement pair: {(repeat_index, prompt_index)}")
        baseline = pair[NO_PROTECTION]
        protected = pair[FIRST_TOKEN]
        item: dict[str, Any] = {
            "repeat_index": repeat_index,
            "prompt_index": prompt_index,
            "source_position": int(baseline["source_position"]),
            "dataset_index": int(baseline["dataset_index"]),
            "first_mode": (
                baseline["mode"]
                if int(baseline["mode_position_in_pair"]) == 0
                else protected["mode"]
            ),
            "outputs_identical": baseline["token_sha256"] == protected["token_sha256"],
        }
        for metric, _unit in SUMMARY_METRICS:
            base_value = float(baseline[metric])
            protected_value = float(protected[metric])
            delta = protected_value - base_value
            item[f"{metric}_no_protection"] = baseline[metric]
            item[f"{metric}_first_token"] = protected[metric]
            item[f"{metric}_delta"] = delta
            item[f"{metric}_change_percent"] = (
                100.0 * delta / base_value if base_value != 0.0 else None
            )
        paired_rows.append(item)

    comparison: dict[str, Any] = {
        "pair_count": len(paired_rows),
        "baseline": NO_PROTECTION,
        "candidate": FIRST_TOKEN,
        "scope_statement": SCOPE_STATEMENT,
        "metrics": {},
    }
    for metric, unit in SUMMARY_METRICS:
        baseline_values = [float(row[f"{metric}_no_protection"]) for row in paired_rows]
        protected_values = [float(row[f"{metric}_first_token"]) for row in paired_rows]
        deltas = [float(row[f"{metric}_delta"]) for row in paired_rows]
        baseline_mean = statistics.fmean(baseline_values)
        protected_mean = statistics.fmean(protected_values)
        comparison["metrics"][metric] = {
            "unit": unit,
            "no_protection_mean": baseline_mean,
            "first_token_mean": protected_mean,
            "first_token_minus_no_protection_mean": statistics.fmean(deltas),
            "paired_delta": _describe(deltas),
            "ratio_of_means": (
                protected_mean / baseline_mean if baseline_mean != 0.0 else None
            ),
            "change_percent_from_ratio_of_means": (
                100.0 * (protected_mean / baseline_mean - 1.0)
                if baseline_mean != 0.0
                else None
            ),
            "no_protection_max_observed": max(baseline_values),
            "first_token_max_observed": max(protected_values),
            "first_token_minus_no_protection_max_observed": (
                max(protected_values) - max(baseline_values)
            ),
            "max_observed_change_percent": (
                100.0 * (max(protected_values) / max(baseline_values) - 1.0)
                if max(baseline_values) != 0.0
                else None
            ),
        }
    baseline_primary = mode_summary[NO_PROTECTION]["primary_timing"]
    protected_primary = mode_summary[FIRST_TOKEN]["primary_timing"]
    baseline_total = float(baseline_primary["total_decoder_elapsed_seconds"])
    protected_total = float(protected_primary["total_decoder_elapsed_seconds"])
    baseline_ms_token = float(baseline_primary["milliseconds_per_generated_token"])
    protected_ms_token = float(protected_primary["milliseconds_per_generated_token"])
    comparison["primary_timing"] = {
        "metric": "decoder_elapsed_seconds",
        "generated_tokens_per_mode": int(baseline_primary["total_generated_tokens"]),
        "no_protection_total_seconds": baseline_total,
        "first_token_total_seconds": protected_total,
        "first_token_minus_no_protection_total_seconds": (
            protected_total - baseline_total
        ),
        "no_protection_milliseconds_per_generated_token": baseline_ms_token,
        "first_token_milliseconds_per_generated_token": protected_ms_token,
        "first_token_minus_no_protection_milliseconds_per_generated_token": (
            protected_ms_token - baseline_ms_token
        ),
        "ratio": protected_total / baseline_total,
        "overhead_percent": 100.0 * (protected_total / baseline_total - 1.0),
    }
    return {
        "mode_summary": mode_summary,
        "paired_comparison": comparison,
        "paired_rows": paired_rows,
    }


CSV_FIELDS = (
    "phase",
    "included_in_summary",
    "sequence_index",
    "repeat_index",
    "warmup_round",
    "prompt_index",
    "source_position",
    "dataset_index",
    "mode",
    "engine_mode_id",
    "mode_position_in_pair",
    "elapsed_ns",
    "elapsed_seconds",
    "decoder_elapsed_seconds",
    "tokens_per_second",
    "outer_tokens_per_second",
    "allocated_before_bytes",
    "peak_allocated_bytes",
    "peak_extra_allocated_bytes",
    "allocated_after_bytes",
    "reserved_before_bytes",
    "peak_reserved_bytes",
    "peak_extra_reserved_bytes",
    "reserved_after_bytes",
    "unpadded_input_tokens",
    "padded_input_tokens",
    "generated_tokens",
    "first_eos_step",
    "token_sha256",
    "text_sha256",
    "correction_elements",
    "online_bounds_key_count",
    "hook_site_count",
    "hook_calls_min",
    "hook_calls_max",
)


def _measurements_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue()


def _summary_csv(summary: Mapping[str, Any]) -> str:
    fields = (
        "metric",
        "unit",
        "aggregation",
        "pairs",
        "no_protection",
        "first_token",
        "first_token_minus_no_protection",
        "relative_change_percent",
        "paired_delta_median",
        "paired_delta_sample_stdev",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    comparison = summary["paired_comparison"]
    primary = comparison["primary_timing"]
    writer.writerow(
        {
            "metric": "decoder_elapsed_seconds",
            "unit": "seconds",
            "aggregation": "sum_over_measured_runs",
            "pairs": comparison["pair_count"],
            "no_protection": primary["no_protection_total_seconds"],
            "first_token": primary["first_token_total_seconds"],
            "first_token_minus_no_protection": primary[
                "first_token_minus_no_protection_total_seconds"
            ],
            "relative_change_percent": primary["overhead_percent"],
        }
    )
    writer.writerow(
        {
            "metric": "decoder_latency",
            "unit": "milliseconds_per_generated_token",
            "aggregation": "sum_time_divided_by_total_tokens",
            "pairs": comparison["pair_count"],
            "no_protection": primary[
                "no_protection_milliseconds_per_generated_token"
            ],
            "first_token": primary[
                "first_token_milliseconds_per_generated_token"
            ],
            "first_token_minus_no_protection": primary[
                "first_token_minus_no_protection_milliseconds_per_generated_token"
            ],
            "relative_change_percent": primary["overhead_percent"],
        }
    )
    for metric, unit in SUMMARY_METRICS:
        item = comparison["metrics"][metric]
        writer.writerow(
            {
                "metric": metric,
                "unit": unit,
                "aggregation": "mean_per_run",
                "pairs": comparison["pair_count"],
                "no_protection": item["no_protection_mean"],
                "first_token": item["first_token_mean"],
                "first_token_minus_no_protection": item[
                    "first_token_minus_no_protection_mean"
                ],
                "relative_change_percent": item[
                    "change_percent_from_ratio_of_means"
                ],
                "paired_delta_median": item["paired_delta"]["median"],
                "paired_delta_sample_stdev": item["paired_delta"]["sample_stdev"],
            }
        )
        if metric in {
            "peak_allocated_bytes",
            "peak_extra_allocated_bytes",
            "peak_reserved_bytes",
            "peak_extra_reserved_bytes",
        }:
            writer.writerow(
                {
                    "metric": metric,
                    "unit": unit,
                    "aggregation": "maximum_observed",
                    "pairs": comparison["pair_count"],
                    "no_protection": item["no_protection_max_observed"],
                    "first_token": item["first_token_max_observed"],
                    "first_token_minus_no_protection": item[
                        "first_token_minus_no_protection_max_observed"
                    ],
                    "relative_change_percent": item[
                        "max_observed_change_percent"
                    ],
                }
            )
    return stream.getvalue()


def _source_records() -> list[dict[str, str]]:
    paths = [
        SCRIPT_PATH,
        REPRODUCTION_SRC / "ft2_formal" / "main18k.py",
        REPRODUCTION_SRC / "ft2_formal" / "decoding.py",
        REPRODUCTION_SRC / "ft2_formal" / "engine.py",
        REPRODUCTION_SRC / "ft2_formal" / "adapters.py",
        REPRODUCTION_ROOT / "experiments" / "run_scaling_ablation_v2.py",
    ]
    return [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in paths
        if path.is_file()
    ]


def _device_metadata(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": _package_version("datasets"),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    main_root = args.main_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "overhead_benchmark.json"
    measurements_csv_path = output_dir / "overhead_measurements.csv"
    summary_csv_path = output_dir / "overhead_summary.csv"
    output_paths = (json_path, measurements_csv_path, summary_csv_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to replace existing benchmark outputs without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this RTX 4090 benchmark")
    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise ValueError("--device must name a CUDA device")
    device_index = (
        torch.cuda.current_device()
        if requested_device.index is None
        else requested_device.index
    )
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    gpu_name = torch.cuda.get_device_name(device)
    if "4090" not in gpu_name and not args.allow_non_4090:
        raise RuntimeError(
            f"Expected an RTX 4090, found {gpu_name!r}. "
            "Use --allow-non-4090 only for a non-report smoke run."
        )

    selection, prompt_rows = _load_frozen_prompt_rows(main_root)
    model_spec = MODELS[MODEL_KEY]
    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path is not None
        else model_spec.snapshot.resolve()
    )
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Frozen model snapshot not found: {model_path}. Pass --model-path."
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started_at = utc_now()
    print(f"[overhead] scope: {SCOPE_STATEMENT}", flush=True)
    print(f"[overhead] gpu={gpu_name} repeats={args.repeats}", flush=True)
    print(f"[overhead] loading {MODEL_KEY} from {model_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Qwen2-Math-7B tokenizer has neither PAD nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation=model_spec.attention_implementation,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    compatibility_shim = _install_generation_update_compat(model)
    adapter = discover_adapter(model, MODEL_KEY)
    prepared_prompts = _prepare_prompts(tokenizer, prompt_rows, device)
    torch.cuda.synchronize(device)

    all_rows: list[dict[str, Any]] = []
    sequence_index = 0
    with FT2HookEngine(adapter) as engine:
        # Each warmup round runs both conditions on one frozen prompt and is
        # retained in the raw outputs but excluded from every summary statistic.
        for warmup_round in range(args.warmup_rounds):
            prompt_index = warmup_round % EXPECTED_PROMPTS
            order = MODE_IDS if warmup_round % 2 == 0 else tuple(reversed(MODE_IDS))
            for mode_position, mode in enumerate(order):
                row = _run_once(
                    model=model,
                    tokenizer=tokenizer,
                    prepared=prepared_prompts[prompt_index],
                    engine=engine,
                    adapter=adapter,
                    device=device,
                    mode=mode,
                    phase="warmup",
                    repeat_index=None,
                    warmup_round=warmup_round,
                    prompt_row=prompt_rows[prompt_index],
                    mode_position_in_pair=mode_position,
                    sequence_index=sequence_index,
                )
                all_rows.append(row)
                sequence_index += 1
                print(
                    f"[warmup] round={warmup_round + 1}/{args.warmup_rounds} "
                    f"mode={mode}",
                    flush=True,
                )

        # Alternating paired crossover: the first condition flips for adjacent
        # prompt/repeat pairs, balancing first/second position within each pass.
        for repeat_index in range(args.repeats):
            for prompt_index, (prompt_row, prepared) in enumerate(
                zip(prompt_rows, prepared_prompts)
            ):
                order = (
                    MODE_IDS
                    if (repeat_index + prompt_index) % 2 == 0
                    else tuple(reversed(MODE_IDS))
                )
                for mode_position, mode in enumerate(order):
                    row = _run_once(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=prepared,
                        engine=engine,
                        adapter=adapter,
                        device=device,
                        mode=mode,
                        phase="measurement",
                        repeat_index=repeat_index,
                        warmup_round=None,
                        prompt_row=prompt_row,
                        mode_position_in_pair=mode_position,
                        sequence_index=sequence_index,
                    )
                    all_rows.append(row)
                    sequence_index += 1
                    print(
                        f"[measure] repeat={repeat_index + 1}/{args.repeats} "
                        f"prompt={prompt_index + 1}/{EXPECTED_PROMPTS} mode={mode} "
                        f"seconds={row['elapsed_seconds']:.6f} "
                        f"peak_MiB={row['peak_allocated_bytes'] / 2**20:.2f}",
                        flush=True,
                    )

    summary = _summarize_measurements(all_rows)
    atomic_write_text(measurements_csv_path, _measurements_csv(all_rows))
    atomic_write_text(summary_csv_path, _summary_csv(summary))

    prompt_manifest = [
        {
            key: row[key]
            for key in (
                "prompt_index",
                "source_position",
                "dataset_index",
                "frozen_target_step_unused_no_fault",
                "prompt_sha256",
                "screening_artifact",
            )
        }
        | {"unpadded_input_tokens": prepared.unpadded_token_count}
        for row, prepared in zip(prompt_rows, prepared_prompts)
    ]
    result = {
        "schema_version": 1,
        "benchmark_id": "ft2_qwen_gsm8k_hook_overhead_rtx4090_v1",
        "status": "completed",
        "scope_statement": SCOPE_STATEMENT,
        "timing_scope": {
            "primary_metric": "decoder_elapsed_seconds",
            "primary_metric_definition": (
                "The synchronized 180-step generation loop reported by "
                "fixed_greedy_generate; it includes FT2 hook work."
            ),
            "outer_metric": "elapsed_seconds",
            "outer_metric_definition": (
                "A separately synchronized wall timer around the whole "
                "fixed_greedy_generate call, including record finalization and "
                "host-side token decoding."
            ),
            "included": (
                "fixed_greedy_generate call, FT2 hook execution, CUDA work, and "
                "generation record finalization; both primary and outer metrics "
                "are retained"
            ),
            "excluded": (
                "model/dataset loading, tokenization, prompt preparation, warmups, "
                "console logging, and result serialization"
            ),
            "timer": "time.perf_counter_ns with torch.cuda.synchronize before/after",
        },
        "memory_scope": {
            "allocator": "PyTorch CUDA caching allocator only",
            "per_run_reset": "torch.cuda.reset_peak_memory_stats",
            "peak_extra_definition": "max_memory_* minus memory_* immediately before run",
            "limitations": (
                "Does not include non-PyTorch driver allocations; reserved-memory "
                "deltas may be zero after warmup because the caching allocator reuses blocks."
            ),
        },
        "protocol": {
            "model_key": MODEL_KEY,
            "model_id": model_spec.model_id,
            "model_revision": model_spec.revision,
            "model_path": str(model_path),
            "attention_implementation": model_spec.attention_implementation,
            "dtype": "float16",
            "dataset_key": DATASET_KEY,
            "dataset_id": DATASETS[DATASET_KEY].dataset_id,
            "dataset_revision": DATASETS[DATASET_KEY].revision,
            "dataset_split": DATASETS[DATASET_KEY].evaluation_split,
            "prompt_count": EXPECTED_PROMPTS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "batch_size": 1,
            "num_new_tokens": GENERATION_STEPS,
            "decoder": "greedy_argmax",
            "stop_on_eos": False,
            "fault": None,
            "modes": {
                NO_PROTECTION: {
                    "bounds_source": "none",
                    "correction": "none",
                    "hooks_installed": True,
                },
                FIRST_TOKEN: {
                    "bounds_source": "first_token",
                    "correction": "paper_clamp",
                    "scaling_factor": 2.0,
                    "hooks_installed": True,
                },
            },
            "warmup_rounds": args.warmup_rounds,
            "repeats": args.repeats,
            "measured_runs_per_mode": args.repeats * EXPECTED_PROMPTS,
            "order": "paired alternating crossover; parity(repeat + prompt) chooses AB or BA",
            "seed": SEED,
        },
        "environment": _device_metadata(device),
        "compatibility_shim_enabled": compatibility_shim,
        "inputs": {
            "main_root": str(main_root),
            "selection_path": str((main_root / "selection.json").resolve()),
            "selection_file_sha256": sha256_file(main_root / "selection.json"),
            "selection_artifact_sha256": selection["artifact_sha256"],
            "campaign_fingerprint": selection.get("campaign_fingerprint"),
            "prompts": prompt_manifest,
            "source_files": _source_records(),
        },
        "summary": summary,
        "runs": all_rows,
        "output_files": {
            measurements_csv_path.name: sha256_file(measurements_csv_path),
            summary_csv_path.name: sha256_file(summary_csv_path),
        },
        "started_at": started_at,
        "completed_at": utc_now(),
    }
    artifact = atomic_write_artifact(json_path, result)
    print(
        f"[overhead] completed artifact_sha256={artifact['artifact_sha256']}\n"
        f"[overhead] json={json_path}\n"
        f"[overhead] measurements_csv={measurements_csv_path}\n"
        f"[overhead] summary_csv={summary_csv_path}",
        flush=True,
    )

    del prepared_prompts, adapter, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
