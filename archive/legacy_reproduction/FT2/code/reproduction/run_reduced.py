#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault(
    "HF_HOME",
    "/mnt/ft2-data/cache/huggingface",
)
os.environ.setdefault(
    "HF_HUB_CACHE",
    "/mnt/ft2-data/cache/huggingface/hub",
)
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    "/mnt/ft2-data/cache/datasets",
)
os.environ.setdefault(
    "TORCH_HOME",
    "/mnt/ft2-data/cache/torch",
)

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ft2_repro import (
    ProtectionMode,
    RunState,
    install_qwen2_hooks,
)
from ft2_repro.generation import (
    GenerationResult,
    PreparedPrompt,
    generate_fixed,
    prepare_prompt,
    squad_prompt,
)
from ft2_repro.manifest import (
    AUTHOR_PROJECTION_WEIGHTS,
    ManifestEntry,
    ModelShape,
    build_fault_manifest,
    load_manifest,
    manifest_sha256,
    save_manifest,
    validate_manifest,
)
from ft2_repro.metrics import (
    compare_decoded_text,
    compare_token_ids,
    squad_semantic_score,
    token_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REVISION = (
    "a257a714105a2c4a79c898201cfd907bc9b02138"
)
DEFAULT_DATASET_REVISION = (
    "3ffb306f725f7d2ce8394bc1873b24868140c412"
)
AUTHOR_EVAL_DIR = REPO_ROOT / "performance" / "sigcode" / "evaluation"
DEFAULT_QID_FILE = AUTHOR_EVAL_DIR / "qidsquadfinal.txt"
DEFAULT_STEP_FILE = AUTHOR_EVAL_DIR / "squadrandomtokenqwen1.5b.txt"
MODES: Tuple[ProtectionMode, ...] = (
    ProtectionMode.UNPROTECTED,
    ProtectionMode.REPOSITORY_ZERO,
    ProtectionMode.PAPER_CLAMP,
)


class CampaignInvariantError(RuntimeError):
    """A run violated a predeclared experimental invariant."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_int_lines(path: Path) -> Tuple[int, ...]:
    values = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number} is not an integer"
            ) from exc
    if not values:
        raise ValueError(f"{path} contains no integers")
    return tuple(values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def reproduction_code_sha256() -> str:
    digest = hashlib.sha256()
    roots = (
        REPO_ROOT / "reproduction" / "src",
        REPO_ROOT / "reproduction" / "run_reduced.py",
        REPO_ROOT / "reproduction" / "summarize.py",
    )
    paths = []
    for root in roots:
        if root.is_dir():
            paths.extend(root.rglob("*.py"))
        elif root.exists():
            paths.append(root)
    for path in sorted(paths):
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        record = json.loads(raw)
        run_id = str(record["run_id"])
        if run_id in records:
            raise ValueError(
                f"Duplicate run_id {run_id} in {path}:{line_number}"
            )
        records[run_id] = record
    return records


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def bounds_sha256(state: RunState) -> str:
    payload = [
        {
            "layer_index": layer_index,
            "projection": projection,
            "lower": bounds.lower,
            "upper": bounds.upper,
        }
        for (layer_index, projection), bounds in sorted(
            state.bounds.items()
        )
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def result_fields(result: GenerationResult) -> Dict[str, Any]:
    return {
        "generated_token_ids": list(result.token_ids),
        "generated_token_sha256": result.token_sha256,
        "decoded_text": result.text,
        "first_eos_step": result.first_eos_step,
        "forward_input_lengths": list(result.forward_input_lengths),
        "elapsed_seconds": result.elapsed_seconds,
    }


def answer_references(example: Mapping[str, Any]) -> Tuple[str, ...]:
    answers = example.get("answers")
    if not isinstance(answers, Mapping):
        raise TypeError("SQuAD example has no answers mapping")
    texts = answers.get("text")
    if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
        raise TypeError("SQuAD answers.text must be a sequence")
    references = tuple(str(text) for text in texts)
    return references if references else ("",)


def run_mode(
    model: torch.nn.Module,
    tokenizer: Any,
    prepared: PreparedPrompt,
    mode: ProtectionMode,
    faults: Sequence[Any],
    shape: ModelShape,
    num_new_tokens: int,
    state: RunState | None = None,
) -> Tuple[GenerationResult, RunState]:
    planned_faults = tuple(faults)
    if state is None:
        state = RunState(mode=mode, faults=planned_faults)
    elif state.mode != mode or tuple(state.faults) != planned_faults:
        raise ValueError(
            "Supplied RunState does not match requested mode and faults"
        )
    with install_qwen2_hooks(model, state):
        result = generate_fixed(
            model,
            tokenizer,
            prepared,
            num_new_tokens=num_new_tokens,
        )
    try:
        state.assert_complete()
    except AssertionError as error:
        raise CampaignInvariantError(str(error)) from error

    expected_bounds = shape.num_layers * 4
    if len(state.bounds) != expected_bounds:
        raise CampaignInvariantError(
            f"Expected {expected_bounds} bounds, got {len(state.bounds)}"
        )
    expected_lengths = (
        prepared.input_ids.shape[1],
        *([1] * (num_new_tokens - 1)),
    )
    if result.forward_input_lengths != expected_lengths:
        raise CampaignInvariantError(
            "Generation forward input lengths violate the fixed-step design"
        )
    return result, state


def clean_record(
    sample_index: int,
    dataset_index: int,
    prompt_hash: str,
    mode: ProtectionMode,
    result: GenerationResult,
    state: RunState,
    common_tokens: Sequence[int] | None,
    common_text: str | None,
    reference_answers: Sequence[str],
    manifest_hash: str,
    campaign_id: str,
) -> Dict[str, Any]:
    token_comparison = (
        compare_token_ids(result.token_ids, common_tokens)
        if common_tokens is not None
        else compare_token_ids(result.token_ids, result.token_ids)
    )
    text_comparison = (
        compare_decoded_text(result.text, common_text)
        if common_text is not None
        else compare_decoded_text(result.text, result.text)
    )
    semantic = squad_semantic_score(result.text, reference_answers)
    return {
        "schema_version": 2,
        "record_type": "clean",
        "run_id": f"clean:{sample_index}:{mode.value}",
        "campaign_id": campaign_id,
        "sample_index": sample_index,
        "dataset_index": dataset_index,
        "prompt_sha256": prompt_hash,
        "mode": mode.value,
        "status": "COMPLETED",
        "manifest_sha256": manifest_hash,
        "bounds_sha256": bounds_sha256(state),
        "injection_count": sum(state.injection_counts.values()),
        "reference_answers": list(reference_answers),
        "common_golden_sha256": (
            token_sha256(common_tokens)
            if common_tokens is not None
            else result.token_sha256
        ),
        "common_golden_text_sha256": text_comparison[
            "expected_text_sha256"
        ],
        "clean_drift": not token_comparison["equal"],
        "clean_token_drift": not token_comparison["equal"],
        "clean_text_drift": not text_comparison["equal"],
        "clean_semantic_correct": semantic["semantic_correct"],
        "semantic_score": semantic,
        "first_different_token_common": token_comparison[
            "first_different_token"
        ],
        "different_token_count_common": token_comparison[
            "different_token_count"
        ],
        **result_fields(result),
        "completed_at": utc_now(),
    }


def fault_record(
    entry: ManifestEntry,
    prompt_hash: str,
    mode: ProtectionMode,
    result: GenerationResult,
    state: RunState,
    common_tokens: Sequence[int],
    mode_clean_tokens: Sequence[int],
    common_text: str,
    mode_clean_text: str,
    reference_answers: Sequence[str],
    manifest_hash: str,
    campaign_id: str,
) -> Dict[str, Any]:
    common_tokens_comparison = compare_token_ids(result.token_ids, common_tokens)
    mode_tokens_comparison = compare_token_ids(
        result.token_ids, mode_clean_tokens
    )
    common_text_comparison = compare_decoded_text(result.text, common_text)
    mode_text_comparison = compare_decoded_text(result.text, mode_clean_text)
    semantic = squad_semantic_score(result.text, reference_answers)
    injection_count = state.injection_counts[entry.spec_id]
    if injection_count != 1:
        raise CampaignInvariantError(
            f"Fault {entry.spec_id} injection count is {injection_count}"
        )
    telemetry = state.injection_records[entry.spec_id].to_dict()
    if not telemetry["bit_flip_verified"]:
        raise CampaignInvariantError(
            "Unverified bit flip cannot be an eligible trial"
        )
    semantic_sdc_common = (
        not common_text_comparison["equal"]
        and not semantic["semantic_correct"]
    )
    semantic_sdc_mode_clean = (
        not mode_text_comparison["equal"]
        and not semantic["semantic_correct"]
    )
    return {
        "schema_version": 2,
        "record_type": "fault",
        "run_id": f"{entry.spec_id}:{mode.value}",
        "campaign_id": campaign_id,
        "spec_id": entry.spec_id,
        "sample_index": entry.fault.sample_index,
        "dataset_index": entry.dataset_index,
        "trial_index": entry.fault.trial_index,
        "prompt_sha256": prompt_hash,
        "mode": mode.value,
        "error_phase": None,
        "status": (
            "COMPLETED_SEMANTIC_SDC"
            if semantic_sdc_common
            else "COMPLETED_MASKED"
        ),
        "outcome": (
            "SEMANTIC_SDC"
            if semantic_sdc_common
            else "MASKED"
        ),
        "eligible": True,
        "due": False,
        "invalid": False,
        "manifest_sha256": manifest_hash,
        "fault": entry.to_dict(),
        "fault_step_reached": state.current_step >= entry.fault.token_step,
        "injection_count": injection_count,
        "injection": telemetry,
        "bit_flip_verified": telemetry["bit_flip_verified"],
        "injected_value_before_fp16_hex": telemetry["before_bits_hex"],
        "injected_value_after_fp16_hex": telemetry["after_bits_hex"],
        "protected_value_fp16_hex": telemetry["protected_bits_hex"],
        "bounds_sha256": bounds_sha256(state),
        "reference_answers": list(reference_answers),
        "common_golden_sha256": token_sha256(common_tokens),
        "mode_clean_sha256": token_sha256(mode_clean_tokens),
        "common_golden_text_sha256": common_text_comparison[
            "expected_text_sha256"
        ],
        "mode_clean_text_sha256": mode_text_comparison[
            "expected_text_sha256"
        ],
        "token_divergence_common": not common_tokens_comparison["equal"],
        "token_divergence_mode_clean": not mode_tokens_comparison["equal"],
        "text_divergence_common": not common_text_comparison["equal"],
        "text_divergence_mode_clean": not mode_text_comparison["equal"],
        "semantic_correct": semantic["semantic_correct"],
        "semantic_score": semantic,
        "semantic_sdc_common": semantic_sdc_common,
        "semantic_sdc_mode_clean": semantic_sdc_mode_clean,
        "paper_masked": not semantic_sdc_common,
        "mission_failure": semantic_sdc_common,
        "first_different_token_common": common_tokens_comparison[
            "first_different_token"
        ],
        "different_token_count_common": common_tokens_comparison[
            "different_token_count"
        ],
        "first_different_token_mode_clean": mode_tokens_comparison[
            "first_different_token"
        ],
        "different_token_count_mode_clean": mode_tokens_comparison[
            "different_token_count"
        ],
        **result_fields(result),
        "completed_at": utc_now(),
    }


def fault_error_record(
    entry: ManifestEntry,
    prompt_hash: str,
    mode: ProtectionMode,
    state: RunState,
    error: Exception,
    error_phase: str,
    common_tokens: Sequence[int],
    mode_clean_tokens: Sequence[int],
    common_text: str,
    mode_clean_text: str,
    reference_answers: Sequence[str],
    manifest_hash: str,
    campaign_id: str,
) -> Dict[str, Any]:
    telemetry_record = state.injection_records.get(entry.spec_id)
    telemetry = (
        telemetry_record.to_dict()
        if telemetry_record is not None
        else None
    )
    injection_count = state.injection_counts.get(entry.spec_id, 0)
    injection_verified = bool(
        injection_count == 1
        and telemetry is not None
        and telemetry["bit_flip_verified"]
    )
    eligible = error_phase == "generation" and injection_verified
    return {
        "schema_version": 2,
        "record_type": "fault",
        "run_id": f"{entry.spec_id}:{mode.value}",
        "campaign_id": campaign_id,
        "spec_id": entry.spec_id,
        "sample_index": entry.fault.sample_index,
        "dataset_index": entry.dataset_index,
        "trial_index": entry.fault.trial_index,
        "prompt_sha256": prompt_hash,
        "mode": mode.value,
        "error_phase": error_phase,
        "status": "DUE_EXCEPTION" if eligible else "INVALID_EXCEPTION",
        "outcome": "DUE" if eligible else "INVALID",
        "eligible": eligible,
        "due": eligible,
        "invalid": not eligible,
        "mission_failure": True if eligible else None,
        "semantic_sdc_common": None,
        "semantic_sdc_mode_clean": None,
        "semantic_correct": None,
        "semantic_score": None,
        "paper_masked": None,
        "token_divergence_common": None,
        "token_divergence_mode_clean": None,
        "text_divergence_common": None,
        "text_divergence_mode_clean": None,
        "manifest_sha256": manifest_hash,
        "fault": entry.to_dict(),
        "fault_step_reached": state.current_step >= entry.fault.token_step,
        "injection_count": injection_count,
        "injection": telemetry,
        "bit_flip_verified": (
            telemetry["bit_flip_verified"] if telemetry else False
        ),
        "injected_value_before_fp16_hex": (
            telemetry["before_bits_hex"] if telemetry else None
        ),
        "injected_value_after_fp16_hex": (
            telemetry["after_bits_hex"] if telemetry else None
        ),
        "protected_value_fp16_hex": (
            telemetry["protected_bits_hex"] if telemetry else None
        ),
        "bounds_sha256": bounds_sha256(state),
        "reference_answers": list(reference_answers),
        "common_golden_sha256": token_sha256(common_tokens),
        "mode_clean_sha256": token_sha256(mode_clean_tokens),
        "common_golden_text_sha256": sha256_text(common_text),
        "mode_clean_text_sha256": sha256_text(mode_clean_text),
        "error_type": type(error).__name__,
        "error_message": str(error)[:1000],
        "completed_at": utc_now(),
    }


def load_or_plan_manifest(
    path: Path,
    dataset_indices: Sequence[int],
    fault_steps: Sequence[int],
    trials_per_sample: int,
    shape: ModelShape,
    seed: int,
    num_new_tokens: int,
) -> Tuple[Tuple[ManifestEntry, ...], str, bool]:
    expected = build_fault_manifest(
        dataset_indices=dataset_indices,
        fault_steps=fault_steps,
        trials_per_sample=trials_per_sample,
        shape=shape,
        seed=seed,
    )

    validate_manifest(
        expected,
        sample_count=len(dataset_indices),
        trials_per_sample=trials_per_sample,
        shape=shape,
        num_new_tokens=num_new_tokens,
    )
    expected_payload = [entry.to_dict() for entry in expected]
    expected_hash = manifest_sha256(expected)
    if not path.exists():
        return expected, expected_hash, True

    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if canonical_json(raw_payload) != canonical_json(expected_payload):
        raise ValueError(
            "Existing manifest differs from the deterministic expected manifest"
        )
    observed = load_manifest(path)
    validate_manifest(
        observed,
        sample_count=len(dataset_indices),
        trials_per_sample=trials_per_sample,
        shape=shape,
        num_new_tokens=num_new_tokens,
    )
    return expected, expected_hash, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired reduced-scale FT2 reproduction on Qwen2-1.5B"
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2-Math-1.5B",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
    )
    parser.add_argument(
        "--dataset-revision",
        default=DEFAULT_DATASET_REVISION,
    )
    parser.add_argument("--dataset", default="squad_v2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--trials-per-sample", type=int, default=10)
    parser.add_argument("--num-new-tokens", type=int, default=60)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=196)
    parser.add_argument("--qid-file", type=Path, default=DEFAULT_QID_FILE)
    parser.add_argument(
        "--fault-step-file",
        type=Path,
        default=DEFAULT_STEP_FILE,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one sample and one paired fault trial",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.samples = 1
        args.trials_per_sample = 1
        args.num_new_tokens = 24
    if min(
        args.samples,
        args.trials_per_sample,
        args.num_new_tokens,
        args.max_prompt_tokens,
    ) <= 0:
        raise ValueError("All campaign sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen2 FT2 campaign")

    output_dir = args.output_dir
    if output_dir is None:
        name = "qwen2_1p5b_smoke" if args.smoke else "qwen2_1p5b_reduced"
        output_dir = Path("/mnt/ft2-data/results") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    print(f"[{utc_now()}] Loading tokenizer {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    added_pad_token = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        added_pad_token = True

    print(f"[{utc_now()}] Loading FP16 model with SDPA", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    model.to(torch.device("cuda"))
    model.eval()
    model.config.use_cache = True
    if getattr(model.config, "model_type", None) != "qwen2":
        raise TypeError(
            f"Expected Qwen2 model, got {model.config.model_type!r}"
        )

    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    if (
        args.model_revision is not None
        and resolved_model_revision != args.model_revision
    ):
        raise RuntimeError("Resolved model revision differs from requested pin")

    shape = ModelShape.from_config(model.config)
    dataset_indices = read_int_lines(args.qid_file)[: args.samples]
    fault_steps = read_int_lines(args.fault_step_file)[: args.samples]
    if len(dataset_indices) != args.samples:
        raise ValueError("QID file does not contain enough samples")
    if len(fault_steps) != args.samples:
        raise ValueError("Fault-step file does not contain enough samples")

    manifest_path = output_dir / "manifest.json"
    entries, manifest_hash, manifest_needs_create = load_or_plan_manifest(
        path=manifest_path,
        dataset_indices=dataset_indices,
        fault_steps=fault_steps,
        trials_per_sample=args.trials_per_sample,
        shape=shape,
        seed=args.seed,
        num_new_tokens=args.num_new_tokens,
    )

    print(f"[{utc_now()}] Loading {args.dataset}/{args.split}", flush=True)
    dataset = load_dataset(
        args.dataset,
        revision=args.dataset_revision,
        split=args.split,
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    for dataset_index in dataset_indices:
        if not 0 <= dataset_index < len(dataset):
            raise IndexError(f"Dataset index {dataset_index} is invalid")

    immutable_metadata = {
        "schema_version": 2,
        "model_id": args.model_id,
        "model_revision": resolved_model_revision,
        "dataset": args.dataset,
        "dataset_split": args.split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_indices": list(dataset_indices),
        "fault_steps": list(fault_steps),
        "samples": args.samples,
        "trials_per_sample": args.trials_per_sample,
        "num_new_tokens": args.num_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "seed": args.seed,
        "campaign_kind": "smoke" if args.smoke else "reduced_pilot",
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": resolved_model_revision,
        "dataset_revision_requested": args.dataset_revision,
        "dataset_revision_resolved": args.dataset_revision,
        "qid_file_sha256": sha256_file(args.qid_file),
        "fault_step_file_sha256": sha256_file(args.fault_step_file),
        "batch_size": 1,
        "calibration_bound_scale": 2.0,
        "critical_projections": ["v_proj", "o_proj", "up_proj", "down_proj"],
        "model_shape": {
            "num_layers": shape.num_layers,
            "projection_widths": dict(shape.projection_widths),
        },
        "fault_bit_pool": list(range(16)),
        "projection_sampling_weights": dict(AUTHOR_PROJECTION_WEIGHTS),
        "semantic_sdc_definition": "text_divergent_and_no_complete_reference",
        "semantic_evaluator": "deterministic_normalized_reference_recall_proxy",
        "reference_answer_policy": "accept_any_nonempty_SQuAD_alias_at_recall_1",
        "empty_answer_policy": "empty_output_matches_only_empty_reference",
        "clean_eligibility_rule": "all_unprotected_clean_answers_correct_before_faults",
        "due_observability": "caught_python_exceptions_after_verified_injection_only",
        "repository_mode_scope": "zero_operator_only_not_full_software_stack",
        "fixed_length_no_eos_stop": True,
        "fault_model": "uniform_single_bit_fp16_xor",
        "pairing": "same_prebuilt_fault_manifest_across_all_modes",
        "modes": [mode.value for mode in MODES],
        "attention_implementation": "sdpa",
        "torch_dtype": "float16",
        "prompt_format": "context + two_newlines + question",
        "padding_side": "left",
        "truncation_side": tokenizer.truncation_side,
        "tokenizer_pad_token": tokenizer.pad_token,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "tokenizer_eos_token": tokenizer.eos_token,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "pad_token_added": added_pad_token,
        "tokenizer_length": len(tokenizer),
        "use_cache": bool(model.config.use_cache),
        "decoding": {"strategy": "greedy_argmax", "do_sample": False},
        "manifest_sha256": manifest_hash,
        "git_commit": git_commit(),
        "reproduction_code_sha256": reproduction_code_sha256(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
    }
    campaign_id = sha256_text(canonical_json(immutable_metadata))
    immutable_metadata["campaign_id"] = campaign_id

    metadata_path = output_dir / "metadata.json"
    clean_path = output_dir / "clean.jsonl"
    fault_path = output_dir / "faults.jsonl"
    check_path = output_dir / "isolation_checks.jsonl"
    result_paths = (clean_path, fault_path, check_path)
    if metadata_path.exists():
        if manifest_needs_create:
            raise ValueError("Campaign metadata exists but manifest is missing")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous.pop("created_at", None)
        if canonical_json(previous) != canonical_json(immutable_metadata):
            raise ValueError(
                "Existing campaign metadata differs; choose a new output dir"
            )
    else:
        if any(path.exists() for path in result_paths):
            raise ValueError(
                "Result JSONL exists without campaign metadata; choose a new output dir"
            )
        if manifest_needs_create:
            save_manifest(manifest_path, entries)
        atomic_write_json(
            metadata_path,
            {**immutable_metadata, "created_at": utc_now()},
        )

    existing_clean = load_jsonl(clean_path)
    existing_faults = load_jsonl(fault_path)
    existing_checks = load_jsonl(check_path)

    entries_by_sample: Dict[int, list[ManifestEntry]] = {
        sample_index: [] for sample_index in range(args.samples)
    }
    for entry in entries:
        entries_by_sample[entry.fault.sample_index].append(entry)

    for sample_index, dataset_index in enumerate(dataset_indices):
        run_id = f"clean:{sample_index}:unprotected"
        example = dataset[dataset_index]
        reference_answers = answer_references(example)
        prompt = squad_prompt(example)
        prompt_hash = sha256_text(prompt)
        if run_id in existing_clean:
            record = existing_clean[run_id]
            if record.get("prompt_sha256") != prompt_hash:
                raise ValueError("Preflight clean prompt hash mismatch")
            if record.get("campaign_id") != campaign_id:
                raise ValueError("Preflight clean campaign mismatch")
            if record.get("manifest_sha256") != manifest_hash:
                raise ValueError("Preflight clean manifest mismatch")
            if not record["clean_semantic_correct"]:
                raise AssertionError(
                    f"Sample {sample_index} failed clean eligibility"
                )
            continue

        prepared = prepare_prompt(
            tokenizer,
            prompt,
            device=torch.device("cuda"),
            max_prompt_tokens=args.max_prompt_tokens,
        )
        result, state = run_mode(
            model,
            tokenizer,
            prepared,
            mode=ProtectionMode.UNPROTECTED,
            faults=(),
            shape=shape,
            num_new_tokens=args.num_new_tokens,
        )
        record = clean_record(
            sample_index=sample_index,
            dataset_index=dataset_index,
            prompt_hash=prompt_hash,
            mode=ProtectionMode.UNPROTECTED,
            result=result,
            state=state,
            common_tokens=None,
            common_text=None,
            reference_answers=reference_answers,
            manifest_hash=manifest_hash,
            campaign_id=campaign_id,
        )
        append_jsonl(clean_path, record)
        existing_clean[run_id] = record
        if not record["clean_semantic_correct"]:
            raise AssertionError(
                f"Sample {sample_index} failed clean eligibility: {result.text!r}"
            )

        print(
            f"[preflight {sample_index + 1}/{args.samples}] semantic_ok=True",
            flush=True,
        )

    print("[preflight] all unprotected clean answers are eligible", flush=True)

    for sample_index, dataset_index in enumerate(dataset_indices):
        example = dataset[dataset_index]
        reference_answers = answer_references(example)
        prompt = squad_prompt(example)
        prompt_hash = sha256_text(prompt)
        prepared = prepare_prompt(
            tokenizer,
            prompt,
            device=torch.device("cuda"),
            max_prompt_tokens=args.max_prompt_tokens,
        )
        clean_tokens: Dict[ProtectionMode, Tuple[int, ...]] = {}
        clean_texts: Dict[ProtectionMode, str] = {}

        for mode in MODES:
            run_id = f"clean:{sample_index}:{mode.value}"
            if run_id in existing_clean:
                record = existing_clean[run_id]
                if record["prompt_sha256"] != prompt_hash:
                    raise ValueError("Resumed clean prompt hash mismatch")
                clean_tokens[mode] = tuple(record["generated_token_ids"])
                if record.get("campaign_id") != campaign_id:
                    raise ValueError("Resumed clean campaign mismatch")
                if record.get("manifest_sha256") != manifest_hash:
                    raise ValueError("Resumed clean manifest mismatch")
                clean_texts[mode] = str(record["decoded_text"])
                if mode is ProtectionMode.UNPROTECTED and not record[
                    "clean_semantic_correct"
                ]:
                    raise AssertionError("Unprotected clean answer is semantically wrong")
                continue

            common = clean_tokens.get(ProtectionMode.UNPROTECTED)
            common_text = clean_texts.get(ProtectionMode.UNPROTECTED)
            result, state = run_mode(
                model,
                tokenizer,
                prepared,
                mode=mode,
                faults=(),
                shape=shape,
                num_new_tokens=args.num_new_tokens,
            )
            clean_tokens[mode] = result.token_ids
            clean_texts[mode] = result.text
            record = clean_record(
                sample_index=sample_index,
                dataset_index=dataset_index,
                prompt_hash=prompt_hash,
                mode=mode,
                result=result,
                state=state,
                common_tokens=common,
                common_text=common_text,
                reference_answers=reference_answers,
                manifest_hash=manifest_hash,
                campaign_id=campaign_id,
            )
            append_jsonl(clean_path, record)
            existing_clean[run_id] = record
            if mode is ProtectionMode.UNPROTECTED and not record[
                "clean_semantic_correct"
            ]:
                raise AssertionError(
                    f"Sample {sample_index} clean answer is semantically wrong: "
                    f"{result.text!r}"
                )
            print(
                f"[clean {sample_index + 1}/{args.samples}] "
                f"{mode.value} token_drift={record['clean_token_drift']} "
                f"semantic_ok={record['clean_semantic_correct']}",
                flush=True,
            )

        common_tokens = clean_tokens[ProtectionMode.UNPROTECTED]
        common_text = clean_texts[ProtectionMode.UNPROTECTED]
        for entry in sorted(
            entries_by_sample[sample_index],
            key=lambda item: item.fault.trial_index,
        ):
            for mode in MODES:
                run_id = f"{entry.spec_id}:{mode.value}"
                if run_id in existing_faults:
                    record = existing_faults[run_id]
                    if record.get("campaign_id") != campaign_id:
                        raise ValueError("Resumed fault campaign mismatch")
                    if record.get("manifest_sha256") != manifest_hash:
                        raise ValueError("Resumed fault manifest mismatch")
                    continue
                state = RunState(mode=mode, faults=(entry.fault,))
                try:
                    result, state = run_mode(
                        model,
                        tokenizer,
                        prepared,
                        mode=mode,
                        faults=(entry.fault,),
                        shape=shape,
                        num_new_tokens=args.num_new_tokens,
                        state=state,
                    )
                    record = fault_record(
                        entry=entry,
                        prompt_hash=prompt_hash,
                        mode=mode,
                        result=result,
                        state=state,
                        common_tokens=common_tokens,
                        mode_clean_tokens=clean_tokens[mode],
                        common_text=common_text,
                        mode_clean_text=clean_texts[mode],
                        reference_answers=reference_answers,
                        manifest_hash=manifest_hash,
                        campaign_id=campaign_id,
                    )
                except Exception as error:
                    record = fault_error_record(
                        entry=entry,
                        prompt_hash=prompt_hash,
                        mode=mode,
                        state=state,
                        error=error,
                        error_phase=(
                            "verification"
                            if isinstance(error, CampaignInvariantError)
                            else "generation"
                        ),
                        common_tokens=common_tokens,
                        mode_clean_tokens=clean_tokens[mode],
                        common_text=common_text,
                        mode_clean_text=clean_texts[mode],
                        reference_answers=reference_answers,
                        manifest_hash=manifest_hash,
                        campaign_id=campaign_id,
                    )
                    append_jsonl(fault_path, record)
                    existing_faults[run_id] = record
                    print(
                        f"[fault s={sample_index} "
                        f"t={entry.fault.trial_index}] "
                        f"{mode.value} status={record['outcome']}",
                        flush=True,
                    )
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    elif "CUDA error" in str(error):
                        raise
                    continue
                append_jsonl(fault_path, record)
                existing_faults[run_id] = record
                print(
                    f"[fault s={sample_index} "
                    f"t={entry.fault.trial_index}] "
                    f"{mode.value} semantic_sdc="
                    f"{record['semantic_sdc_common']}",
                    flush=True,
                )

        check_id = f"isolation:{sample_index}"
        if check_id in existing_checks:
            check = existing_checks[check_id]
            if check.get("campaign_id") != campaign_id:
                raise ValueError("Resumed isolation campaign mismatch")
            if check.get("manifest_sha256") != manifest_hash:
                raise ValueError("Resumed isolation manifest mismatch")
            if check.get("prompt_sha256") != prompt_hash:
                raise ValueError("Resumed isolation prompt mismatch")
            if not check.get("passed"):
                raise CampaignInvariantError("Resumed isolation check failed")
        if check_id not in existing_checks:
            repeated, state = run_mode(
                model,
                tokenizer,
                prepared,
                mode=ProtectionMode.UNPROTECTED,
                faults=(),
                shape=shape,
                num_new_tokens=args.num_new_tokens,
            )
            comparison = compare_token_ids(
                repeated.token_ids,
                common_tokens,
            )
            check = {
                "schema_version": 2,
                "record_type": "isolation_check",
                "run_id": check_id,
                "campaign_id": campaign_id,
                "sample_index": sample_index,
                "dataset_index": dataset_index,
                "prompt_sha256": prompt_hash,
                "manifest_sha256": manifest_hash,
                "passed": comparison["equal"],
                "first_different_token": comparison[
                    "first_different_token"
                ],
                "repeated_token_sha256": repeated.token_sha256,
                "golden_token_sha256": token_sha256(common_tokens),
                "bounds_sha256": bounds_sha256(state),
                "completed_at": utc_now(),
            }
            append_jsonl(check_path, check)
            existing_checks[check_id] = check
            if not check["passed"]:
                raise AssertionError(
                    "Clean-fault-clean isolation check failed"
                )

    expected_clean = args.samples * len(MODES)
    expected_faults = (
        args.samples * args.trials_per_sample * len(MODES)
    )
    if len(existing_clean) != expected_clean:
        raise AssertionError(
            f"Expected {expected_clean} clean records, "
            f"got {len(existing_clean)}"
        )
    if len(existing_faults) != expected_faults:
        raise AssertionError(
            f"Expected {expected_faults} fault records, "
            f"got {len(existing_faults)}"
        )

    from summarize_v2 import (
        IncompleteCampaign,
        audit_campaign,
    )

    summary, invalid_count = audit_campaign(output_dir)
    summary_path = output_dir / "summary_v2.json"
    atomic_write_json(summary_path, summary)
    print(
        json.dumps(summary, indent=2, ensure_ascii=False),
        flush=True,
    )
    if invalid_count:
        raise IncompleteCampaign(
            f"{invalid_count} INVALID terminal rows; see {summary_path}"
        )
    print(f"[{utc_now()}] Campaign complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
