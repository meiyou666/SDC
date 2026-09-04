from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import datasets as datasets_package
import numpy as np
import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import ModelAdapter, discover_adapter
from .artifacts import (
    atomic_write_artifact,
    json_sha256,
    load_artifact,
    sha256_file,
    stable_run_id,
    tensor_ids_sha256,
    utc_now,
)
from .decoding import fixed_greedy_generate, prepare_prompt, profile_prefill
from .engine import FT2HookEngine, OfflineBoundsProfiler
from .manifest import (
    build_pair_manifest,
    load_manifest,
    manifest_sha256,
    save_manifest,
)
from .schema import Bounds, BoundsSource, Correction, FaultSpec, ProtectionSpec
from .tasks import (
    DATASET_CACHE,
    DATASETS,
    EVAL_DIR,
    HF_CACHE,
    MODELS,
    MODEL_DATASETS,
    REPO_ROOT,
    author_candidates,
    author_source_metadata,
    calibration_indices,
    load_task_dataset,
    prompt_and_references,
    score_output,
)


CONFIG_PATH = REPO_ROOT / "reproduction" / "config" / "formal_experiment_config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reproduction" / "results" / "pilot"
PAIR_ORDER = tuple(
    (model_key, dataset_key)
    for model_key in ("opt_2_7b", "qwen2_math_7b")
    for dataset_key in MODEL_DATASETS[model_key]
)
MAIN_MODES = (
    ProtectionSpec(BoundsSource.NONE, Correction.NONE, 2.0),
    ProtectionSpec(BoundsSource.FIRST_TOKEN, Correction.PAPER_CLAMP, 2.0),
    ProtectionSpec(BoundsSource.OFFLINE, Correction.PAPER_CLAMP, 2.0),
)


def pair_id(model_key: str, dataset_key: str) -> str:
    return f"{model_key}__{dataset_key}"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _file_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    result = []
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(
            {
                "path": _relative(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def _model_snapshot_files(model_key: str) -> list[Path]:
    root = MODELS[model_key].snapshot
    if not root.is_dir():
        raise FileNotFoundError(root)
    allowed = {
        ".bin",
        ".json",
        ".model",
        ".py",
        ".safetensors",
        ".txt",
        ".tiktoken",
    }
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]
    weight_suffixes = {".bin", ".safetensors"}
    if not any(path.suffix in weight_suffixes for path in paths):
        raise FileNotFoundError(
            f"No .bin or .safetensors weights under {root}"
        )
    return paths


def _dataset_files(dataset_key: str) -> list[Path]:
    spec = DATASETS[dataset_key]
    if dataset_key == "squad_v2":
        root = (
            DATASET_CACHE
            / "squad_v2"
            / "squad_v2"
            / "0.0.0"
            / "c9090cb5f89e659d"
        )
        return sorted(root.glob("*.arrow"))
    if dataset_key == "xtreme_mlqa_en_en":
        root = (
            DATASET_CACHE
            / "raw"
            / f"google_xtreme_{spec.revision}"
            / "MLQA.en.en"
        )
        return sorted(root.glob("*.parquet"))
    if dataset_key == "gsm8k":
        root = (
            DATASET_CACHE
            / "raw"
            / f"openai_gsm8k_{spec.revision}"
            / "main"
        )
        return sorted(root.glob("*.parquet"))
    raise ValueError(dataset_key)


def _code_files() -> list[Path]:
    package = REPO_ROOT / "reproduction" / "src" / "ft2_formal"
    result = sorted(package.glob("*.py"))
    result.append(REPO_ROOT / "reproduction" / "run_pilot.py")
    return [path for path in result if path.exists()]


def _command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def _runtime_metadata() -> dict[str, Any]:
    return {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets_package.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
        "driver": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": _command_output(["nvcc", "--version"]),
    }


def _parameter_sentinel(adapter: ModelAdapter) -> str:
    values: list[dict[str, Any]] = []
    for site in adapter.sites:
        weight = site.module.weight.detach()
        bits = weight.contiguous().view(torch.int16).reshape(-1)
        values.append(
            {
                "site": site.key,
                "first": int(bits[0].item()) & 0xFFFF,
                "last": int(bits[-1].item()) & 0xFFFF,
                "shape": list(weight.shape),
            }
        )
    return json_sha256(values)


def _input_metadata(tokenizer: Any, prompt: str, prepared: Any) -> dict[str, Any]:
    untruncated = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )["input_ids"]
    return {
        "prompt": prompt,
        "prompt_sha256": prepared.prompt_sha256,
        "input_ids_sha256": tensor_ids_sha256(prepared.input_ids),
        "attention_mask_sha256": tensor_ids_sha256(prepared.attention_mask),
        "padded_token_count": int(prepared.input_ids.shape[1]),
        "unpadded_token_count": prepared.unpadded_token_count,
        "untruncated_token_count": len(untruncated),
        "truncated": len(untruncated) > int(prepared.input_ids.shape[1]),
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "add_special_tokens": True,
    }


def _output_payload(output: Any) -> dict[str, Any]:
    return {
        "token_ids": list(output.token_ids),
        "token_sha256": output.token_sha256,
        "text": output.text,
        "first_eos_step": output.first_eos_step,
        "forward_input_lengths": list(output.forward_input_lengths),
        "elapsed_seconds": output.elapsed_seconds,
    }


class PilotCampaign:
    def __init__(self, output_root: str | Path = DEFAULT_OUTPUT_ROOT):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self._validate_frozen_config()
        self.campaign_lock = self._ensure_campaign_lock()
        self.campaign_fingerprint = self.campaign_lock["campaign_fingerprint"]

    def _validate_frozen_config(self) -> None:
        config = self.config
        if config["experiment"]["id"] != "ft2_rtx4090_reduced_v1":
            raise ValueError("Unexpected experiment ID")
        if int(config["experiment"]["seed"]) != 196:
            raise ValueError("Unexpected campaign seed")
        pilot = config["pilot"]
        expected = {
            "model_dataset_pairs": 5,
            "prompts_per_pair": 2,
            "faults_per_prompt_per_fault_type": 2,
            "fault_types": 3,
            "main_protection_modes": 3,
            "expected_fault_inferences": 180,
        }
        for key, value in expected.items():
            if int(pilot[key]) != value:
                raise ValueError(f"Frozen pilot setting changed: {key}")
        generation = config["generation"]
        if (
            int(generation["qa_generation_steps"]) != 60
            or int(generation["gsm8k_generation_steps"]) != 180
            or bool(generation["stop_on_eos"])
        ):
            raise ValueError("Frozen generation semantics changed")
        if int(config["protection"]["offline_bounds"]["calibration_examples_per_dataset"]) != 200:
            raise ValueError("Offline calibration count changed")

    def _immutable_identity(self) -> dict[str, Any]:
        model_files = {
            key: _file_records(_model_snapshot_files(key))
            for key in sorted(MODELS)
        }
        dataset_files = {
            key: _file_records(_dataset_files(key))
            for key in sorted(DATASETS)
        }
        author_files = {
            pair_id(model_key, dataset_key): author_source_metadata(
                model_key, dataset_key
            )
            for model_key, dataset_key in PAIR_ORDER
        }
        code_files = _file_records(_code_files())
        return {
            "schema_version": 1,
            "experiment_id": self.config["experiment"]["id"],
            "repo_commit_configured": self.config["experiment"]["repo_commit"],
            "repo_commit_observed": _command_output(["git", "rev-parse", "HEAD"]),
            "config_path": _relative(CONFIG_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "source_files": code_files,
            "models": {
                key: {
                    "model_id": spec.model_id,
                    "revision": spec.revision,
                    "snapshot": str(spec.snapshot),
                    "attention_implementation": spec.attention_implementation,
                    "files": model_files[key],
                }
                for key, spec in sorted(MODELS.items())
            },
            "datasets": {
                key: {
                    "dataset_id": spec.dataset_id,
                    "revision": spec.revision,
                    "config": spec.config,
                    "evaluation_split": spec.evaluation_split,
                    "calibration_split": spec.calibration_split,
                    "files": dataset_files[key],
                }
                for key, spec in sorted(DATASETS.items())
            },
            "author_selection_files": author_files,
            "seed": 196,
            "rng": "numpy.PCG64 with SHA256-derived per-spec seed",
            "max_input_tokens": 1024,
            "dtype": "float16",
            "batch_size": 1,
            "pilot_dimensions": {
                "pairs": 5,
                "prompts_per_pair": 2,
                "fault_types": 3,
                "trials_per_prompt_per_fault_type": 2,
                "modes": 3,
                "fault_specs": 60,
                "fault_runs": 180,
                "offline_examples_per_pair": 200,
            },
        }

    def _ensure_campaign_lock(self) -> dict[str, Any]:
        path = self.output_root / "campaign.json"
        immutable = self._immutable_identity()
        fingerprint = json_sha256(immutable)
        if path.exists():
            existing = load_artifact(path)
            if existing.get("campaign_fingerprint") != fingerprint:
                raise RuntimeError(
                    "Campaign identity changed; refusing to mix old and new results"
                )
            return existing
        payload = {
            "schema_version": 1,
            "campaign_fingerprint": fingerprint,
            "immutable": immutable,
            "runtime": _runtime_metadata(),
            "created_at": utc_now(),
        }
        return atomic_write_artifact(path, payload)

    def _assert_artifact_campaign(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("campaign_fingerprint") != self.campaign_fingerprint:
            raise RuntimeError("Artifact belongs to a different campaign")

    @contextmanager
    def loaded_model(
        self, model_key: str
    ) -> Iterator[tuple[Any, Any, ModelAdapter]]:
        spec = MODELS[model_key]
        print(f"[model] loading {model_key} from {spec.snapshot}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.snapshot),
            local_files_only=True,
        )
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(f"{model_key} has neither PAD nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(spec.snapshot),
            local_files_only=True,
            torch_dtype=torch.float16,
            attn_implementation=spec.attention_implementation,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        adapter = discover_adapter(model, model_key)
        before = _parameter_sentinel(adapter)
        try:
            yield model, tokenizer, adapter
        finally:
            after = _parameter_sentinel(adapter)
            if after != before:
                raise AssertionError(
                    f"Model parameter sentinel changed during {model_key} campaign"
                )
            del adapter
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[model] unloaded {model_key}", flush=True)

    def _preflight_path(self, model_key: str) -> Path:
        return self.output_root / "preflight" / f"{model_key}.json"

    def ensure_model_preflight(
        self,
        model_key: str,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
    ) -> dict[str, Any]:
        path = self._preflight_path(model_key)
        if path.exists():
            record = load_artifact(path)
            self._assert_artifact_campaign(record)
            return record
        dataset = load_task_dataset("squad_v2")
        candidate = author_candidates(model_key, "squad_v2")[0]
        example = dataset[DATASETS["squad_v2"].evaluation_split][
            candidate.dataset_index
        ]
        prompt, references = prompt_and_references("squad_v2", example)
        prepared = prepare_prompt(
            tokenizer, prompt, device="cuda", max_input_tokens=1024
        )
        native = model.generate(
            input_ids=prepared.input_ids,
            attention_mask=prepared.attention_mask,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        native_tokens = tuple(int(value) for value in native[0, -2:].tolist())
        none = MAIN_MODES[0]
        first = MAIN_MODES[1]
        with FT2HookEngine(adapter) as engine:
            engine.start_inference(fault=None, protection=none)
            custom = fixed_greedy_generate(
                model, tokenizer, prepared, engine, num_new_tokens=2
            )
            engine.start_inference(fault=None, protection=first)
            protected = fixed_greedy_generate(
                model, tokenizer, prepared, engine, num_new_tokens=2
            )
        if custom.token_ids != native_tokens:
            raise AssertionError(
                f"{model_key} fixed decoder diverges from transformers.generate"
            )
        if custom.forward_input_lengths != (1024, 1):
            raise AssertionError("KV cache did not reduce the second forward to 1 token")
        if len(protected.engine_record.online_bounds) != len(
            adapter.critical_sites
        ):
            raise AssertionError("Preflight did not cover all critical sites")
        payload = {
            "schema_version": 1,
            "campaign_fingerprint": self.campaign_fingerprint,
            "model_key": model_key,
            "model_revision": MODELS[model_key].revision,
            "candidate_dataset_index": candidate.dataset_index,
            "native_tokens": list(native_tokens),
            "custom_tokens": list(custom.token_ids),
            "forward_input_lengths": list(custom.forward_input_lengths),
            "candidate_sites": len(adapter.sites),
            "critical_sites": len(adapter.critical_sites),
            "first_token_bounds": len(protected.engine_record.online_bounds),
            "references": list(references),
            "status": "passed",
            "completed_at": utc_now(),
        }
        print(f"[preflight] {model_key} PASS", flush=True)
        return atomic_write_artifact(path, payload)

    def _screen_path(
        self, model_key: str, dataset_key: str, source_position: int
    ) -> Path:
        return (
            self.output_root
            / "screening"
            / model_key
            / dataset_key
            / f"{source_position:02d}.json"
        )

    def _load_screen_record(
        self, model_key: str, dataset_key: str, source_position: int
    ) -> dict[str, Any]:
        record = load_artifact(
            self._screen_path(model_key, dataset_key, source_position)
        )
        self._assert_artifact_campaign(record)
        if (
            record.get("model_key") != model_key
            or record.get("dataset_key") != dataset_key
            or int(record.get("source_position", -1)) != source_position
        ):
            raise RuntimeError("Screening artifact identity mismatch")
        return record

    def ensure_screening(
        self,
        model_key: str,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
    ) -> None:
        no_protection = MAIN_MODES[0]
        with FT2HookEngine(adapter) as engine:
            for dataset_key in MODEL_DATASETS[model_key]:
                dataset_spec = DATASETS[dataset_key]
                dataset = load_task_dataset(dataset_key)
                split = dataset_spec.evaluation_split
                correct_count = 0
                for candidate in author_candidates(model_key, dataset_key):
                    path = self._screen_path(
                        model_key, dataset_key, candidate.source_position
                    )
                    if path.exists():
                        existing = self._load_screen_record(
                            model_key, dataset_key, candidate.source_position
                        )
                        if existing["status"] == "completed" and existing[
                            "score"
                        ]["task_correct"]:
                            correct_count += 1
                    else:
                        example = dataset[split][candidate.dataset_index]
                        prompt, references = prompt_and_references(
                            dataset_key, example
                        )
                        prepared = prepare_prompt(
                            tokenizer,
                            prompt,
                            device="cuda",
                            max_input_tokens=1024,
                        )
                        started = utc_now()
                        engine.start_inference(
                            fault=None, protection=no_protection
                        )
                        try:
                            output = fixed_greedy_generate(
                                model,
                                tokenizer,
                                prepared,
                                engine,
                                num_new_tokens=dataset_spec.generation_steps,
                            )
                        except Exception as exc:
                            partial = engine.abort()
                            invalid = {
                                "schema_version": 1,
                                "campaign_fingerprint": self.campaign_fingerprint,
                                "status": "invalid_clean",
                                "model_key": model_key,
                                "dataset_key": dataset_key,
                                "source_position": candidate.source_position,
                                "dataset_index": candidate.dataset_index,
                                "target_step": candidate.target_step,
                                "started_at": started,
                                "ended_at": utc_now(),
                                "error": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                    "traceback": traceback.format_exc(),
                                },
                                "engine": partial.to_dict(),
                            }
                            atomic_write_artifact(path, invalid)
                            raise RuntimeError(
                                f"Clean screening failed for {model_key}/{dataset_key}"
                            ) from exc
                        score = score_output(
                            dataset_key, output.text, references
                        )
                        record = {
                            "schema_version": 1,
                            "campaign_fingerprint": self.campaign_fingerprint,
                            "status": "completed",
                            "model_key": model_key,
                            "model_revision": MODELS[model_key].revision,
                            "dataset_key": dataset_key,
                            "dataset_revision": dataset_spec.revision,
                            "dataset_split": split,
                            "dataset_fingerprint": dataset[split]._fingerprint,
                            "source_position": candidate.source_position,
                            "source_line": candidate.source_position + 1,
                            "dataset_index": candidate.dataset_index,
                            "target_step": candidate.target_step,
                            "input": _input_metadata(
                                tokenizer, prompt, prepared
                            ),
                            "references": list(references),
                            "output": _output_payload(output),
                            "score": score,
                            "engine": output.engine_record.to_dict(),
                            "started_at": started,
                            "ended_at": utc_now(),
                        }
                        existing = atomic_write_artifact(path, record)
                        if existing["score"]["task_correct"]:
                            correct_count += 1
                        print(
                            f"[screen] {model_key}/{dataset_key} "
                            f"{candidate.source_position + 1}/50 "
                            f"correct={correct_count}",
                            flush=True,
                        )
                    if dataset_key == "gsm8k" and correct_count >= 2:
                        break
                if dataset_key != "gsm8k":
                    missing = [
                        position
                        for position in range(50)
                        if not self._screen_path(
                            model_key, dataset_key, position
                        ).exists()
                    ]
                    if missing:
                        raise AssertionError(
                            f"Incomplete QA screening: {model_key}/{dataset_key}"
                        )
                if correct_count < 2:
                    raise RuntimeError(
                        f"Fewer than two clean-correct candidates for "
                        f"{model_key}/{dataset_key}"
                    )

    def _selection_path(self) -> Path:
        return self.output_root / "selection.json"

    def ensure_selection(self) -> dict[str, Any]:
        path = self._selection_path()
        if path.exists():
            selection = load_artifact(path)
            self._assert_artifact_campaign(selection)
            self._validate_selection(selection)
            return selection

        selected_positions: dict[str, list[int]] = {}
        for dataset_key in ("squad_v2", "xtreme_mlqa_en_en"):
            common = []
            for position in range(50):
                opt = self._load_screen_record(
                    "opt_2_7b", dataset_key, position
                )
                qwen = self._load_screen_record(
                    "qwen2_math_7b", dataset_key, position
                )
                if (
                    opt["status"] == "completed"
                    and qwen["status"] == "completed"
                    and opt["score"]["task_correct"]
                    and qwen["score"]["task_correct"]
                ):
                    if opt["dataset_index"] != qwen["dataset_index"]:
                        raise AssertionError("Shared QA qid mapping diverged")
                    common.append(position)
            if len(common) < 2:
                raise RuntimeError(
                    f"Fewer than two shared clean-correct rows for {dataset_key}"
                )
            selected_positions[dataset_key] = common[:2]

        gsm = []
        for position in range(50):
            path_for_row = self._screen_path(
                "qwen2_math_7b", "gsm8k", position
            )
            if not path_for_row.exists():
                continue
            record = self._load_screen_record(
                "qwen2_math_7b", "gsm8k", position
            )
            if (
                record["status"] == "completed"
                and record["score"]["task_correct"]
            ):
                gsm.append(position)
            if len(gsm) == 2:
                break
        if len(gsm) < 2:
            raise RuntimeError("Fewer than two Qwen GSM8K clean-correct rows")
        selected_positions["gsm8k"] = gsm

        pairs: dict[str, Any] = {}
        for model_key, dataset_key in PAIR_ORDER:
            positions = selected_positions[dataset_key]
            candidates = author_candidates(model_key, dataset_key)
            rows = [candidates[position] for position in positions]
            pairs[pair_id(model_key, dataset_key)] = {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "source_positions": positions,
                "dataset_indices": [row.dataset_index for row in rows],
                "target_steps": [row.target_step for row in rows],
                "screening_artifacts": [
                    _relative(
                        self._screen_path(model_key, dataset_key, position)
                    )
                    for position in positions
                ],
            }
        payload = {
            "schema_version": 1,
            "campaign_fingerprint": self.campaign_fingerprint,
            "policy": (
                "first two author-order clean-correct; QA positions shared "
                "between OPT and Qwen"
            ),
            "pairs": pairs,
            "created_at": utc_now(),
        }
        selection = atomic_write_artifact(path, payload)
        self._validate_selection(selection)
        print("[selection] locked 2 prompts for all 5 pairs", flush=True)
        return selection

    def _validate_selection(self, selection: Mapping[str, Any]) -> None:
        pairs = selection.get("pairs", {})
        expected_ids = {
            pair_id(model_key, dataset_key)
            for model_key, dataset_key in PAIR_ORDER
        }
        if set(pairs) != expected_ids:
            raise RuntimeError("Selection does not contain exactly five pairs")
        for identifier, entry in pairs.items():
            for field in (
                "source_positions",
                "dataset_indices",
                "target_steps",
            ):
                if len(entry[field]) != 2:
                    raise RuntimeError(
                        f"{identifier} selection must contain two prompts"
                    )

    def _bounds_path(self, model_key: str, dataset_key: str) -> Path:
        return (
            self.output_root
            / "offline_bounds"
            / f"{pair_id(model_key, dataset_key)}.json"
        )

    def ensure_offline_bounds(
        self,
        model_key: str,
        dataset_key: str,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
    ) -> tuple[dict[str, Bounds], dict[str, Any]]:
        path = self._bounds_path(model_key, dataset_key)
        if path.exists():
            artifact = load_artifact(path)
            self._assert_artifact_campaign(artifact)
            bounds = {
                key: Bounds.from_dict(value)
                for key, value in artifact["bounds"].items()
            }
            if len(bounds) != len(adapter.critical_sites):
                raise RuntimeError("Offline bounds cardinality mismatch")
            if artifact["bounds_sha256"] != json_sha256(artifact["bounds"]):
                raise RuntimeError("Offline bounds content hash mismatch")
            return bounds, artifact

        dataset_spec = DATASETS[dataset_key]
        dataset = load_task_dataset(dataset_key)
        split = dataset_spec.calibration_split
        indices = calibration_indices(dataset_key, count=200, seed=196)
        profiler = OfflineBoundsProfiler(adapter, scaling_factor=2.0)
        prompt_records = []
        profiler.install()
        try:
            for ordinal, index in enumerate(indices, start=1):
                example = dataset[split][index]
                prompt, _ = prompt_and_references(dataset_key, example)
                prepared = prepare_prompt(
                    tokenizer,
                    prompt,
                    device="cuda",
                    max_input_tokens=1024,
                )
                profile_prefill(model, prepared, profiler)
                prompt_records.append(
                    {
                        "ordinal": ordinal - 1,
                        "dataset_index": index,
                        "sample_id": str(example.get("id", index)),
                        "prompt_sha256": prepared.prompt_sha256,
                        "unpadded_token_count": prepared.unpadded_token_count,
                    }
                )
                if ordinal % 10 == 0 or ordinal == len(indices):
                    print(
                        f"[bounds] {model_key}/{dataset_key} "
                        f"{ordinal}/{len(indices)}",
                        flush=True,
                    )
        finally:
            profiler.remove()
        bounds = profiler.finalize(expected_examples=200)
        if len(bounds) != len(adapter.critical_sites):
            raise AssertionError("Offline profiler missed critical sites")
        serialized = {
            key: value.to_dict() for key, value in sorted(bounds.items())
        }
        artifact = atomic_write_artifact(
            path,
            {
                "schema_version": 1,
                "campaign_fingerprint": self.campaign_fingerprint,
                "provenance": "recomputed_offline_bounds_reduced_200",
                "model_key": model_key,
                "model_revision": MODELS[model_key].revision,
                "dataset_key": dataset_key,
                "dataset_revision": dataset_spec.revision,
                "dataset_split": split,
                "dataset_fingerprint": dataset[split]._fingerprint,
                "selection": {
                    "algorithm": "numpy.PCG64 uniform without replacement",
                    "seed": 196,
                    "count": 200,
                    "indices": list(indices),
                },
                "samples": prompt_records,
                "aggregate": "global extrema across all prompt tokens and features",
                "statistic_dtype": "float16 activations serialized as float",
                "scaling_factor": 2.0,
                "critical_site_count": len(adapter.critical_sites),
                "critical_site_keys": sorted(
                    site.key for site in adapter.critical_sites
                ),
                "bounds": serialized,
                "bounds_sha256": json_sha256(serialized),
                "completed_at": utc_now(),
            },
        )
        print(f"[bounds] saved {pair_id(model_key, dataset_key)}", flush=True)
        return bounds, artifact

    def _manifest_path(self, model_key: str, dataset_key: str) -> Path:
        return (
            self.output_root
            / "manifests"
            / f"{pair_id(model_key, dataset_key)}.json"
        )

    def ensure_manifest(
        self,
        model_key: str,
        dataset_key: str,
        adapter: ModelAdapter,
        selection: Mapping[str, Any],
    ) -> tuple[FaultSpec, ...]:
        entry = selection["pairs"][pair_id(model_key, dataset_key)]
        dataset_spec = DATASETS[dataset_key]
        expected = build_pair_manifest(
            adapter=adapter,
            dataset_key=dataset_key,
            dataset_indices=entry["dataset_indices"],
            target_steps=entry["target_steps"],
            trials_per_fault_type=2,
            generation_steps=dataset_spec.generation_steps,
            campaign_seed=196,
            max_input_tokens=1024,
        )
        if len(expected) != 12:
            raise AssertionError("Pilot pair manifest must have 12 specs")
        path = self._manifest_path(model_key, dataset_key)
        if path.exists():
            metadata, existing = load_manifest(path)
            if metadata.get("campaign_fingerprint") != self.campaign_fingerprint:
                raise RuntimeError("Manifest belongs to another campaign")
            if manifest_sha256(existing) != manifest_sha256(expected):
                raise RuntimeError("Frozen manifest differs from regenerated specs")
            return existing
        metadata = {
            "campaign_fingerprint": self.campaign_fingerprint,
            "pair_id": pair_id(model_key, dataset_key),
            "model_key": model_key,
            "model_revision": MODELS[model_key].revision,
            "dataset_key": dataset_key,
            "dataset_revision": dataset_spec.revision,
            "source_positions": entry["source_positions"],
            "dataset_indices": entry["dataset_indices"],
            "target_steps": entry["target_steps"],
            "author_sources": author_source_metadata(model_key, dataset_key),
            "campaign_seed": 196,
            "derived_seed_algorithm": "SHA256 first 128 bits -> numpy.PCG64",
            "candidate_sites": [
                {
                    "key": site.key,
                    "module_path": site.module_path,
                    "critical": site.critical,
                    "sampling_weight": site.sampling_weight,
                    "out_features": site.out_features,
                }
                for site in adapter.sites
            ],
        }
        save_manifest(path, expected, metadata)
        print(f"[manifest] saved {pair_id(model_key, dataset_key)}", flush=True)
        return expected

    def _control_path(
        self,
        model_key: str,
        dataset_key: str,
        source_position: int,
        mode_id: str,
    ) -> Path:
        return (
            self.output_root
            / "controls"
            / pair_id(model_key, dataset_key)
            / f"{source_position:02d}__{mode_id}.json"
        )

    def ensure_controls(
        self,
        model_key: str,
        dataset_key: str,
        model: Any,
        tokenizer: Any,
        engine: FT2HookEngine,
        bounds: Mapping[str, Bounds],
        bounds_artifact: Mapping[str, Any],
        selection: Mapping[str, Any],
    ) -> dict[tuple[int, str], dict[str, Any]]:
        dataset_spec = DATASETS[dataset_key]
        dataset = load_task_dataset(dataset_key)
        split = dataset_spec.evaluation_split
        entry = selection["pairs"][pair_id(model_key, dataset_key)]
        controls: dict[tuple[int, str], dict[str, Any]] = {}
        for source_position, dataset_index in zip(
            entry["source_positions"], entry["dataset_indices"]
        ):
            canonical = self._load_screen_record(
                model_key, dataset_key, source_position
            )
            if not canonical["score"]["task_correct"]:
                raise AssertionError("Selected canonical clean output is incorrect")
            example = dataset[split][dataset_index]
            prompt, references = prompt_and_references(dataset_key, example)
            prepared = prepare_prompt(
                tokenizer, prompt, device="cuda", max_input_tokens=1024
            )
            for protection in MAIN_MODES:
                mode_id = protection.mode_id
                path = self._control_path(
                    model_key, dataset_key, source_position, mode_id
                )
                if path.exists():
                    control = load_artifact(path)
                    self._assert_artifact_campaign(control)
                elif protection.bounds_source is BoundsSource.NONE:
                    control = atomic_write_artifact(
                        path,
                        {
                            "schema_version": 1,
                            "campaign_fingerprint": self.campaign_fingerprint,
                            "status": "completed",
                            "pair_id": pair_id(model_key, dataset_key),
                            "model_key": model_key,
                            "dataset_key": dataset_key,
                            "source_position": source_position,
                            "dataset_index": dataset_index,
                            "mode_id": mode_id,
                            "reused_screening_artifact": _relative(
                                self._screen_path(
                                    model_key,
                                    dataset_key,
                                    source_position,
                                )
                            ),
                            "input": canonical["input"],
                            "references": canonical["references"],
                            "output": canonical["output"],
                            "score": canonical["score"],
                            "engine": canonical["engine"],
                            "completed_at": utc_now(),
                        },
                    )
                else:
                    supplied = (
                        bounds
                        if protection.bounds_source is BoundsSource.OFFLINE
                        else None
                    )
                    engine.start_inference(
                        fault=None,
                        protection=protection,
                        offline_bounds=supplied,
                    )
                    started = utc_now()
                    try:
                        output = fixed_greedy_generate(
                            model,
                            tokenizer,
                            prepared,
                            engine,
                            num_new_tokens=dataset_spec.generation_steps,
                        )
                    except Exception as exc:
                        partial = engine.abort()
                        invalid = {
                            "schema_version": 1,
                            "campaign_fingerprint": self.campaign_fingerprint,
                            "status": "invalid_clean_control",
                            "pair_id": pair_id(model_key, dataset_key),
                            "source_position": source_position,
                            "mode_id": mode_id,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                            "engine": partial.to_dict(),
                            "started_at": started,
                            "ended_at": utc_now(),
                        }
                        atomic_write_artifact(path, invalid)
                        raise RuntimeError(
                            f"Clean control failed for {model_key}/{dataset_key}/{mode_id}"
                        ) from exc
                    score = score_output(
                        dataset_key, output.text, references
                    )
                    control = atomic_write_artifact(
                        path,
                        {
                            "schema_version": 1,
                            "campaign_fingerprint": self.campaign_fingerprint,
                            "status": "completed",
                            "pair_id": pair_id(model_key, dataset_key),
                            "model_key": model_key,
                            "model_revision": MODELS[model_key].revision,
                            "dataset_key": dataset_key,
                            "dataset_revision": dataset_spec.revision,
                            "source_position": source_position,
                            "dataset_index": dataset_index,
                            "mode_id": mode_id,
                            "bounds_artifact_sha256": (
                                bounds_artifact["artifact_sha256"]
                                if protection.bounds_source
                                is BoundsSource.OFFLINE
                                else None
                            ),
                            "input": _input_metadata(
                                tokenizer, prompt, prepared
                            ),
                            "references": list(references),
                            "output": _output_payload(output),
                            "score": score,
                            "engine": output.engine_record.to_dict(),
                            "started_at": started,
                            "ended_at": utc_now(),
                        },
                    )
                if control.get("status") != "completed":
                    raise RuntimeError(
                        f"Mode-matched clean control is incomplete: "
                        f"{model_key}/{dataset_key}/{mode_id}"
                    )
                controls[(source_position, mode_id)] = control
                clean_correct = bool(control["score"]["task_correct"])
                print(
                    f"[control] {model_key}/{dataset_key} "
                    f"sample={source_position} mode={mode_id} "
                    f"clean_correct={clean_correct}",
                    flush=True,
                )
        return controls

    def _run_path(
        self,
        model_key: str,
        dataset_key: str,
        spec_id: str,
        mode_id: str,
    ) -> Path:
        return (
            self.output_root
            / "runs"
            / pair_id(model_key, dataset_key)
            / f"{spec_id}__{mode_id}.json"
        )

    def _validate_existing_run(
        self,
        path: Path,
        spec: FaultSpec,
        mode_id: str,
    ) -> dict[str, Any]:
        record = load_artifact(path)
        self._assert_artifact_campaign(record)
        expected_id = stable_run_id(
            campaign_fingerprint=self.campaign_fingerprint,
            spec_id=spec.spec_id,
            mode_id=mode_id,
        )
        if (
            record.get("run_id") != expected_id
            or record.get("spec_id") != spec.spec_id
            or record.get("mode_id") != mode_id
            or record.get("terminal_status") not in {"completed", "due"}
        ):
            raise RuntimeError(f"Invalid existing run artifact: {path}")
        return record

    def _base_run_record(
        self,
        *,
        run_id: str,
        spec: FaultSpec,
        protection: ProtectionSpec,
        site: Any,
        canonical: Mapping[str, Any],
        control: Mapping[str, Any],
        bounds_artifact: Mapping[str, Any],
        started_at: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "campaign_fingerprint": self.campaign_fingerprint,
            "run_id": run_id,
            "spec_id": spec.spec_id,
            "pair_id": pair_id(spec.model_key, spec.dataset_key),
            "model_key": spec.model_key,
            "model_revision": MODELS[spec.model_key].revision,
            "dataset_key": spec.dataset_key,
            "dataset_revision": DATASETS[spec.dataset_key].revision,
            "dataset_split": DATASETS[spec.dataset_key].evaluation_split,
            "selection_rank": spec.sample_position,
            "source_position": canonical["source_position"],
            "dataset_index": spec.dataset_index,
            "mode_id": protection.mode_id,
            "protection": {
                "bounds_source": protection.bounds_source.value,
                "correction": protection.correction.value,
                "scaling_factor": protection.scaling_factor,
            },
            "offline_bounds_artifact_sha256": (
                bounds_artifact["artifact_sha256"]
                if protection.bounds_source is BoundsSource.OFFLINE
                else None
            ),
            "fault": spec.to_dict(),
            "target_site": {
                "key": site.key,
                "module_path": site.module_path,
                "critical": site.critical,
                "sampling_weight": site.sampling_weight,
            },
            "canonical_clean": {
                "artifact_sha256": canonical["artifact_sha256"],
                "output": canonical["output"],
                "score": canonical["score"],
            },
            "mode_clean": {
                "artifact_sha256": control["artifact_sha256"],
                "output": control["output"],
                "score": control["score"],
            },
            "started_at": started_at,
            "attempt": 1,
        }

    def run_pair(
        self,
        model_key: str,
        dataset_key: str,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
        engine: FT2HookEngine,
        bounds: Mapping[str, Bounds],
        bounds_artifact: Mapping[str, Any],
        selection: Mapping[str, Any],
    ) -> None:
        specs = self.ensure_manifest(
            model_key, dataset_key, adapter, selection
        )
        controls = self.ensure_controls(
            model_key,
            dataset_key,
            model,
            tokenizer,
            engine,
            bounds,
            bounds_artifact,
            selection,
        )
        dataset_spec = DATASETS[dataset_key]
        dataset = load_task_dataset(dataset_key)
        split = dataset_spec.evaluation_split
        completed = 0
        expected_runs = len(specs) * len(MAIN_MODES)
        for spec in specs:
            source_position = selection["pairs"][
                pair_id(model_key, dataset_key)
            ]["source_positions"][spec.sample_position]
            canonical = self._load_screen_record(
                model_key, dataset_key, source_position
            )
            example = dataset[split][spec.dataset_index]
            prompt, references = prompt_and_references(dataset_key, example)
            prepared = prepare_prompt(
                tokenizer, prompt, device="cuda", max_input_tokens=1024
            )
            if prepared.prompt_sha256 != canonical["input"]["prompt_sha256"]:
                raise AssertionError("Fault prompt differs from clean screening")
            site = adapter.sites_by_key[spec.site_key]
            for protection in MAIN_MODES:
                mode_id = protection.mode_id
                path = self._run_path(
                    model_key, dataset_key, spec.spec_id, mode_id
                )
                if path.exists():
                    self._validate_existing_run(path, spec, mode_id)
                    completed += 1
                    continue
                control = controls[(source_position, mode_id)]
                run_id = stable_run_id(
                    campaign_fingerprint=self.campaign_fingerprint,
                    spec_id=spec.spec_id,
                    mode_id=mode_id,
                )
                started = utc_now()
                base = self._base_run_record(
                    run_id=run_id,
                    spec=spec,
                    protection=protection,
                    site=site,
                    canonical=canonical,
                    control=control,
                    bounds_artifact=bounds_artifact,
                    started_at=started,
                )
                supplied = (
                    bounds
                    if protection.bounds_source is BoundsSource.OFFLINE
                    else None
                )
                engine.start_inference(
                    fault=spec,
                    protection=protection,
                    offline_bounds=supplied,
                )
                try:
                    output = fixed_greedy_generate(
                        model,
                        tokenizer,
                        prepared,
                        engine,
                        num_new_tokens=dataset_spec.generation_steps,
                    )
                except Exception as exc:
                    partial = engine.abort()
                    trace = partial.injection_trace
                    valid_fault = (
                        partial.injection_count == 1
                        and trace is not None
                        and trace.bit_flip_verified
                        and trace.hamming_distance
                        == len(spec.bit_positions)
                    )
                    scientific_due = (
                        valid_fault
                        and protection.bounds_source is BoundsSource.FIRST_TOKEN
                        and spec.target_step == 0
                        and isinstance(exc, ValueError)
                        and str(exc).startswith("Bounds values must be finite")
                    )
                    due_reason = (
                        "DUE_INVALID_FIRST_TOKEN_BOUNDS"
                        if scientific_due
                        else None
                    )
                    terminal = "due" if scientific_due else "invalid"
                    due_record = {
                        **base,
                        "terminal_status": terminal,
                        "due": scientific_due,
                        "due_reason": due_reason,
                        "classification": {
                            "outcome": "DUE" if scientific_due else "INVALID_RUN",
                            "exact_masked": False,
                            "semantic_masked": False,
                            "masked": False,
                            "sdc": False,
                            "due": scientific_due,
                        },
                        "engine": partial.to_dict(),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        "ended_at": utc_now(),
                    }
                    atomic_write_artifact(path, due_record)
                    if not scientific_due:
                        raise RuntimeError(
                            f"Framework-invalid run {run_id}; campaign stopped"
                        ) from exc
                    print(
                        f"[run] DUE {model_key}/{dataset_key} "
                        f"{completed + 1}/{expected_runs} {mode_id}",
                        flush=True,
                    )
                    completed += 1
                    continue

                record = output.engine_record
                if record.injection_count != 1 or record.injection_trace is None:
                    raise AssertionError("Completed fault run lacks one XOR trace")
                if (
                    protection.bounds_source is BoundsSource.NONE
                    and record.correction_elements != 0
                ):
                    raise AssertionError("No-protection run performed correction")
                if (
                    protection.bounds_source is BoundsSource.FIRST_TOKEN
                    and len(record.online_bounds) != len(adapter.critical_sites)
                ):
                    raise AssertionError("First-token bounds count is incomplete")
                if len(output.token_ids) != dataset_spec.generation_steps:
                    raise AssertionError("Fixed generation length changed")
                score = score_output(dataset_key, output.text, references)
                exact_text = output.text == control["output"]["text"]
                exact_tokens = (
                    list(output.token_ids) == control["output"]["token_ids"]
                )
                canonical_exact_text = (
                    output.text == canonical["output"]["text"]
                )
                canonical_exact_tokens = (
                    list(output.token_ids)
                    == canonical["output"]["token_ids"]
                )
                task_correct = bool(score["task_correct"])
                mode_clean_task_correct = bool(
                    control["score"]["task_correct"]
                )
                if exact_text and task_correct != mode_clean_task_correct:
                    raise AssertionError(
                        "Identical mode-clean text received a different score"
                    )
                evaluable = mode_clean_task_correct
                if not evaluable:
                    outcome = "NON_EVALUABLE_CLEAN_FAILURE"
                elif exact_text:
                    outcome = "MASKED_IDENTICAL"
                elif task_correct:
                    outcome = "MASKED_SEMANTIC"
                else:
                    outcome = "SDC"
                classification = {
                    "outcome": outcome,
                    "evaluable": evaluable,
                    "clean_status": (
                        "CLEAN_OK"
                        if mode_clean_task_correct
                        else "PROTECTION_CLEAN_REGRESSION"
                    ),
                    "exact_masked": evaluable and exact_text,
                    "semantic_masked": (
                        evaluable and (not exact_text) and task_correct
                    ),
                    "masked": evaluable and task_correct,
                    "sdc": evaluable and (not task_correct),
                    "due": False,
                    "exact_text_match_mode_clean": exact_text,
                    "exact_token_match_mode_clean": exact_tokens,
                    "exact_text_match_canonical_clean": canonical_exact_text,
                    "exact_token_match_canonical_clean": canonical_exact_tokens,
                    "faulty_task_correct": task_correct,
                    "canonical_clean_task_correct": bool(
                        canonical["score"]["task_correct"]
                    ),
                    "mode_clean_task_correct": mode_clean_task_correct,
                }
                completed_record = {
                    **base,
                    "terminal_status": "completed",
                    "due": False,
                    "output": _output_payload(output),
                    "score": score,
                    "classification": classification,
                    "engine": record.to_dict(),
                    "online_bounds_sha256": (
                        json_sha256(
                            {
                                key: value.to_dict()
                                for key, value in sorted(
                                    record.online_bounds.items()
                                )
                            }
                        )
                        if record.online_bounds
                        else None
                    ),
                    "ended_at": utc_now(),
                }
                atomic_write_artifact(path, completed_record)
                completed += 1
                print(
                    f"[run] {model_key}/{dataset_key} {completed}/{expected_runs} "
                    f"{spec.fault_type.value} {mode_id} {outcome}",
                    flush=True,
                )
        if completed != expected_runs:
            raise AssertionError(
                f"{model_key}/{dataset_key} has {completed}, expected {expected_runs} runs"
            )

    def run_model_faults(
        self,
        model_key: str,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
        selection: Mapping[str, Any],
    ) -> None:
        artifacts: dict[str, tuple[dict[str, Bounds], dict[str, Any]]] = {}
        for dataset_key in MODEL_DATASETS[model_key]:
            artifacts[dataset_key] = self.ensure_offline_bounds(
                model_key, dataset_key, model, tokenizer, adapter
            )
        with FT2HookEngine(adapter) as engine:
            for dataset_key in MODEL_DATASETS[model_key]:
                bounds, artifact = artifacts[dataset_key]
                self.run_pair(
                    model_key,
                    dataset_key,
                    model,
                    tokenizer,
                    adapter,
                    engine,
                    bounds,
                    artifact,
                    selection,
                )

    def _model_fault_stage_complete(self, model_key: str) -> bool:
        for dataset_key in MODEL_DATASETS[model_key]:
            if not self._bounds_path(model_key, dataset_key).exists():
                return False
            manifest_path = self._manifest_path(model_key, dataset_key)
            if not manifest_path.exists():
                return False
            try:
                _, specs = load_manifest(manifest_path)
            except Exception:
                return False
            if len(specs) != 12:
                return False
            for spec in specs:
                for protection in MAIN_MODES:
                    path = self._run_path(
                        model_key,
                        dataset_key,
                        spec.spec_id,
                        protection.mode_id,
                    )
                    if not path.exists():
                        return False
                    try:
                        self._validate_existing_run(
                            path, spec, protection.mode_id
                        )
                    except Exception:
                        return False
        return True

    def run(self) -> dict[str, Any]:
        selection_path = self._selection_path()
        selection: dict[str, Any]
        if not selection_path.exists():
            with self.loaded_model("opt_2_7b") as (
                model,
                tokenizer,
                adapter,
            ):
                self.ensure_model_preflight(
                    "opt_2_7b", model, tokenizer, adapter
                )
                self.ensure_screening(
                    "opt_2_7b", model, tokenizer, adapter
                )
            del model, tokenizer, adapter
            gc.collect()
            torch.cuda.empty_cache()
            with self.loaded_model("qwen2_math_7b") as (
                model,
                tokenizer,
                adapter,
            ):
                self.ensure_model_preflight(
                    "qwen2_math_7b", model, tokenizer, adapter
                )
                self.ensure_screening(
                    "qwen2_math_7b", model, tokenizer, adapter
                )
                selection = self.ensure_selection()
                self.run_model_faults(
                    "qwen2_math_7b",
                    model,
                    tokenizer,
                    adapter,
                    selection,
                )
            del model, tokenizer, adapter
            gc.collect()
            torch.cuda.empty_cache()
        else:
            selection = self.ensure_selection()
            if not self._model_fault_stage_complete("qwen2_math_7b"):
                with self.loaded_model("qwen2_math_7b") as (
                    model,
                    tokenizer,
                    adapter,
                ):
                    self.ensure_model_preflight(
                        "qwen2_math_7b", model, tokenizer, adapter
                    )
                    self.run_model_faults(
                        "qwen2_math_7b",
                        model,
                        tokenizer,
                        adapter,
                        selection,
                    )
                del model, tokenizer, adapter
                gc.collect()
                torch.cuda.empty_cache()

        if not self._model_fault_stage_complete("opt_2_7b"):
            with self.loaded_model("opt_2_7b") as (
                model,
                tokenizer,
                adapter,
            ):
                self.ensure_model_preflight(
                    "opt_2_7b", model, tokenizer, adapter
                )
                self.run_model_faults(
                    "opt_2_7b",
                    model,
                    tokenizer,
                    adapter,
                    selection,
                )
            del model, tokenizer, adapter
            gc.collect()
            torch.cuda.empty_cache()

        from .audit import audit_pilot

        return audit_pilot(
            self.output_root,
            campaign_fingerprint=self.campaign_fingerprint,
            write_outputs=True,
        )
