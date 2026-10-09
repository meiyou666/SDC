#!/usr/bin/env python3
"""Finite dynamic parity check between the submitted protected models and ft2_formal."""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reproduction" / "src"))
sys.path.insert(0, str(REPO_ROOT / "performance" / "sigcode" / "modeling"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ft2_formal.adapters import discover_adapter
from ft2_formal.artifacts import atomic_write_artifact, sha256_file, utc_now
from ft2_formal.campaign import MAIN_MODES, _parameter_sentinel
from ft2_formal.decoding import fixed_greedy_generate, prepare_prompt
from ft2_formal.engine import FT2HookEngine
from ft2_formal.schema import FaultSpec, FaultType, fp16_bits_at, flip_fp16_bits
from ft2_formal.tasks import DATASETS, MODELS, load_task_dataset, prompt_and_references
from modeling_opt_protected import OPTForFICausalLM
from modeling_qwen2_protected import Qwen2ForFICausalLM


OUTPUT_DIR = REPO_ROOT / "reproduction" / "results" / "equivalence_validation_v1"
SELECTION = REPO_ROOT / "reproduction" / "results" / "formal_reduced_v1" / "selection.json"
MODEL_ORDER = ("opt_2_7b", "qwen2_math_7b")
AUTHOR_CLASSES = {"opt_2_7b": OPTForFICausalLM, "qwen2_math_7b": Qwen2ForFICausalLM}


def clean_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def tokenizer_for(model_key: str):
    tok = AutoTokenizer.from_pretrained(str(MODELS[model_key].snapshot), local_files_only=True)
    tok.padding_side = "left"
    tok.truncation_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def frozen_prompt(model_key: str, tokenizer: Any):
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    pair = selection["pairs"][f"{model_key}__squad_v2"]
    dataset_index = int(pair["dataset_indices"][0])
    split = DATASETS["squad_v2"].evaluation_split
    example = load_task_dataset("squad_v2")[split][dataset_index]
    prompt, _ = prompt_and_references("squad_v2", example)
    return dataset_index, prepare_prompt(tokenizer, prompt, device="cuda", max_input_tokens=1024)


def set_calibration(model_key: str, model: Any, enabled: bool) -> None:
    if model_key == "opt_2_7b":
        model.model.decoder.calibration_mode = enabled
    else:
        model.model.calibration_mode = enabled


def collect_author_bounds(model_key: str, model: Any) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    if model_key == "opt_2_7b":
        for i, layer in enumerate(model.model.decoder.layers):
            attn = layer.self_attn
            result[f"layer.{i}.v_proj"] = {"lower": float(attn.min_v_proj), "upper": float(attn.max_v_proj)}
            result[f"layer.{i}.out_proj"] = {"lower": float(attn.min_out_proj), "upper": float(attn.max_out_proj)}
            result[f"layer.{i}.fc2"] = {"lower": float(layer.min_fc2), "upper": float(layer.max_fc2)}
    else:
        for i, layer in enumerate(model.model.layers):
            attn, mlp = layer.self_attn, layer.mlp
            result[f"layer.{i}.v_proj"] = {"lower": float(attn.min_v), "upper": float(attn.max_v)}
            result[f"layer.{i}.o_proj"] = {"lower": float(attn.min_o), "upper": float(attn.max_o)}
            result[f"layer.{i}.up_proj"] = {"lower": float(mlp.min_up), "upper": float(mlp.max_up)}
            result[f"layer.{i}.down_proj"] = {"lower": float(mlp.min_down), "upper": float(mlp.max_down)}
    return result


def make_fault(model_key: str, dataset_index: int, width: int) -> FaultSpec:
    return FaultSpec(
        model_key=model_key,
        dataset_key="squad_v2",
        dataset_index=dataset_index,
        sample_position=0,
        trial_index=999,
        fault_type=FaultType.FP16_EXPONENT_BIT,
        target_step=1,
        layer_index=0,
        projection="v_proj",
        site_key="layer.0.v_proj",
        sequence_index=0,
        feature_index=0,
        expected_sequence_length=1,
        expected_out_features=width,
        bit_positions=(10,),
        campaign_seed=196,
    )


@torch.inference_mode()
def author_generate(model_key: str, model: Any, prepared: Any, fault: FaultSpec | None):
    input_ids = prepared.input_ids.detach().clone()
    kwargs: dict[str, Any] = {"attention_mask": prepared.attention_mask.detach().clone(), "use_cache": True}
    if model._supports_default_dynamic_cache():
        kwargs["past_key_values"] = DynamicCache()
    kwargs = model._get_initial_cache_position(input_ids, kwargs)
    state = {"step": -1}
    tokens: list[int] = []
    lengths: list[int] = []
    trace: dict[str, Any] | None = None
    handle = None
    if fault is not None:
        module = discover_adapter(model, model_key).sites_by_key[fault.site_key].module

        def inject(_module: Any, _inputs: Any, output: torch.Tensor):
            nonlocal trace
            if state["step"] != fault.target_step:
                return output
            if trace is not None:
                raise RuntimeError("author harness injected more than once")
            before = fp16_bits_at(output, fault.flat_index)
            changed = flip_fp16_bits(output, fault.flat_index, fault.bit_positions)
            after = fp16_bits_at(changed, fault.flat_index)
            trace = {
                "step": state["step"],
                "site_key": fault.site_key,
                "flat_index": fault.flat_index,
                "before_bits": before,
                "after_bits": after,
                "xor_mask": before ^ after,
                "hamming_distance": (before ^ after).bit_count(),
            }
            return changed

        handle = module.register_forward_hook(inject)
    try:
        for step in range(3):
            set_calibration(model_key, model, step == 0)
            state["step"] = step
            model_inputs = model.prepare_inputs_for_generation(input_ids, **kwargs)
            lengths.append(int(model_inputs["input_ids"].shape[1]))
            output = model(**model_inputs, return_dict=True)
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
            tokens.append(int(next_token.item()))
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            kwargs = model._update_model_kwargs_for_generation(
                output,
                kwargs,
                is_encoder_decoder=False,
                standardize_cache_format=False,
                num_new_tokens=1,
            )
    finally:
        if handle is not None:
            handle.remove()
    if fault is not None and trace is None:
        raise RuntimeError("author harness did not reach the fixed fault")
    return tokens, lengths, trace


def compare_bounds(author: dict[str, dict[str, float]], official: dict[str, Any]):
    missing = sorted(set(author) - set(official))
    extra = sorted(set(official) - set(author))
    mismatches: list[dict[str, Any]] = []
    max_delta = 0.0
    for key in sorted(set(author) & set(official)):
        for field in ("lower", "upper"):
            left, right = author[key][field], float(getattr(official[key], field))
            max_delta = max(max_delta, abs(left - right))
            if left != right:
                mismatches.append({"site": key, "field": field, "author": left, "official": right})
    return {
        "author_count": len(author),
        "official_count": len(official),
        "missing": missing,
        "extra": extra,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "max_abs_delta": max_delta,
        "passed": not missing and not extra and not mismatches,
    }


def validate_model(model_key: str) -> dict[str, Any]:
    spec = MODELS[model_key]
    tokenizer = tokenizer_for(model_key)
    dataset_index, prepared = frozen_prompt(model_key, tokenizer)
    print(f"[validation] load author {model_key}", flush=True)
    author = AUTHOR_CLASSES[model_key].from_pretrained(
        str(spec.snapshot),
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation=spec.attention_implementation,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    author_adapter = discover_adapter(author, model_key)
    author_sentinel_before = _parameter_sentinel(author_adapter)
    fault = make_fault(model_key, dataset_index, author_adapter.sites_by_key["layer.0.v_proj"].out_features)
    author_clean, author_clean_lengths, _ = author_generate(model_key, author, prepared, None)
    author_bounds = collect_author_bounds(model_key, author)
    author_fault, author_fault_lengths, author_trace = author_generate(model_key, author, prepared, fault)
    author_sentinel_after = _parameter_sentinel(author_adapter)
    author_backend = str(author.config._attn_implementation)
    del author_adapter, author
    clean_gpu()

    print(f"[validation] load official {model_key}", flush=True)
    official = AutoModelForCausalLM.from_pretrained(
        str(spec.snapshot),
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation=spec.attention_implementation,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    adapter = discover_adapter(official, model_key)
    official_sentinel_before = _parameter_sentinel(adapter)
    with FT2HookEngine(adapter) as engine:
        engine.start_inference(fault=None, protection=MAIN_MODES[1])
        official_clean = fixed_greedy_generate(official, tokenizer, prepared, engine, num_new_tokens=3)
        engine.start_inference(fault=fault, protection=MAIN_MODES[1])
        official_fault = fixed_greedy_generate(official, tokenizer, prepared, engine, num_new_tokens=3)
    official_sentinel_after = _parameter_sentinel(adapter)
    official_backend = str(official.config._attn_implementation)
    bounds = compare_bounds(author_bounds, dict(official_clean.engine_record.online_bounds))
    official_trace = official_fault.engine_record.injection_trace
    assert official_trace is not None and author_trace is not None
    injection = {
        "author": author_trace,
        "official": official_trace.to_dict(),
        "same_before_bits": author_trace["before_bits"] == official_trace.before_bits,
        "same_after_bits": author_trace["after_bits"] == official_trace.after_bits,
        "author_xor_correct": author_trace["xor_mask"] == (1 << 10),
        "official_xor_correct": official_trace.bit_flip_verified,
        "author_hamming": author_trace["hamming_distance"],
        "official_hamming": official_trace.hamming_distance,
        "official_injection_count": official_fault.engine_record.injection_count,
    }
    tokens = {
        "author_clean": author_clean,
        "official_clean": list(official_clean.token_ids),
        "author_fault": author_fault,
        "official_fault": list(official_fault.token_ids),
        "clean_equal": author_clean == list(official_clean.token_ids),
        "fault_equal": author_fault == list(official_fault.token_ids),
    }
    lengths = {
        "author_clean": author_clean_lengths,
        "author_fault": author_fault_lengths,
        "official_clean": list(official_clean.forward_input_lengths),
        "official_fault": list(official_fault.forward_input_lengths),
    }
    sentinels = {
        "author_before": author_sentinel_before,
        "author_after": author_sentinel_after,
        "official_before": official_sentinel_before,
        "official_after": official_sentinel_after,
    }
    passed = all(
        [
            bounds["passed"], tokens["clean_equal"], tokens["fault_equal"],
            injection["same_before_bits"], injection["same_after_bits"],
            injection["author_xor_correct"], injection["official_xor_correct"],
            injection["author_hamming"] == 1, injection["official_hamming"] == 1,
            injection["official_injection_count"] == 1,
            all(value == [1024, 1, 1] for value in lengths.values()),
            len(set(sentinels.values())) == 1,
            author_backend == official_backend == spec.attention_implementation,
        ]
    )
    result = {
        "model_key": model_key,
        "model_revision": spec.revision,
        "dataset_key": "squad_v2",
        "dataset_index": dataset_index,
        "prompt_sha256": prepared.prompt_sha256,
        "attention_backend": {"expected": spec.attention_implementation, "author": author_backend, "official": official_backend},
        "parameter_sentinel": {**sentinels, "all_equal": len(set(sentinels.values())) == 1},
        "bounds": bounds,
        "tokens": tokens,
        "input_lengths": lengths,
        "injection": injection,
        "passed": passed,
    }
    del adapter, official, tokenizer, prepared
    clean_gpu()
    print(f"[validation] {model_key}: {'PASS' if passed else 'FAIL'}", flush=True)
    return result


def unit_tests() -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "reproduction" / "src")
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(REPO_ROOT / "reproduction" / "tests"), "-v"]
    run = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"returncode": run.returncode, "passed": run.returncode == 0 and "Ran 41 tests" in run.stdout, "tail": run.stdout[-2000:]}


def main() -> int:
    started = utc_now()
    original = getattr(torch, "protectclamp", None)

    def paper_clamp(value: torch.Tensor, min: Any = None, max: Any = None):
        clamped = torch.clamp(value, min=min, max=max)
        return torch.where(torch.isnan(clamped), torch.zeros_like(clamped), clamped)

    torch.protectclamp = paper_clamp
    tests = unit_tests()
    models: dict[str, Any] = {}
    error = None
    try:
        for key in MODEL_ORDER:
            models[key] = validate_model(key)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if original is None:
            delattr(torch, "protectclamp")
        else:
            torch.protectclamp = original
        clean_gpu()

    sources = [
        Path(__file__),
        REPO_ROOT / "reproduction" / "src" / "ft2_formal" / "engine.py",
        REPO_ROOT / "reproduction" / "src" / "ft2_formal" / "schema.py",
        REPO_ROOT / "reproduction" / "src" / "ft2_formal" / "decoding.py",
        REPO_ROOT / "performance" / "sigcode" / "modeling" / "modeling_opt_protected.py",
        REPO_ROOT / "performance" / "sigcode" / "modeling" / "modeling_qwen2_protected.py",
        REPO_ROOT / "package" / "pytorch" / "aten" / "src" / "ATen" / "native" / "cuda" / "TensorCompare.cu",
    ]
    passed = tests["passed"] and error is None and set(models) == set(MODEL_ORDER) and all(v["passed"] for v in models.values())
    payload = {
        "schema_version": 1,
        "validation_id": "ft2_author_equivalence_v1",
        "status": "passed" if passed else "failed",
        "started_at": started,
        "ended_at": utc_now(),
        "scope": {
            "primary_semantics": "paper_clamp",
            "author_cuda_semantics": "repository_zero",
            "known_semantic_conflict": True,
            "batch_size": 1,
            "generated_tokens_per_model": 3,
            "note": "Parity uses the paper-required saturating clamp. The repository CUDA zeroes out-of-bound values and is recorded as a source contradiction.",
        },
        "unit_tests": tests,
        "models": models,
        "error": error,
        "source_files": [{"path": str(p.relative_to(REPO_ROOT)), "size": p.stat().st_size, "sha256": sha256_file(p)} for p in sources],
    }
    report = atomic_write_artifact(OUTPUT_DIR / "validation.json", payload)
    print(json.dumps({"status": report["status"], "artifact_sha256": report["artifact_sha256"], "error": error}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
