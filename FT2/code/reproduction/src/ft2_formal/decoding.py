from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from transformers.cache_utils import DynamicCache
from .engine import EngineRecord, FT2HookEngine, OfflineBoundsProfiler


@dataclass(frozen=True)
class PreparedPrompt:
    prompt: str
    prompt_sha256: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    unpadded_token_count: int

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.ndim != 2:
            raise ValueError("Prompt tensors must have shape [batch, sequence]")
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids and attention_mask shapes must match")
        if self.input_ids.shape[0] != 1:
            raise ValueError("FT2 requires batch size 1")


@dataclass(frozen=True)
class GenerationOutput:
    token_ids: Tuple[int, ...]
    text: str
    first_eos_step: Optional[int]
    forward_input_lengths: Tuple[int, ...]
    elapsed_seconds: float
    engine_record: EngineRecord

    @property
    def token_sha256(self) -> str:
        encoded = ",".join(str(token) for token in self.token_ids).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def prepare_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device | str,
    max_input_tokens: int = 1024,
) -> PreparedPrompt:
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    if tokenizer.padding_side != "left":
        raise ValueError("Tokenizer padding_side must be left")
    if tokenizer.truncation_side != "right":
        raise ValueError("Tokenizer truncation_side must be right")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id")
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_input_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return PreparedPrompt(
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        input_ids=input_ids,
        attention_mask=attention_mask,
        unpadded_token_count=int(attention_mask.sum().item()),
    )


def _eos_ids(tokenizer: Any) -> frozenset[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return frozenset()
    if isinstance(value, int):
        return frozenset((value,))
    return frozenset(int(item) for item in value)


@torch.inference_mode()
def fixed_greedy_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prepared: PreparedPrompt,
    engine: FT2HookEngine,
    *,
    num_new_tokens: int,
) -> GenerationOutput:
    if model.training:
        raise RuntimeError("Call model.eval() before generation")
    if num_new_tokens <= 0:
        raise ValueError("num_new_tokens must be positive")
    input_ids = prepared.input_ids.detach().clone()
    model_kwargs: dict[str, Any] = {
        "attention_mask": prepared.attention_mask.detach().clone(),
        "use_cache": True,
    }
    # Mirror transformers 4.42.4 GenerationMixin.generate(): Qwen2 requires a
    # Cache object from the first forward, while OPT keeps its legacy tuple cache.
    if model._supports_default_dynamic_cache():
        past = model_kwargs.get("past_key_values")
        if past is None:
            model_kwargs["past_key_values"] = DynamicCache()
        elif isinstance(past, tuple):
            model_kwargs["past_key_values"] = DynamicCache.from_legacy_cache(past)
    model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
    generated: list[int] = []
    input_lengths: list[int] = []
    first_eos_step: Optional[int] = None
    eos_ids = _eos_ids(tokenizer)
    uses_cuda = input_ids.device.type == "cuda"
    if uses_cuda:
        torch.cuda.synchronize(input_ids.device)
    started = time.perf_counter()

    for step in range(num_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        forward_ids = model_inputs.get("input_ids")
        if not isinstance(forward_ids, torch.Tensor):
            raise RuntimeError("prepare_inputs_for_generation returned no input_ids")
        actual_length = int(forward_ids.shape[1])
        expected_length = int(prepared.input_ids.shape[1]) if step == 0 else 1
        if actual_length != expected_length:
            raise RuntimeError(
                f"Forward step {step} uses {actual_length} tokens, "
                f"expected {expected_length}"
            )
        input_lengths.append(actual_length)
        engine.begin_step(step)
        outputs = model(**model_inputs, return_dict=True)
        if getattr(outputs, "past_key_values", None) is None:
            raise RuntimeError("Model did not return a KV cache")
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError("Model returned invalid causal LM logits")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.shape != (1, 1):
            raise RuntimeError("Greedy decoder produced an invalid token shape")
        token_id = int(next_token.item())
        generated.append(token_id)
        if first_eos_step is None and token_id in eos_ids:
            first_eos_step = step
        input_ids = torch.cat((input_ids, next_token), dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=False,
            standardize_cache_format=False,
            num_new_tokens=1,
        )

    if uses_cuda:
        torch.cuda.synchronize(prepared.input_ids.device)
    elapsed = time.perf_counter() - started
    record = engine.finish_inference()
    tokens = tuple(generated)
    if len(tokens) != num_new_tokens:
        raise AssertionError("Fixed decoder returned the wrong number of tokens")
    return GenerationOutput(
        token_ids=tokens,
        text=tokenizer.decode(list(tokens), skip_special_tokens=True),
        first_eos_step=first_eos_step,
        forward_input_lengths=tuple(input_lengths),
        elapsed_seconds=elapsed,
        engine_record=record,
    )


@torch.inference_mode()
def profile_prefill(
    model: torch.nn.Module,
    prepared: PreparedPrompt,
    profiler: OfflineBoundsProfiler,
) -> None:
    if model.training:
        raise RuntimeError("Call model.eval() before profiling")
    profiler.start_example()
    outputs = model(
        input_ids=prepared.input_ids,
        attention_mask=prepared.attention_mask,
        use_cache=False,
        return_dict=True,
    )
    if not isinstance(getattr(outputs, "logits", None), torch.Tensor):
        raise RuntimeError("Profiling forward returned no logits")
    profiler.finish_example()
