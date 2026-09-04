from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PreparedPrompt:
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    unpadded_token_count: int

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.ndim != 2:
            raise ValueError("Prompt tensors must have shape [batch, sequence]")
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids and attention_mask shapes must match")
        if self.input_ids.shape[0] != 1:
            raise ValueError("FT2 reduced reproduction requires batch size 1")


@dataclass(frozen=True)
class GenerationResult:
    token_ids: Tuple[int, ...]
    text: str
    first_eos_step: Optional[int]
    forward_input_lengths: Tuple[int, ...]
    elapsed_seconds: float

    @property
    def token_sha256(self) -> str:
        encoded = ",".join(str(token) for token in self.token_ids).encode(
            "ascii"
        )
        return hashlib.sha256(encoded).hexdigest()


def squad_prompt(example: Any) -> str:
    context = str(example["context"])
    question = str(example["question"])
    return context + "\n\n" + question


def prepare_prompt(
    tokenizer: Any,
    prompt: str,
    device: torch.device | str,
    max_prompt_tokens: int = 1024,
) -> PreparedPrompt:
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return PreparedPrompt(
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        unpadded_token_count=int(attention_mask.sum().item()),
    )


def _normalise_eos_ids(eos_token_id: Any) -> frozenset[int]:
    if eos_token_id is None:
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset((eos_token_id,))
    if isinstance(eos_token_id, Sequence):
        return frozenset(int(value) for value in eos_token_id)
    raise TypeError("eos_token_id must be int, sequence, or None")


@torch.inference_mode()
def generate_fixed(
    model: torch.nn.Module,
    tokenizer: Any,
    prepared: PreparedPrompt,
    num_new_tokens: int = 24,
) -> GenerationResult:
    if model.training:
        raise RuntimeError("Call model.eval() before generation")
    if num_new_tokens <= 0:
        raise ValueError("num_new_tokens must be positive")

    input_ids = prepared.input_ids.detach().clone()
    attention_mask = prepared.attention_mask.detach().clone()
    generated_tokens = []
    forward_input_lengths = []
    past_key_values = None
    eos_ids = _normalise_eos_ids(getattr(tokenizer, "eos_token_id", None))
    first_eos_step: Optional[int] = None

    uses_cuda = input_ids.device.type == "cuda"
    if uses_cuda:
        torch.cuda.synchronize(input_ids.device)
    started = time.perf_counter()

    for step in range(num_new_tokens):
        expected_length = input_ids.shape[1] if step == 0 else 1
        actual_length = int(input_ids.shape[1])
        if actual_length != expected_length:
            raise AssertionError(
                f"Forward step {step} expected input length "
                f"{expected_length}, got {actual_length}"
            )
        forward_input_lengths.append(actual_length)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("Model did not return past_key_values")
        if outputs.logits.ndim != 3 or outputs.logits.shape[0] != 1:
            raise RuntimeError("Unexpected causal LM logits shape")

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )
        token_id = int(next_token.item())
        generated_tokens.append(token_id)
        if first_eos_step is None and token_id in eos_ids:
            first_eos_step = step

        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(next_token)),
            dim=1,
        )
        input_ids = next_token
        past_key_values = outputs.past_key_values

    if uses_cuda:
        torch.cuda.synchronize(prepared.input_ids.device)
    elapsed = time.perf_counter() - started

    token_ids = tuple(generated_tokens)
    if len(token_ids) != num_new_tokens:
        raise AssertionError("Fixed generation returned wrong token count")

    text = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
    )
    return GenerationResult(
        token_ids=token_ids,
        text=text,
        first_eos_step=first_eos_step,
        forward_input_lengths=tuple(forward_input_lengths),
        elapsed_seconds=elapsed,
    )
