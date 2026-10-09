from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ft2_repro.metrics import squad_semantic_score


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "performance" / "sigcode" / "evaluation"
DATASET_CACHE = Path("/mnt/ft2-data/cache/datasets")
HF_CACHE = Path("/mnt/ft2-data/cache/huggingface/hub")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    snapshot: Path
    attention_implementation: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_id: str
    revision: str
    config: str | None
    evaluation_split: str
    calibration_split: str
    qid_file: str
    step_files: Mapping[str, str]
    generation_steps: int
    prompt_family: str


@dataclass(frozen=True)
class AuthorCandidate:
    source_position: int
    dataset_index: int
    target_step: int

    def to_dict(self) -> dict[str, int]:
        return {
            "source_position": self.source_position,
            "source_line": self.source_position + 1,
            "dataset_index": self.dataset_index,
            "target_step": self.target_step,
        }


MODELS = {
    "opt_2_7b": ModelSpec(
        key="opt_2_7b",
        model_id="facebook/opt-2.7b",
        revision="905a4b602cda5c501f1b3a2650a4152680238254",
        snapshot=HF_CACHE
        / "models--facebook--opt-2.7b"
        / "snapshots"
        / "905a4b602cda5c501f1b3a2650a4152680238254",
        attention_implementation="eager",
    ),
    "qwen2_math_7b": ModelSpec(
        key="qwen2_math_7b",
        model_id="Qwen/Qwen2-Math-7B",
        revision="47a44ff4136da8960adbab02b2326787086bcf6c",
        snapshot=HF_CACHE
        / "models--Qwen--Qwen2-Math-7B"
        / "snapshots"
        / "47a44ff4136da8960adbab02b2326787086bcf6c",
        attention_implementation="sdpa",
    ),
}


DATASETS = {
    "squad_v2": DatasetSpec(
        key="squad_v2",
        dataset_id="rajpurkar/squad_v2",
        revision="3ffb306f725f7d2ce8394bc1873b24868140c412",
        config=None,
        evaluation_split="validation",
        calibration_split="train",
        qid_file="qidsquadfinal.txt",
        step_files={
            "opt_2_7b": "squadrandomtokenopt2.7b.txt",
            "qwen2_math_7b": "squadrandomtokenqwen.txt",
        },
        generation_steps=60,
        prompt_family="extractive_qa",
    ),
    "xtreme_mlqa_en_en": DatasetSpec(
        key="xtreme_mlqa_en_en",
        dataset_id="google/xtreme",
        revision="ec5f1f46e9af79639a90684a7a70a956c4998f04",
        config="MLQA.en.en",
        evaluation_split="validation",
        calibration_split="test",
        qid_file="qidxtremefinal.txt",
        step_files={
            "opt_2_7b": "xtremerandomtokenopt2.7b.txt",
            "qwen2_math_7b": "xtremerandomtokenqwen.txt",
        },
        generation_steps=60,
        prompt_family="extractive_qa",
    ),
    "gsm8k": DatasetSpec(
        key="gsm8k",
        dataset_id="openai/gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
        config="main",
        evaluation_split="test",
        calibration_split="train",
        qid_file="qidgsmfinal.txt",
        step_files={"qwen2_math_7b": "gsmrandomtokenqwen.txt"},
        generation_steps=180,
        prompt_family="math",
    ),
}


MODEL_DATASETS = {
    "opt_2_7b": ("squad_v2", "xtreme_mlqa_en_en"),
    "qwen2_math_7b": ("squad_v2", "xtreme_mlqa_en_en", "gsm8k"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_int_file(path: Path) -> Tuple[int, ...]:
    values = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} is not an integer") from exc
    return tuple(values)


def author_candidates(model_key: str, dataset_key: str) -> Tuple[AuthorCandidate, ...]:
    spec = DATASETS[dataset_key]
    try:
        step_file = spec.step_files[model_key]
    except KeyError as exc:
        raise ValueError(f"{model_key} does not evaluate {dataset_key}") from exc
    qid_path = EVAL_DIR / spec.qid_file
    step_path = EVAL_DIR / step_file
    qids = _read_int_file(qid_path)
    steps = _read_int_file(step_path)
    if len(qids) != 50 or len(steps) != 50:
        raise ValueError(
            f"Expected 50 author qids and steps, got {len(qids)} and {len(steps)}"
        )
    result = tuple(
        AuthorCandidate(position, int(qid), int(step))
        for position, (qid, step) in enumerate(zip(qids, steps))
    )
    bad = [item for item in result if not 0 <= item.target_step < spec.generation_steps]
    if bad:
        raise ValueError(f"Author fault steps are out of range: {bad}")
    return result


def author_source_metadata(model_key: str, dataset_key: str) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    qid_path = EVAL_DIR / spec.qid_file
    step_path = EVAL_DIR / spec.step_files[model_key]
    return {
        "qid_file": str(qid_path.relative_to(REPO_ROOT)),
        "qid_file_sha256": sha256_file(qid_path),
        "step_file": str(step_path.relative_to(REPO_ROOT)),
        "step_file_sha256": sha256_file(step_path),
    }


@lru_cache(maxsize=None)
def load_task_dataset(dataset_key: str):
    from datasets import load_dataset

    spec = DATASETS[dataset_key]
    if dataset_key == "squad_v2":
        from datasets import Dataset, DatasetDict
        root = (
            DATASET_CACHE
            / "squad_v2"
            / "squad_v2"
            / "0.0.0"
            / "c9090cb5f89e659d"
        )
        return DatasetDict({
            "train": Dataset.from_file(str(root / "squad_v2-train.arrow")),
            "validation": Dataset.from_file(str(root / "squad_v2-validation.arrow")),
        })
    if dataset_key == "xtreme_mlqa_en_en":
        root = (
            DATASET_CACHE
            / "raw"
            / f"google_xtreme_{spec.revision}"
            / "MLQA.en.en"
        )
        return load_dataset(
            "parquet",
            data_files={
                "validation": str(root / "validation-00000-of-00001.parquet"),
                "test": str(root / "test-00000-of-00001.parquet"),
            },
            cache_dir=str(DATASET_CACHE),
        )
    if dataset_key == "gsm8k":
        root = (
            DATASET_CACHE
            / "raw"
            / f"openai_gsm8k_{spec.revision}"
            / "main"
        )
        return load_dataset(
            "parquet",
            data_files={
                "train": str(root / "train-00000-of-00001.parquet"),
                "test": str(root / "test-00000-of-00001.parquet"),
            },
            cache_dir=str(DATASET_CACHE),
        )
    raise ValueError(f"Unknown dataset_key: {dataset_key}")


def prompt_and_references(
    dataset_key: str,
    example: Mapping[str, Any],
) -> tuple[str, Tuple[str, ...]]:
    spec = DATASETS[dataset_key]
    if spec.prompt_family == "extractive_qa":
        answers = example["answers"]["text"]
        references = tuple(str(value) for value in answers) or ("",)
        return (
            str(example["context"]) + "\n\n" + str(example["question"]),
            references,
        )
    answer = str(example["answer"])
    gold = answer.rsplit("####", 1)[-1].strip()
    return str(example["question"]) + "\n", (gold,)


_NUMBER = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"(?:/[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))?"
)


def _canonical_number(value: str) -> str | None:
    cleaned = value.replace(",", "").strip()
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", 1)
            number = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        else:
            number = Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return f"{number.numerator}/{number.denominator}"


def extract_last_number(text: str) -> str | None:
    matches = _NUMBER.findall(text)
    return _canonical_number(matches[-1]) if matches else None


def score_output(
    dataset_key: str,
    output: str,
    references: Sequence[str],
) -> dict[str, Any]:
    if DATASETS[dataset_key].prompt_family == "extractive_qa":
        score = squad_semantic_score(output, references)
        return {
            "task_correct": bool(score["semantic_correct"]),
            "evaluator": "normalized_complete_reference_recall_proxy",
            **score,
        }
    predicted = extract_last_number(output)
    gold = _canonical_number(references[0])
    return {
        "task_correct": predicted is not None and predicted == gold,
        "evaluator": "last_numeric_answer_match",
        "predicted_number": predicted,
        "gold_number": gold,
    }


def classify_fault(
    clean_text: str,
    faulty_text: str,
    faulty_score: Mapping[str, Any],
) -> dict[str, bool]:
    exact = faulty_text == clean_text
    correct = bool(faulty_score["task_correct"])
    return {
        "exact_masked": exact,
        "semantic_masked": correct,
        "changed_but_correct": (not exact) and correct,
        "sdc": not correct,
        "due": False,
    }


def calibration_indices(
    dataset_key: str,
    *,
    count: int = 200,
    seed: int = 196,
) -> Tuple[int, ...]:
    dataset = load_task_dataset(dataset_key)
    split = DATASETS[dataset_key].calibration_split
    size = len(dataset[split])
    if not 0 < count <= size:
        raise ValueError(f"Cannot draw {count} examples from {dataset_key}/{split}")
    rng = np.random.Generator(np.random.PCG64(seed))
    return tuple(int(value) for value in rng.choice(size, size=count, replace=False))
