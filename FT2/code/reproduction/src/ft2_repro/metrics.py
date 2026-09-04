from __future__ import annotations

import hashlib
import re
import string
from collections import Counter
from math import comb, sqrt
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


def token_sha256(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token)) for token in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def compare_token_ids(
    actual: Sequence[int],
    expected: Sequence[int],
) -> Dict[str, Any]:
    actual_tuple = tuple(int(value) for value in actual)
    expected_tuple = tuple(int(value) for value in expected)
    maximum = max(len(actual_tuple), len(expected_tuple))
    differing = [
        index
        for index in range(maximum)
        if index >= len(actual_tuple)
        or index >= len(expected_tuple)
        or actual_tuple[index] != expected_tuple[index]
    ]
    return {
        "equal": not differing,
        "first_different_token": differing[0] if differing else None,
        "different_token_count": len(differing),
        "actual_token_sha256": token_sha256(actual_tuple),
        "expected_token_sha256": token_sha256(expected_tuple),
    }


def compare_decoded_text(actual: str, expected: str) -> Dict[str, Any]:
    return {
        "equal": actual == expected,
        "actual_text_sha256": hashlib.sha256(
            actual.encode("utf-8")
        ).hexdigest(),
        "expected_text_sha256": hashlib.sha256(
            expected.encode("utf-8")
        ).hexdigest(),
    }


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_squad_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(
        character
        for character in lowered
        if character not in string.punctuation
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def squad_reference_recall(
    prediction: str,
    reference: str,
) -> float:
    prediction_tokens = normalize_squad_answer(prediction).split()
    reference_tokens = normalize_squad_answer(reference).split()
    if not reference_tokens:
        return 1.0 if not prediction_tokens else 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    return sum(common.values()) / len(reference_tokens)


def squad_semantic_score(
    prediction: str,
    references: Sequence[str],
) -> Dict[str, Any]:
    if not references:
        raise ValueError("At least one SQuAD reference answer is required")
    recalls = tuple(
        squad_reference_recall(prediction, reference)
        for reference in references
    )
    maximum = max(recalls)
    matched_index = next(
        (index for index, recall in enumerate(recalls) if recall == 1.0),
        None,
    )
    return {
        "semantic_correct": matched_index is not None,
        "max_reference_recall": maximum,
        "matched_reference_index": matched_index,
        "reference_recalls": list(recalls),
        "normalized_prediction": normalize_squad_answer(prediction),
        "normalized_references": [
            normalize_squad_answer(reference)
            for reference in references
        ],
    }


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")

    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (
        proportion + z_squared / (2.0 * trials)
    ) / denominator
    half = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar(n10: int, n01: int) -> float:
    if n10 < 0 or n01 < 0:
        raise ValueError("counts must be non-negative")
    discordant = n10 + n01
    if discordant == 0:
        return 1.0

    smaller = min(n10, n01)
    lower_tail = (
        sum(comb(discordant, value) for value in range(smaller + 1))
        / (2 ** discordant)
    )
    return min(1.0, 2.0 * lower_tail)


def paired_binary_counts(
    records: Iterable[Mapping[str, Any]],
    mode_a: str,
    mode_b: str,
    outcome_field: str,
) -> Dict[str, int]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        spec_id = str(record["spec_id"])
        mode = str(record["mode"])
        if mode not in (mode_a, mode_b):
            continue
        grouped.setdefault(spec_id, {})[mode] = record

    result = {
        "n00": 0,
        "n01": 0,
        "n10": 0,
        "n11": 0,
        "excluded": 0,
    }
    for pair in grouped.values():
        if mode_a not in pair or mode_b not in pair:
            result["excluded"] += 1
            continue
        value_a = pair[mode_a].get(outcome_field)
        value_b = pair[mode_b].get(outcome_field)
        if not isinstance(value_a, bool) or not isinstance(value_b, bool):
            result["excluded"] += 1
            continue
        result[f"n{int(value_a)}{int(value_b)}"] += 1
    return result
