"""Metric logging and the non-finite loss policy.

JSON has no NaN or Infinity, so :func:`json_record` writes non-finite numbers
as ``null`` and records their original values under ``nonfinite_metrics``.

A non-finite loss raises :class:`NonFiniteLossError` and stops the run. Other
non-finite metrics, such as a clipped gradient norm, are recorded and training
continues.
"""

import json
import math
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def json_record(value: Any) -> Any:
    """Make a value JSON-safe, collecting non-finite numbers by their path."""
    nonfinite: dict[str, str] = {}

    def clean(item: Any, key: str = "") -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            nonfinite[key] = str(item)
            return None
        if isinstance(item, dict):
            return {name: clean(child, f"{key}.{name}" if key else str(name)) for name, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child, f"{key}[{index}]") for index, child in enumerate(item)]
        return item

    record = clean(value)
    if nonfinite:
        record["nonfinite_metrics"] = nonfinite
    return record


class NonFiniteLossError(ValueError):
    """Raised when a loss becomes NaN or Inf, so the run stops instead of drifting."""

    def __init__(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics
        names = ", ".join(f"{name}={value}" for name, value in metrics.items())
        super().__init__(f"loss 非有限值：{names}。请检查学习率、精度和训练数据。")


def check_finite_loss(metrics: dict[str, Any]) -> None:
    """Raise :class:`NonFiniteLossError` if any loss metric is not finite."""
    invalid = {
        name: value for name, value in metrics.items()
        if (name == "loss" or name.endswith("_loss")) and not math.isfinite(value)
    }
    if invalid:
        raise NonFiniteLossError(invalid)


def write_json(filename: str | Path, value: Any) -> None:
    """Write JSON atomically, so a crash cannot leave a half-written record."""
    filename = Path(filename)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(json_record(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(filename)


class JsonMetrics(TrainerCallback):
    """Append every Trainer log line to ``metrics.jsonl`` and police the loss."""

    def __init__(self, output: Path, session: str) -> None:
        self.filename = output / "metrics.jsonl"
        self.session = session

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if state.is_world_process_zero:
            record = {"session": self.session, "step": state.global_step, **logs}
            with self.filename.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_record(record), allow_nan=False) + "\n")
        check_finite_loss(logs)

    def on_step_end(self, args, state, control, **kwargs):
        # Keep the final checkpoint when saving is enabled, including when
        # save_steps exceeds max_steps. Respect config's explicit "no".
        if args.save_strategy != "no" and state.global_step >= state.max_steps:
            control.should_save = True
        return control
