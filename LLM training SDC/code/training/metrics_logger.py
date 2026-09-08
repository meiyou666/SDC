"""Local, offline metrics logger that replaces wandb.

Writes one JSON object per line to `<save_dir>/metrics.jsonl` and a final
`<save_dir>/summary.json` at the end of the run.  All writes are guarded to
rank 0 in distributed settings.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class MetricsLogger:
    def __init__(self, save_dir: Optional[str], rank: int = 0, run_name: Optional[str] = None):
        self.rank = rank
        self.run_name = run_name or "offline_run"
        self.start_time = time.time()

        if save_dir is None or rank != 0:
            self.metrics_path = None
            self.summary_path = None
            return

        os.makedirs(save_dir, exist_ok=True)
        self.metrics_path = Path(save_dir) / "metrics.jsonl"
        self.summary_path = Path(save_dir) / "summary.json"

        # Truncate existing metrics file at run start.
        self.metrics_path.write_text("")

    def log(self, step: int, **metrics: Any) -> None:
        """Append a metrics record for the given step."""
        if self.metrics_path is None:
            return

        record = {"_step": step, "_time": time.time() - self.start_time}
        for key, value in metrics.items():
            record[key] = self._convert(value)

        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def log_dict(self, step: int, metrics: Dict[str, Any]) -> None:
        """Same as log() but accepts a single dict."""
        if self.metrics_path is None:
            return

        record = {"_step": step, "_time": time.time() - self.start_time}
        record.update({k: self._convert(v) for k, v in metrics.items()})

        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def summary(self, **kwargs: Any) -> None:
        """Write (or overwrite) the final summary JSON."""
        if self.summary_path is None:
            return

        summary = {
            "run_name": self.run_name,
            "total_time": time.time() - self.start_time,
        }
        summary.update({k: self._convert(v) for k, v in kwargs.items()})

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            f.flush()

    def _convert(self, value: Any) -> Any:
        """Make PyTorch scalars / numpy values JSON-serializable."""
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, float):
            if value != value:  # NaN
                return None
            if value == float("inf"):
                return "inf"
            if value == float("-inf"):
                return "-inf"
        return value


def log_run_config(save_dir: Optional[str], run_config: Dict[str, Any], rank: int = 0) -> None:
    """Persist the run configuration as JSON."""
    if save_dir is None or rank != 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    config_path = Path(save_dir) / "run_config.json"

    def _make_serializable(obj: Any) -> Any:
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, float):
            if obj != obj:
                return None
            if obj == float("inf"):
                return "inf"
            if obj == float("-inf"):
                return "-inf"
        return obj

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(run_config), f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to write run_config.json: {e}")
