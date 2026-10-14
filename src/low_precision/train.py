"""Run the shared Qwen task with Transformer Engine FP8 or NVFP4 Linear."""

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import TrainerCallback

from training.config import load_config
from training.metrics import write_json
from training.runner import require_device, resolve_runtime_config, train

from .adapter import install, metadata


def build_key(precision):
    """Include the recipe, adapter source and installed libraries in resumes."""
    details = {
        **metadata(precision),
        "torch": torch.__version__,
        "transformer_engine": importlib.metadata.version("transformer-engine"),
    }
    digest = hashlib.sha256(json.dumps(details, sort_keys=True).encode())
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


class PrecisionRecord(TrainerCallback):
    """Add the selected numerical path to the ordinary run record."""

    def __init__(self, precision):
        self.precision = precision

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero:
            filename = Path(args.output_dir) / "run.json"
            record = json.loads(filename.read_text())
            record["low_precision"] = {
                **metadata(self.precision),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformer_engine": importlib.metadata.version("transformer-engine"),
            }
            write_json(filename, record)


def main():
    parser = argparse.ArgumentParser(description="Qwen FP8 / NVFP4 继续预训练")
    parser.add_argument("--precision", choices=("fp8", "fp4"), required=True)
    parser.add_argument("--config", default="configs/training/qwen3_06b.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = resolve_runtime_config(load_config(args.config))
    if config.training.device != "cuda" or config.training.precision != "bf16":
        parser.error("低精度训练使用 NVIDIA 容器，公共 training.precision 保持 bf16。")
    require_device("cuda")
    config = replace(config, training=replace(config.training, output_dir=args.output_dir))
    identity = f"transformer-engine:{args.precision}:{build_key(args.precision)}"
    print(json.dumps(metadata(args.precision), ensure_ascii=False, indent=2))
    train(
        config, resume=args.resume, offline=args.offline,
        model_transform=install, transform_id=identity,
        transform_params={"precision": args.precision},
        callbacks=[PrecisionRecord(args.precision)],
    )


if __name__ == "__main__":
    main()
