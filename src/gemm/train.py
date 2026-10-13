"""Use the existing Qwen training task with the shared GEMM backend."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import yaml

from training.config import load_config
from training.runner import train

from .linear import install


def main():
    parser = argparse.ArgumentParser(description="Qwen 训练接入公共 GEMM")
    parser.add_argument("--config", default="configs/training/qwen3_06b.yaml")
    parser.add_argument("--gemm-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.gemm_config).read_text())
    if settings.get("backend") != "nvidia":
        raise ValueError("该训练入口使用已实现的 NVIDIA 后端。")
    config = load_config(args.config)
    config = replace(config, training=replace(config.training, output_dir=args.output_dir))
    # Changing native sources changes the resume identity as well as build cache.
    from backends.nvidia.build import build_key

    key, contract = build_key()
    implementation = hashlib.sha256()
    source = Path(__file__).resolve().parents[1]
    for folder in ("gemm", "algorithms", "backends"):
        for file in sorted((source / folder).rglob("*.py")):
            implementation.update(str(file.relative_to(source)).encode())
            implementation.update(file.read_bytes())
    print(json.dumps({"gemm": settings, "build": key, "contract": contract}, indent=2))
    train(config, resume=args.resume, offline=args.offline, model_transform=install,
          transform_id=f"sdc-gemm:v1:{key}:{implementation.hexdigest()[:16]}", transform_params=settings)


if __name__ == "__main__":
    main()
