"""Run the existing Qwen task with the official CUTLASS Linear adapter."""

import argparse
from dataclasses import replace

import torch

from training.config import load_config
from training.runner import resolve_runtime_config, train

from .backend import build_key, require_device
from .linear import install


def main():
    parser = argparse.ArgumentParser(description="CUTLASS Qwen 基础训练")
    parser.add_argument("--config", default="configs/training/qwen3_06b.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = resolve_runtime_config(load_config(args.config))
    if config.training.device != "cuda" or config.training.precision not in {"bf16", "fp32"}:
        parser.error("本入口使用 NVIDIA GPU，训练精度为 BF16 或 FP32。")
    require_device(torch.device("cuda"))
    config = replace(config, training=replace(config.training, output_dir=args.output_dir))
    train(config, resume=args.resume, offline=args.offline, model_transform=install,
          transform_id=f"cutlass-baseline:{build_key()}")


if __name__ == "__main__":
    main()
