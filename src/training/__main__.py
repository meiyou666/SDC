"""Command line entry point: ``python -m training prepare|train --config PATH``.

``prepare`` downloads and freezes the data, tokenizer and model; ``train`` runs
the job. Heavy imports stay inside the subcommands so ``--help`` is instant and
a configuration error is reported before torch is loaded.
"""

import argparse
import json

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一语言模型训练入口")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="下载并固定文本数据、分词器和模型")
    prepare.add_argument("--config", required=True, help="训练配置的 YAML 路径")
    prepare.add_argument("--data-only", action="store_true", help="只准备数据和分词器，不下载模型权重")
    run = commands.add_parser("train", help="全参数 next-token 训练与验证")
    run.add_argument("--config", required=True, help="训练配置的 YAML 路径")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "prepare":
            from .data import download_model, prepare_data

            manifest = prepare_data(config)
            if not args.data_only:
                download_model(config)
            print(json.dumps(
                {"prepared_dir": config.data.prepared_dir, "statistics": manifest["statistics"]},
                indent=2,
            ))
        else:
            print(f"[训练] 配置已加载：{args.config}", flush=True)
            from .runner import train

            train(config)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"训练任务：{exc}\n")


if __name__ == "__main__":
    main()
