"""python -m training prepare|train --config PATH"""

import argparse
import json

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="统一语言模型训练入口")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="下载并固定文本数据、分词器和模型")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--data-only", action="store_true", help="只准备数据和分词器")
    prepare.add_argument("--verify-data", action="store_true", help="重新计算全部数据文件的 SHA256")
    run = commands.add_parser("train", help="全参数 next-token 训练与验证")
    run.add_argument("--config", required=True)
    run.add_argument("--resume", help="latest 或 checkpoint 目录")
    run.add_argument("--offline", action="store_true", help="模型和分词器只从本地缓存读取")
    run.add_argument("--verify-data", action="store_true", help="重新计算全部数据文件的 SHA256")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "prepare":
            from .data import download_model, prepare_data

            manifest = prepare_data(config, verify_data=args.verify_data)
            if not args.data_only:
                download_model(config)
            print(json.dumps({"prepared_dir": config.data.prepared_dir, "statistics": manifest["statistics"]}, indent=2))
        else:
            from .runner import train

            train(config, resume=args.resume, offline=args.offline, verify_data=args.verify_data)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Training: {exc}\n")


if __name__ == "__main__":
    main()
