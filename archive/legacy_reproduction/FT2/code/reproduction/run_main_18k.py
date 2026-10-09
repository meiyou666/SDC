#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reproduction" / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ft2_formal.main18k import (
    DEFAULT_MAIN18K_ROOT,
    Main18kCampaign,
    audit_main18k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen FT2 18,000-fault main campaign"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_MAIN18K_ROOT,
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".runner.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                f"Another main18k process holds {lock_path}"
            ) from exc
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        os.fsync(lock.fileno())

        campaign = Main18kCampaign(root)
        if args.audit_only:
            result = audit_main18k(campaign)
        elif args.prepare_only:
            result = campaign.prepare()
        else:
            result = campaign.run()

    print(
        json.dumps(
            {
                "status": result["status"],
                "campaign_fingerprint":
                    result["campaign_fingerprint"],
                "counts": result["counts"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
