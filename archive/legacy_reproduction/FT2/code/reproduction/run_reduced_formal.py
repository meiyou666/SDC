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

from ft2_formal.artifacts import load_artifact
from ft2_formal.audit import audit_reduced_formal
from ft2_formal.reduced_campaign import (
    FORMAL_OUTPUT_ROOT,
    ReducedFormalCampaign,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen FT2 RTX4090 reduced formal campaign"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FORMAL_OUTPUT_ROOT,
        help="Reduced-formal artifact directory",
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--prepare-only",
        action="store_true",
        help="Lock prompts, dependencies, bounds references, and manifests",
    )
    stage.add_argument(
        "--audit-only",
        action="store_true",
        help="Verify an already completed campaign without loading a model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".runner.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                f"Another FT2 campaign process holds {lock_path}"
            ) from exc
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        os.fsync(lock.fileno())

        if args.audit_only:
            campaign = load_artifact(output_root / "campaign.json")
            result = audit_reduced_formal(
                output_root,
                campaign_fingerprint=campaign["campaign_fingerprint"],
                write_outputs=True,
            )
        else:
            campaign_runner = ReducedFormalCampaign(output_root)
            result = (
                campaign_runner.prepare()
                if args.prepare_only
                else campaign_runner.run()
            )

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

