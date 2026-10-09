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

from ft2_formal.audit import audit_pilot
from ft2_formal.campaign import DEFAULT_OUTPUT_ROOT, PilotCampaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen FT2 RTX4090 pilot campaign"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Pilot artifact directory",
    )
    parser.add_argument(
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
            campaign_path = output_root / "campaign.json"
            if not campaign_path.exists():
                raise SystemExit(f"Missing campaign lock: {campaign_path}")
            from ft2_formal.artifacts import load_artifact

            campaign = load_artifact(campaign_path)
            result = audit_pilot(
                output_root,
                campaign_fingerprint=campaign["campaign_fingerprint"],
                write_outputs=True,
            )
        else:
            campaign = PilotCampaign(output_root)
            result = campaign.run()

    print(
        json.dumps(
            {
                "status": result["status"],
                "campaign_fingerprint": result["campaign_fingerprint"],
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
