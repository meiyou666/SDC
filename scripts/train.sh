#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
source environment/activate.sh
python environment/verify_environment.py --project "$PWD" --quiet
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
exec uv run --locked --extra "${SDC_RUNTIME:-development}" python -m training "$@"
