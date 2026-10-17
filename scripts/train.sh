#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f /.dockerenv && ! -f /run/.containerenv ]]; then
    echo '请在项目 Dev Container 中运行训练命令。' >&2
    exit 1
fi
cd -- "$(dirname -- "$0")/.."
source /opt/sdc/activate-accelerator.sh
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
exec uv run --locked --extra "${SDC_RUNTIME:-development}" python -m training "$@"
