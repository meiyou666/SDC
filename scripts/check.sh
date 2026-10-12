#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f /.dockerenv && ! -f /run/.containerenv ]]; then
    echo '请在项目容器中运行检查。' >&2
    exit 1
fi
cd -- "$(dirname -- "$0")/.."
source /opt/sdc/activate-accelerator.sh
python /opt/sdc/project/environment/verify_environment.py --project "$PWD" --smoke
uv run --locked --extra "${SDC_RUNTIME:-development}" pytest -q -p no:cacheprovider "$@"
uv run --locked --extra "${SDC_RUNTIME:-development}" ruff check --no-cache .
