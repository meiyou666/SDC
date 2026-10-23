#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source environment/activate.sh
python environment/verify_environment.py --project "$PWD" --smoke
uv run --locked --extra "${SDC_RUNTIME:-development}" pytest -q -p no:cacheprovider "$@"
uv run --locked --extra "${SDC_RUNTIME:-development}" ruff check --no-cache .
