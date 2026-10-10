#!/bin/sh
set -eu

project_root=/opt/sdc/project
if [ -f /workspace/SDC/pyproject.toml ]; then
    project_root=/workspace/SDC
fi
python /opt/sdc/project/environment/verify_environment.py --project "$project_root" --quiet
if [ ! -w "${HOME:-/}" ]; then
    sdc_codex_state="/tmp/sdc-codex-$(id -u)"
    mkdir -p "$sdc_codex_state"
    chmod 700 "$sdc_codex_state"
    export CODEX_HOME="$sdc_codex_state"
fi
python "$project_root/environment/codex/configure.py"

exec "$@"
