#!/usr/bin/env bash
set -euo pipefail
case "${1:?}" in common|development|nvidia|ascend) ;; *) exit 1 ;; esac
sdc_extra=()
if [[ $1 != common ]]; then sdc_extra=(--extra "$1"); fi
UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=4 uv sync --locked --all-groups "${sdc_extra[@]}" --no-build --project /opt/sdc/project
touch /opt/venv/.lock
chmod a+rw /opt/venv/.lock
