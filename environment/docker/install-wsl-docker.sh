#!/usr/bin/env bash
set -euo pipefail

sdc_script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$sdc_script_dir/install-docker.sh" "$@"
