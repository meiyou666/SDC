#!/bin/sh
set -eu

sdc_codex_directory="${CODEX_HOME:-$HOME/.codex}"
if [ ! -f "$sdc_codex_directory/config.toml" ]; then
    (
        umask 077
        mkdir -p "$sdc_codex_directory"
        cp "$(dirname "$0")/codex-default.toml" "$sdc_codex_directory/config.toml"
    )
fi

python "$(dirname "$0")/login-codex.py"
