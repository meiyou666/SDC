#!/bin/sh
set -eu

sdc_project=/workspace/SDC
if [ ! -e "$sdc_project/.env" ] && [ -f "$sdc_project/.env.example" ]; then
    (umask 077; cp -n "$sdc_project/.env.example" "$sdc_project/.env") ||
        printf '%s\n' 'Could not create .env; create it from .env.example when configuring Codex.' >&2
fi

sdc_codex_directory="${CODEX_HOME:-$HOME/.codex}"
if [ ! -f "$sdc_codex_directory/config.toml" ]; then
    (
        umask 077
        mkdir -p "$sdc_codex_directory"
        cp "$(dirname "$0")/codex-default.toml" "$sdc_codex_directory/config.toml"
    )
fi

python "$(dirname "$0")/login-codex.py"
