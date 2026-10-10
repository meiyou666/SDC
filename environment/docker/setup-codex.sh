#!/bin/sh
set -eu

if [ -n "${OPENAI_API_KEY:-}" ]; then
    printf '%s\n' "$OPENAI_API_KEY" | codex login --with-api-key >/dev/null
fi
