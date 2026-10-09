#!/bin/sh
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_root"
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker was not found. Install Docker Engine and Compose v2 first." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "Docker is unavailable. Start it and check your access permissions." >&2
    exit 1
fi
docker compose version >/dev/null
if [ -t 0 ] && [ -t 1 ]; then
    exec docker compose run --build --rm --user "$(id -u):$(id -g)" dev "$@"
fi
exec docker compose run --build --rm -T --user "$(id -u):$(id -g)" dev "$@"
