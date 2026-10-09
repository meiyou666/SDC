#!/bin/sh
set -eu

project_root=/opt/sdc/project
if [ -f /workspace/SDC/pyproject.toml ]; then
    project_root=/workspace/SDC
fi
python /opt/sdc/project/environment/verify_environment.py --project "$project_root" --quiet

exec "$@"
