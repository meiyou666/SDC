#!/bin/sh
set -eu

if [ -f /workspace/SDC/environment/baseline.json ]; then
    python /opt/sdc/environment/verify_environment.py \
        --baseline /workspace/SDC/environment/baseline.json \
        --lock /workspace/SDC/environment/dependencies/requirements.lock --quiet
else
    python /opt/sdc/environment/verify_environment.py --quiet
fi

exec "$@"
