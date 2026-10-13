#!/usr/bin/env bash
set -euo pipefail
sdc_cutlass_commit=$(python -c 'import json; print(json.load(open("/opt/sdc/gemm.json"))["cutlass_commit"])')
git init /opt/cutlass
git -C /opt/cutlass remote add origin https://github.com/NVIDIA/cutlass.git
for attempt in 1 2 3; do
    if git -c http.version=HTTP/1.1 -C /opt/cutlass fetch --depth 1 origin "$sdc_cutlass_commit"; then
        break
    fi
    if [[ "$attempt" == 3 ]]; then exit 1; fi
done
git -C /opt/cutlass checkout --detach FETCH_HEAD
test "$(git -C /opt/cutlass rev-parse HEAD)" = "$sdc_cutlass_commit"
printf '%s\n' "$sdc_cutlass_commit" > /opt/cutlass/SDC_COMMIT
chmod -R a+rX /opt/cutlass
