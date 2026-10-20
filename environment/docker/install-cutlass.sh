#!/usr/bin/env bash
set -euo pipefail
sdc_gemm_config=${SDC_GEMM_CONFIG:-/opt/sdc/gemm.json}
sdc_cutlass_path=${CUTLASS_PATH:-/opt/cutlass}
sdc_cutlass_commit=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["cutlass_commit"])' "$sdc_gemm_config")
if [[ -e $sdc_cutlass_path ]]; then
    test "$(git -C "$sdc_cutlass_path" rev-parse HEAD)" = "$sdc_cutlass_commit"
    test -z "$(git -C "$sdc_cutlass_path" status --porcelain)"
    exit 0
fi
mkdir -p "$(dirname "$sdc_cutlass_path")"
sdc_cutlass_stage=$(mktemp -d "${sdc_cutlass_path}.tmp.XXXXXXXX")
trap 'rm -rf -- "$sdc_cutlass_stage"' EXIT
git init "$sdc_cutlass_stage"
git -C "$sdc_cutlass_stage" remote add origin https://github.com/NVIDIA/cutlass.git
for attempt in 1 2 3; do
    if git -c http.version=HTTP/1.1 -C "$sdc_cutlass_stage" fetch --depth 1 origin "$sdc_cutlass_commit"; then
        break
    fi
    if [[ "$attempt" == 3 ]]; then exit 1; fi
done
git -C "$sdc_cutlass_stage" checkout --detach FETCH_HEAD
test "$(git -C "$sdc_cutlass_stage" rev-parse HEAD)" = "$sdc_cutlass_commit"
printf '\n/SDC_COMMIT\n' >> "$sdc_cutlass_stage/.git/info/exclude"
printf '%s\n' "$sdc_cutlass_commit" > "$sdc_cutlass_stage/SDC_COMMIT"
chmod -R a+rX "$sdc_cutlass_stage"
mv -T -- "$sdc_cutlass_stage" "$sdc_cutlass_path"
