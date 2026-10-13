#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f /.dockerenv && ! -f /run/.containerenv ]]; then
    echo '请在 NVIDIA 项目容器中运行。' >&2
    exit 1
fi
cd -- "$(dirname -- "$0")/.."
python environment/verify_environment.py --accelerator
python -c 'import torch; from backends.nvidia.runtime import require_device; require_device(torch.device("cuda"))'
python -m pytest -q tests/gemm
