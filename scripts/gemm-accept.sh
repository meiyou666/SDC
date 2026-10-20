#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source environment/activate.sh
python environment/verify_environment.py --accelerator
python -c 'import torch; from cutlass_gemm.backend import require_device; require_device(torch.device("cuda"))'
python -m pytest -q tests/cutlass_gemm
