#!/usr/bin/env bash
set -euo pipefail
case "${1:?}" in common|development|nvidia|ascend) ;; *) exit 1 ;; esac
sdc_extra=()
if [[ $1 != common ]]; then sdc_extra=(--extra "$1"); fi
sdc_build=(--no-build)
if [[ $1 == nvidia ]]; then
  export MAX_JOBS=${MAX_JOBS:-2} NVTE_BUILD_THREADS_PER_JOB=${NVTE_BUILD_THREADS_PER_JOB:-1}
  # The pinned official source builds the core and PyTorch binding together.
  UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=4 uv sync --locked --all-groups "${sdc_extra[@]}" --no-build --no-install-package transformer-engine --project /opt/sdc/project
  sdc_cuda_includes=$(python -c 'from importlib.metadata import distribution; print(":".join(str(distribution(package).locate_file(path)) for package, path in [("nvidia-cudnn-cu12", "nvidia/cudnn/include"), ("nvidia-nccl-cu12", "nvidia/nccl/include")]))')
  export CPATH="$sdc_cuda_includes${CPATH:+:$CPATH}"
  export CUDNN_PATH=$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-cudnn-cu12").locate_file("nvidia/cudnn"))')
  export NCCL_HOME=$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-nccl-cu12").locate_file("nvidia/nccl"))')
  if [[ -n ${UV_CACHE_DIR:-} && -z ${NVTE_CMAKE_BUILD_DIR:-} ]]; then
    sdc_build_key=$(python -c 'import hashlib, platform; from pathlib import Path; root = Path("/opt/sdc/project"); data = b"".join((root / name).read_bytes() for name in ("pyproject.toml", "uv.lock", "environment/baseline.json")); print(hashlib.sha256(data + platform.machine().encode()).hexdigest())')
    export NVTE_CMAKE_BUILD_DIR="$UV_CACHE_DIR/te-cmake/$sdc_build_key"
  fi
  sdc_build=(--no-build-isolation-package transformer-engine)
  while IFS= read -r sdc_package; do
    sdc_build+=(--no-build-package "$sdc_package")
  done < <(python -c 'import tomllib; from pathlib import Path; lock = tomllib.loads(Path("/opt/sdc/project/uv.lock").read_text()); print("\n".join(sorted({p["name"] for p in lock["package"]} - {"sdc", "transformer-engine"})))')
fi
UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=4 uv sync --locked --all-groups "${sdc_extra[@]}" "${sdc_build[@]}" --project /opt/sdc/project
touch /opt/venv/.lock
chmod a+rw /opt/venv/.lock
