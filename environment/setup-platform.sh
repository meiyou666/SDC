#!/usr/bin/env bash
set -euo pipefail
sdc_project=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == --help ]]; then
    echo '用法：bash environment/setup-platform.sh [nvidia|ascend|development]'
    echo '复用平台 CUDA/CANN，在仓库 .venv 中安装锁定依赖。'
    exit 0
fi
sdc_runtime=${1:-nvidia}
case "$sdc_runtime" in nvidia|ascend|development) ;; *) echo '请选择 nvidia、ascend 或 development。' >&2; exit 2 ;; esac
[[ $# -le 1 ]] || exit 2
cd "$sdc_project"
for sdc_tool in python3 git gcc g++ make ninja; do
    command -v "$sdc_tool" >/dev/null || { echo "缺少 $sdc_tool，请在平台环境安装对应系统工具。" >&2; exit 1; }
done
python3 environment/configure_platform.py check --profile "$sdc_runtime"
# Bootstrap uses only the platform Python standard library to read shared versions.
read -r sdc_python sdc_uv < <(python3 -c 'import json; b=json.load(open("environment/baseline.json")); print(b["python"], b["uv"])')
mkdir -p .sdc/bin
if [[ ! -x .sdc/bin/uv || $(.sdc/bin/uv --version | awk '{print $2}') != "$sdc_uv" ]]; then
    python3 -m pip --version >/dev/null 2>&1 || {
        echo '缺少引导工具 pip，请执行 apt-get install -y python3-pip 后重试。' >&2
        exit 1
    }
    sdc_download=$(mktemp -d "$sdc_project/.sdc/uv-download.XXXXXXXX")
    trap 'rm -rf -- "$sdc_download"' EXIT
    python3 -m pip --isolated install --disable-pip-version-check --no-deps --only-binary=:all: \
        --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
        --timeout 30 --retries 3 --target "$sdc_download" "uv==$sdc_uv"
    [[ $("$sdc_download/bin/uv" --version | awk '{print $2}') == "$sdc_uv" ]]
    install -m 0755 "$sdc_download/bin/uv" .sdc/bin/uv
    install -m 0755 "$sdc_download/bin/uvx" .sdc/bin/uvx
fi
export PATH="$sdc_project/.sdc/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$sdc_project/.sdc/python"
export UV_PROJECT_ENVIRONMENT="$sdc_project/.venv" UV_PYTHON_DOWNLOADS=automatic
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH UV_PYTHON
uv python install "$sdc_python"
export UV_PYTHON="$sdc_python"
UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=4 uv sync --locked --all-groups --extra "$sdc_runtime" --no-build
export PATH="$sdc_project/.venv/bin:$PATH"
if [[ $sdc_runtime == nvidia ]]; then
    SDC_GEMM_CONFIG="$sdc_project/environment/gemm.json" CUTLASS_PATH="$sdc_project/.sdc/cutlass" \
        bash environment/docker/install-cutlass.sh
fi
python environment/configure_platform.py configure --profile "$sdc_runtime"
export SDC_DEPLOYMENT=platform
source environment/activate.sh
python environment/verify_environment.py --smoke
if [[ $sdc_runtime != development ]]; then python environment/verify_environment.py --accelerator; fi
python environment/configure_platform.py workspace
if [[ ! -e .env ]]; then (umask 077; set -o noclobber; cat .env.example > .env); fi
printf '%s\n' '环境就绪。执行 source environment/activate.sh；VS Code 打开 .sdc/SDC-platform.code-workspace。'
