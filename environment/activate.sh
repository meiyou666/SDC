#!/usr/bin/env bash
# Source this file from the repository or from the shared launchers.
sdc_project_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${SDC_DEPLOYMENT:-} == docker ]]; then
    source /opt/sdc/activate-accelerator.sh
elif [[ -f "$sdc_project_root/.sdc/config.sh" ]]; then
    source "$sdc_project_root/.sdc/config.sh"
    export VIRTUAL_ENV="$sdc_project_root/.venv"
    export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV" UV_PYTHON="$VIRTUAL_ENV/bin/python"
    export UV_PYTHON_DOWNLOADS=never PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    export PATH="$VIRTUAL_ENV/bin:$sdc_project_root/.sdc/bin:$PATH"
    export PYTHONPATH="$sdc_project_root/src" MPLBACKEND=Agg SDC_PROJECT_ROOT="$sdc_project_root"
    unset PYTHONHOME
    if [[ $SDC_RUNTIME == nvidia ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    elif [[ $SDC_RUNTIME == ascend ]]; then
        source "$sdc_project_root/environment/docker/activate-accelerator.sh"
    fi
elif [[ -f /opt/sdc/activate-accelerator.sh && -d /opt/venv ]]; then
    source /opt/sdc/activate-accelerator.sh
else
    echo '请先运行 bash environment/setup-platform.sh nvidia，或进入项目 Docker。' >&2
    return 1
fi
unset sdc_project_root
