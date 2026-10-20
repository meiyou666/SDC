#!/usr/bin/env bash
if [[ ${SDC_RUNTIME:-development} == ascend ]]; then
    sdc_cann_env=
    for sdc_candidate in "${SDC_CANN_ENV:-}" /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
        if [[ -f $sdc_candidate ]]; then sdc_cann_env=$sdc_candidate; break; fi
    done
    if [[ -z $sdc_cann_env ]]; then
        echo 'CANN set_env.sh is missing from the image.' >&2
        return 1
    fi
    sdc_restore_nounset=false
    case $- in *u*) sdc_restore_nounset=true; set +u ;; esac
    source "$sdc_cann_env"
    if [[ -f /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh ]]; then
        source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
    fi
    if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
        source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
    fi
    if $sdc_restore_nounset; then set -u; fi
    sdc_python_root=$(dirname "$(dirname "$(readlink -f "${UV_PYTHON:-/usr/local/bin/python}")")")
    export LD_LIBRARY_PATH="${sdc_python_root}/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="${SDC_PROJECT_ROOT:-/workspace/SDC}/src${PYTHONPATH:+:$PYTHONPATH}"
    unset sdc_cann_env sdc_candidate sdc_restore_nounset sdc_python_root
fi
