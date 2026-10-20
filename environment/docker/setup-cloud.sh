#!/usr/bin/env bash
set -euo pipefail

sdc_setup_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$sdc_setup_dir/install-docker.sh"

sdc_cloud_usage() {
    cat <<'EOF'
用法：bash environment/docker/setup-cloud.sh [dev|nvidia|ascend] [--soc 910b|a3|950] [--nested]

不带参数时选择环境。普通模式安装 Docker、构建镜像并生成云端入口；
--nested 安装并启动嵌套 Docker，项目镜像随后由 VS Code 构建。
示例：
  bash environment/docker/setup-cloud.sh dev
  bash environment/docker/setup-cloud.sh nvidia
  bash environment/docker/setup-cloud.sh nvidia --nested
  bash environment/docker/setup-cloud.sh ascend --soc 910b
EOF
}

sdc_install_nvidia_runtime() {
    local sdc_runtimes sdc_key sdc_repository
    sdc_runtimes=$(sdc_docker info --format '{{json .Runtimes}}')
    if [[ $sdc_runtimes == *'"nvidia"'* ]]; then
        return
    fi
    if [[ -n $(sdc_docker ps -q) ]]; then
        printf '%s\n' '配置 NVIDIA 运行时需要重启 Docker。当前有容器运行，请结束这些任务后重跑脚本。' >&2
        return 1
    fi
    if ! command -v nvidia-ctk >/dev/null; then
        sdc_root apt-get update
        sdc_root apt-get install -y ca-certificates curl gnupg
        sdc_key=$(mktemp)
        if ! curl -fsSL --retry 3 https://nvidia.github.io/libnvidia-container/gpgkey -o "$sdc_key"; then
            rm -f "$sdc_key"
            return 1
        fi
        sdc_root gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg "$sdc_key"
        rm -f "$sdc_key"
        sdc_repository=$(curl -fsSL --retry 3 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list)
        printf '%s\n' "$sdc_repository" |
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
            sdc_root tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sdc_root apt-get update
        sdc_root apt-get install -y nvidia-container-toolkit
    fi
    sdc_root nvidia-ctk runtime configure --runtime=docker
    if [[ -d /run/systemd/system ]]; then
        sdc_root systemctl restart docker
    else
        sdc_root service docker restart
    fi
    sdc_runtimes=$(sdc_docker info --format '{{json .Runtimes}}')
    [[ $sdc_runtimes == *'"nvidia"'* ]]
}

sdc_setup_cloud() {
    local sdc_profile='' sdc_soc='' sdc_choice sdc_platform sdc_target sdc_image sdc_nested=0
    local sdc_user=${SUDO_USER:-$(id -un)} sdc_uid sdc_gid sdc_device_gid=0
    local sdc_project sdc_device
    local -a sdc_build_args=()
    while (($#)); do
        case "$1" in
            dev|nvidia|ascend)
                [[ -z $sdc_profile ]] || { printf '%s\n' '每次选择一个环境。' >&2; return 2; }
                sdc_profile=$1; shift ;;
            --soc)
                [[ $# -ge 2 ]] || { printf '%s\n' '--soc 需要芯片型号。' >&2; return 2; }
                sdc_soc=$2; shift 2 ;;
            --nested) sdc_nested=1; shift ;;
            -h|--help) sdc_cloud_usage; return ;;
            *) printf '未知参数：%s\n' "$1" >&2; return 2 ;;
        esac
    done
    if [[ -z $sdc_profile ]]; then
        if [[ ! -t 0 ]]; then
            sdc_cloud_usage >&2
            return 2
        fi
        printf '%s\n' '选择服务器环境：1) 普通开发  2) NVIDIA GPU  3) 昇腾 NPU  4) NVIDIA 容器内 Docker'
        read -r -p '输入编号 [1]：' sdc_choice
        case ${sdc_choice:-1} in
            1) sdc_profile=dev ;;
            2) sdc_profile=nvidia ;;
            3) sdc_profile=ascend ;;
            4) sdc_profile=nvidia; sdc_nested=1 ;;
            *) printf '%s\n' '请输入 1、2、3 或 4。' >&2; return 2 ;;
        esac
    fi
    if [[ $sdc_profile == ascend && -z $sdc_soc && -t 0 ]]; then
        read -r -p '昇腾芯片：910b（A2）、a3 或 950：' sdc_soc
    fi
    if [[ $sdc_profile == ascend ]]; then
        case $sdc_soc in
            910b|a3|950) ;;
            *) printf '%s\n' '请用 --soc 指定 910b、a3 或 950。' >&2; return 2 ;;
        esac
    elif [[ -n $sdc_soc ]]; then
        printf '%s\n' '--soc 用于 ascend 环境。' >&2
        return 2
    fi
    if [[ $sdc_nested == 1 && $sdc_profile != nvidia ]]; then
        printf '%s\n' '--nested 仅用于 NVIDIA 平台容器。' >&2
        return 2
    fi
    case $(uname -m) in
        x86_64) sdc_platform=linux/amd64 ;;
        aarch64|arm64) sdc_platform=linux/arm64 ;;
        *) printf '%s\n' '无法识别服务器 CPU 架构。' >&2; return 1 ;;
    esac
    sdc_project=$(CDPATH= cd -- "$sdc_setup_dir/../.." && pwd)
    cd "$sdc_project"
    sdc_uid=$(id -u "$sdc_user")
    sdc_gid=$(id -g "$sdc_user")

    if [[ $sdc_nested == 1 ]]; then
        bash "$sdc_setup_dir/nested-docker.sh" setup
        if [[ ! -e .env ]]; then (umask 077; set -o noclobber; cat .env.example > .env); fi
        printf '%s\n' '在 VS Code 执行 Reopen in Container，选择“SDC NVIDIA（容器内 Docker）”。首次打开将构建项目镜像。'
        return
    fi

    case $sdc_profile in
        nvidia)
            if ! nvidia-smi >/dev/null 2>&1; then
                printf '%s\n' 'nvidia-smi 未正常运行，请先配置服务器的 NVIDIA 驱动。' >&2
                return 1
            fi ;;
        ascend)
            if ! npu-smi info >/dev/null 2>&1; then
                printf '%s\n' 'npu-smi info 未正常运行，请先配置服务器的昇腾驱动和固件。' >&2
                return 1
            fi
            for sdc_device in /dev/davinci0 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
                [[ -c $sdc_device ]] || { printf '缺少设备：%s\n' "$sdc_device" >&2; return 1; }
            done
            [[ -d /usr/local/Ascend/driver ]] || { printf '%s\n' '缺少 /usr/local/Ascend/driver。' >&2; return 1; }
            sdc_device_gid=$(stat -c '%g' /dev/davinci0) ;;
    esac

    sdc_install_docker "$sdc_user"
    if [[ $sdc_profile == nvidia ]]; then
        sdc_install_nvidia_runtime
    fi
    sdc_target=$sdc_profile
    [[ $sdc_profile != dev ]] || sdc_target=development
    sdc_image="sdc-${sdc_profile}:py311"
    if [[ $sdc_profile == ascend ]]; then
        sdc_build_args+=(--build-arg "ASCEND_SOC=$sdc_soc")
    fi
    sdc_docker build --network host --platform "$sdc_platform" --target "$sdc_target" \
        --tag "$sdc_image" --file environment/docker/Dockerfile "${sdc_build_args[@]}" .
    sdc_docker run --rm --network host --user "$sdc_uid:$sdc_gid" --entrypoint python \
        --volume "$sdc_project:/workspace/SDC" "$sdc_image" \
        /workspace/SDC/environment/docker/configure-cloud.py \
        --profile "$sdc_profile" --platform "$sdc_platform" \
        --uid "$sdc_uid" --soc "$sdc_soc" --device-gid "$sdc_device_gid"

    printf '\n%s\n' '镜像已构建。请在服务器仓库的 .env 中填写个人 API 密钥。'
    if [[ ${SDC_DOCKER_GROUP_CHANGED:-0} == 1 ]] || ! docker info >/dev/null 2>&1; then
        printf '%s\n' '用户组已更新：在 VS Code 执行 Remote-SSH: Kill VS Code Server on Host，再重新连接服务器。'
    fi
    printf '%s\n' '在 Remote-SSH 窗口打开服务器仓库，执行 Dev Containers: Reopen in Container，选择“SDC 云端”入口。'
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    sdc_setup_cloud "$@"
fi
