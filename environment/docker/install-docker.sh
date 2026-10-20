#!/usr/bin/env bash
set -euo pipefail

sdc_root() {
    if [[ $(id -u) == 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

sdc_docker() {
    if docker info >/dev/null 2>&1; then
        docker "$@"
    else
        sdc_root docker "$@"
    fi
}

sdc_install_docker() {
    local sdc_user=${1:-${SUDO_USER:-$(id -un)}}
    local sdc_mode=${2:-normal}
    if [[ $sdc_mode != normal && $sdc_mode != install-only ]]; then
        printf '未知 Docker 安装模式：%s\n' "$sdc_mode" >&2
        return 2
    fi
    local sdc_key sdc_codename sdc_arch sdc_package sdc_status
    local -a sdc_packages=() sdc_conflicts=()
    SDC_DOCKER_GROUP_CHANGED=0
    . /etc/os-release
    if [[ $ID != ubuntu ]]; then
        printf '%s\n' '此安装脚本使用 Ubuntu 的软件源。' >&2
        return 1
    fi
    id "$sdc_user" >/dev/null
    sdc_codename=$VERSION_CODENAME
    sdc_arch=$(dpkg --print-architecture)

    if ! command -v docker >/dev/null || ! command -v dockerd >/dev/null; then
        for sdc_package in docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc; do
            sdc_status=$(dpkg-query -W -f='${db:Status-Status}' "$sdc_package" 2>/dev/null || true)
            [[ $sdc_status != installed ]] || sdc_conflicts+=("$sdc_package")
        done
        if ((${#sdc_conflicts[@]})); then
            printf '检测到与 Docker CE 冲突的软件包：%s\n' "${sdc_conflicts[*]}" >&2
            printf '%s\n' '请先由管理员处理这些包承载的服务，再移除冲突包：' >&2
            if [[ $(id -u) != 0 ]]; then printf 'sudo ' >&2; fi
            printf 'apt-get remove' >&2
            printf ' %q' "${sdc_conflicts[@]}" >&2
            printf '\n%s\n' '处理完成后，重新运行初始化脚本。' >&2
            return 1
        fi
        sdc_packages+=(docker-ce docker-ce-cli containerd.io)
    fi
    if ! docker compose version >/dev/null 2>&1; then
        sdc_packages+=(docker-compose-plugin)
    fi
    if ! docker buildx version >/dev/null 2>&1; then
        sdc_packages+=(docker-buildx-plugin)
    fi
    if [[ $sdc_mode == install-only ]] && ! command -v rootlesskit >/dev/null; then
        sdc_packages+=(docker-ce-rootless-extras)
    fi
    if ((${#sdc_packages[@]})); then
        sdc_root apt-get update
        sdc_root apt-get install -y ca-certificates curl
        sdc_root install -m 0755 -d /etc/apt/keyrings
        sdc_key=$(mktemp)
        if ! curl -fsSL --retry 3 --connect-timeout 15 \
            https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu/gpg -o "$sdc_key"; then
            rm -f "$sdc_key"
            return 1
        fi
        sdc_root install -m 0644 "$sdc_key" /etc/apt/keyrings/docker.asc
        rm -f "$sdc_key"
        sdc_root tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu
Suites: $sdc_codename
Components: stable
Architectures: $sdc_arch
Signed-By: /etc/apt/keyrings/docker.asc
EOF
        sdc_root apt-get update
        sdc_root apt-get install -y "${sdc_packages[@]}"
    fi

    # The nested bootstrap inhibits package service starts around this call.
    # It configures and starts its own daemon only after all tools are ready.
    if [[ $sdc_mode == install-only ]]; then
        return
    fi
    if [[ $(id -un) == "$sdc_user" ]] && docker info >/dev/null 2>&1; then
        docker compose version
        docker buildx version
        printf '%s\n' 'Docker Engine、Compose 和 Buildx 已准备好。'
        return
    fi
    if ! sdc_root docker info >/dev/null 2>&1; then
        if [[ -d /run/systemd/system ]]; then
            sdc_root systemctl enable --now docker
        else
            sdc_root service docker start
        fi
    fi
    sdc_root docker info >/dev/null
    docker compose version
    docker buildx version

    if [[ $(id -u "$sdc_user") != 0 ]]; then
        if ! getent group docker >/dev/null; then
            sdc_root groupadd docker
        fi
        if [[ " $(id -nG "$sdc_user") " != *' docker '* ]]; then
            sdc_root usermod -aG docker "$sdc_user"
            SDC_DOCKER_GROUP_CHANGED=1
        fi
    fi
    printf '%s\n' 'Docker Engine、Compose 和 Buildx 已准备好。'
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    if [[ ${1:-} == --help || ${1:-} == -h ]]; then
        printf '%s\n' '用法：bash environment/docker/install-docker.sh [用户名]'
        exit 0
    fi
    if (($# > 1)); then
        printf '%s\n' '只接受一个可选用户名。' >&2
        exit 2
    fi
    sdc_install_docker "$@"
    if [[ $SDC_DOCKER_GROUP_CHANGED == 1 ]] || ! docker info >/dev/null 2>&1; then
        printf '%s\n' '用户组已更新。请重新登录后使用 Docker。'
    fi
fi
