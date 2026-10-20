#!/usr/bin/env bash

# Source this helper; the manager calls it as root on Ubuntu.
SDC_NESTED_TOOLS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

sdc_nested_install_packages() (
    set -euo pipefail
    local policy=/usr/sbin/policy-rc.d backup='' work
    work=$(mktemp -d)
    trap 'rm -rf -- "$work"' EXIT
    # Preserve both regular files and symlinks, including dangling symlinks.
    if [[ -e $policy || -L $policy ]]; then
        cp -a -- "$policy" "$work/policy-rc.d"
        backup=$work/policy-rc.d
    fi
    trap 'rm -f "$policy"; if [[ -n $backup ]]; then mv -- "$backup" "$policy"; fi; rm -rf -- "$work"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    rm -f -- "$policy"
    printf '#!/bin/sh\nexit 101\n' > "$policy"
    chmod 0755 "$policy"

    local package status
    local -a missing=()
    for package in fuse2fs fuse-overlayfs uidmap supervisor jq curl ca-certificates util-linux e2fsprogs; do
        status=$(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true)
        [[ $status == installed ]] || missing+=("$package")
    done
    if ((${#missing[@]})); then
        apt-get update
        apt-get install -y "${missing[@]}"
    fi
    source "$SDC_NESTED_TOOLS_DIR/../install-docker.sh"
    sdc_install_docker root install-only

    if ! command -v nvidia-ctk >/dev/null || ! command -v nvidia-container-runtime >/dev/null; then
        apt-get update
        apt-get install -y gnupg
        curl -fsSL --retry 3 --connect-timeout 15 \
            https://nvidia.github.io/libnvidia-container/gpgkey -o "$work/nvidia.key"
        install -d -m 0755 /usr/share/keyrings
        gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg "$work/nvidia.key"
        curl -fsSL --retry 3 --connect-timeout 15 \
            https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            -o "$work/nvidia.list"
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            "$work/nvidia.list" > /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update
        apt-get install -y nvidia-container-toolkit
    fi
)

sdc_nested_install_buildkit() (
    set -euo pipefail
    local version=v0.24.0 arch asset digest work
    case $(uname -m) in
        # Official GitHub release asset digests, verified 2026-10-20.
        x86_64)
            arch=amd64
            digest=af8064eca16077b4d6937745988ba2d2dfa439540874cdcd918318315f3ba1d3
            ;;
        aarch64|arm64)
            arch=arm64
            digest=38dc4433d220bb43c198df2070e49d5dde5ed44ee31fb80d6b13722eec21d4ea
            ;;
        *) printf '%s\n' 'BuildKit 自动安装仅支持 x86_64 和 aarch64。' >&2; exit 1 ;;
    esac
    asset=buildkit-${version}.linux-${arch}.tar.gz
    work=$(mktemp -d)
    trap 'rm -rf -- "$work"' EXIT
    curl -fL --retry 3 --connect-timeout 15 \
        "https://github.com/moby/buildkit/releases/download/$version/$asset" -o "$work/$asset"
    printf '%s  %s\n' "$digest" "$work/$asset" | sha256sum --check
    mkdir "$work/extracted"
    tar -xzf "$work/$asset" -C "$work/extracted"
    test -x "$work/extracted/bin/buildkitd"
    install -d -m 0755 /opt/sdc-buildkit/bin
    install -m 0755 "$work/extracted/bin/"* /opt/sdc-buildkit/bin/
)

sdc_nested_install_tools() {
    if [[ $(id -u) != 0 ]]; then
        printf '%s\n' '嵌套 Docker 工具安装需要 root。' >&2
        return 1
    fi
    local ID
    source /etc/os-release
    if [[ $ID != ubuntu ]]; then
        printf '%s\n' '嵌套 Docker 工具安装仅支持 Ubuntu。' >&2
        return 1
    fi
    sdc_nested_install_packages
    if command -v buildkitd >/dev/null; then
        SDC_NESTED_BUILDKIT_BIN=$(command -v buildkitd)
    elif [[ -x /opt/sdc-buildkit/bin/buildkitd ]]; then
        SDC_NESTED_BUILDKIT_BIN=/opt/sdc-buildkit/bin/buildkitd
    else
        sdc_nested_install_buildkit
        SDC_NESTED_BUILDKIT_BIN=/opt/sdc-buildkit/bin/buildkitd
    fi
    export SDC_NESTED_BUILDKIT_BIN
}
