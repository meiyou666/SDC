#!/usr/bin/env bash
set -euo pipefail

. /etc/os-release
if [ "$ID" != ubuntu ]; then
    printf '%s\n' '请在 Ubuntu 中运行此脚本。' >&2
    exit 1
fi

sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings

sdc_docker_key=$(mktemp)
trap 'rm -f "$sdc_docker_key"' EXIT
curl -fsSL --retry 3 --connect-timeout 15 \
    https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu/gpg \
    -o "$sdc_docker_key"
sudo install -m 0644 "$sdc_docker_key" /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu
Suites: $VERSION_CODENAME
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$(id -un)"
printf '%s\n' 'Docker 已安装。退出并重新进入 Ubuntu 后，在仓库目录执行 ./dev.sh。'
