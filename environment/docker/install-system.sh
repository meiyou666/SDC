#!/usr/bin/env bash
set -euo pipefail
case "$(dpkg --print-architecture)" in
    amd64) sdc_apt_mirror=http://mirrors.tuna.tsinghua.edu.cn/ubuntu ;;
    arm64) sdc_apt_mirror=http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports ;;
    *) exit 1 ;;
esac
cat > /etc/apt/sources.list <<EOF
deb ${sdc_apt_mirror} jammy main restricted universe multiverse
deb ${sdc_apt_mirror} jammy-updates main restricted universe multiverse
deb ${sdc_apt_mirror} jammy-security main restricted universe multiverse
EOF
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl git xz-utils libstdc++6
sed -i 's|http://mirrors.tuna|https://mirrors.tuna|g' /etc/apt/sources.list
if [[ ${1:?} == full ]]; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssh-client rsync git-lfs build-essential cmake ninja-build pkg-config \
        gdb strace tmux htop procps psmisc lsof time \
        iproute2 iputils-ping netcat-openbsd dnsutils \
        wget aria2 jq ripgrep less file unzip zip zstd \
        libnuma1 libnuma-dev libgomp1 libopenblas-dev libblas3 liblapack3 \
        libaio1 libelf1 libtinfo5 pciutils zlib1g-dev libffi-dev libssl-dev
    git lfs install --system
fi
rm -rf /var/lib/apt/lists/*
