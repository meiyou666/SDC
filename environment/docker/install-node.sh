#!/usr/bin/env bash
set -euo pipefail
case "$(dpkg --print-architecture)" in
    amd64) sdc_node_arch=x64; sdc_node_hash=fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6 ;;
    arm64) sdc_node_arch=arm64; sdc_node_hash=6ad1325edbdb5649c379b75a237147a666c95d4f9ae8d340fef2d1575d289ad2 ;;
    *) exit 1 ;;
esac
curl -fsSL --retry 3 "https://registry.npmmirror.com/-/binary/node/v24.21.0/node-v24.21.0-linux-${sdc_node_arch}.tar.xz" -o /tmp/sdc-node.tar.xz
printf '%s  /tmp/sdc-node.tar.xz\n' "$sdc_node_hash" | sha256sum -c -
tar -xJf /tmp/sdc-node.tar.xz -C /usr/local --strip-components=1
rm /tmp/sdc-node.tar.xz
