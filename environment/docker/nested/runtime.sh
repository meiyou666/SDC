#!/usr/bin/env bash
set -euo pipefail

config=${SDC_NESTED_CONFIG:-/etc/sdc-docker/settings.conf}
if [[ -f "$config" ]]; then source "$config"; fi
: "${SDC_NESTED_MOUNT:=/srv/sdc-docker-storage}"
: "${SDC_NESTED_RUN:=/run/sdc-docker}"
: "${SDC_NESTED_BUILDKIT_ROOT:=/srv/sdc-buildkit-cache}"
: "${SDC_NESTED_BUILDKIT_SIZE:=50g}"
: "${SDC_NESTED_ETC:=/etc/sdc-docker}"
: "${SDC_NESTED_INSTALL:=/usr/local/lib/sdc-docker}"
: "${SDC_NESTED_LOG:=/var/log}"
: "${SDC_NESTED_BUILDKIT_BIN:=/opt/sdc-buildkit/bin/buildkitd}"
: "${SDC_NESTED_DAEMON_CONFIG:=/etc/docker/daemon.json}"

die() { echo "$*" >&2; exit 1; }
cleanup() {
    trap - EXIT INT TERM
    local pid attempt
    # Only signal still-running children of this shell; never trust PID files.
    for pid in $(jobs -pr); do kill -TERM "$pid" 2>/dev/null || true; done
    for ((attempt=0; attempt<25; attempt++)); do
        [[ -z $(jobs -pr) ]] && break
        sleep 1
    done
    for pid in $(jobs -pr); do kill -KILL "$pid" 2>/dev/null || true; done
    wait || true
}

case "${1:-}" in
    storage)
        [[ ${SDC_NESTED_STORAGE_FILE:-} == /* ]] || die 'SDC_NESTED_STORAGE_FILE must be absolute.'
        [[ -f "$SDC_NESTED_STORAGE_FILE" ]] || die 'Storage image is missing.'
        mkdir -p "$SDC_NESTED_MOUNT"
        exec fuse2fs -f -o fakeroot,allow_other "$SDC_NESTED_STORAGE_FILE" "$SDC_NESTED_MOUNT"
        ;;
    daemon)
        export XDG_RUNTIME_DIR="$SDC_NESTED_RUN"
        mkdir -p "$SDC_NESTED_RUN"
        chmod 0700 "$SDC_NESTED_RUN"
        for ((attempt=0; attempt<30; attempt++)); do
            mountpoint -q "$SDC_NESTED_MOUNT" && break
            sleep 1
        done
        mountpoint -q "$SDC_NESTED_MOUNT" || die 'Storage did not mount within 30 seconds.'
        [[ ${SDC_NESTED_STORAGE_FILE:-} == /* && $(findmnt -n -o SOURCE --target "$SDC_NESTED_MOUNT") == "$SDC_NESTED_STORAGE_FILE" ]] || die 'Storage mount belongs to a different filesystem.'
        mkdir -p "$SDC_NESTED_RUN" "$SDC_NESTED_BUILDKIT_ROOT" "$SDC_NESTED_LOG"
        if mountpoint -q "$SDC_NESTED_BUILDKIT_ROOT"; then
            [[ $(findmnt -n -o FSTYPE --target "$SDC_NESTED_BUILDKIT_ROOT") == tmpfs ]] || die 'BuildKit cache is not tmpfs.'
            mount -o "remount,size=$SDC_NESTED_BUILDKIT_SIZE" "$SDC_NESTED_BUILDKIT_ROOT"
        else
            mount -t tmpfs -o "size=$SDC_NESTED_BUILDKIT_SIZE,mode=0755" tmpfs "$SDC_NESTED_BUILDKIT_ROOT"
        fi
        if [[ -e /run/docker.sock || -L /run/docker.sock ]]; then
            [[ -L /run/docker.sock && $(readlink /run/docker.sock) == "$SDC_NESTED_RUN/docker.sock" ]] || die '/run/docker.sock already belongs to another daemon.'
        else
            ln -s "$SDC_NESTED_RUN/docker.sock" /run/docker.sock
        fi
        "$SDC_NESTED_INSTALL/nested/cdi.sh"
        exec rootlesskit --subid-source=static --state-dir="$SDC_NESTED_RUN/rootlesskit" \
            --net=host --port-driver=none --copy-up=/etc --copy-up=/run \
            "$SDC_NESTED_INSTALL/nested/runtime.sh" child
        ;;
    child)
        export XDG_RUNTIME_DIR="$SDC_NESTED_RUN"
        mkdir -p "$SDC_NESTED_LOG"
        trap cleanup EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        "$SDC_NESTED_BUILDKIT_BIN" --config "$SDC_NESTED_ETC/buildkit.toml" \
            --addr "unix://$SDC_NESTED_RUN/buildkit.sock" --root "$SDC_NESTED_BUILDKIT_ROOT" \
            --oci-worker=true --containerd-worker=false --oci-worker-rootless \
            --oci-worker-no-process-sandbox --oci-worker-snapshotter=fuse-overlayfs \
            --oci-worker-net=host --allow-insecure-entitlement network.host \
            >> "$SDC_NESTED_LOG/sdc-buildkit.log" 2>&1 &
        dockerd --config-file "$SDC_NESTED_DAEMON_CONFIG" &
        # Any helper exit invalidates the service, even when its exit status is zero.
        status=0
        wait -n || status=$?
        ((status != 0)) || status=1
        exit "$status"
        ;;
    *) die 'Usage: runtime.sh {storage|daemon|child}' ;;
esac
