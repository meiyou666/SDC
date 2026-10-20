#!/usr/bin/env bash
set -euo pipefail

sdc_nested_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

sdc_nested_usage() {
    cat <<'EOF'
用法：bash environment/docker/nested-docker.sh COMMAND [选项]
  setup   安装工具、配置服务并启动；已有 ext4 存储文件会复用
          --storage-file PATH  存储文件，默认在仓库的上一级目录
          --storage-gib N      新存储文件大小，默认 64 GiB
  start | stop | restart | status
  logs [docker|buildkit|storage]

在外层 SSH 环境以 root 执行。setup 不修改或扩容已有存储文件。
EOF
}

sdc_nested_defaults() {
    : "${SDC_NESTED_CONFIG:=/etc/sdc-docker/settings.conf}"
    if [[ -f $SDC_NESTED_CONFIG ]]; then source "$SDC_NESTED_CONFIG"; fi
    : "${SDC_NESTED_ETC:=/etc/sdc-docker}"
    : "${SDC_NESTED_MOUNT:=/srv/sdc-docker-storage}"
    : "${SDC_NESTED_RUN:=/run/sdc-docker}"
    : "${SDC_NESTED_INSTALL:=/usr/local/lib/sdc-docker}"
    : "${SDC_NESTED_LOG:=/var/log}"
    : "${SDC_NESTED_CDI_DIR:=/etc/cdi}"
    : "${SDC_NESTED_DAEMON_CONFIG:=/etc/docker/daemon.json}"
    : "${SDC_NESTED_BUILDKIT_ROOT:=/srv/sdc-buildkit-cache}"
    : "${SDC_NESTED_BUILDKIT_SIZE:=50g}"
    : "${SDC_NESTED_STORAGE_GIB:=64}"
    : "${SDC_NESTED_STORAGE_FILE:=$(realpath -m "$sdc_nested_dir/../../../sdc-docker-storage.ext4")}"
}

sdc_nested_docker() { docker --host "unix://$SDC_NESTED_RUN/docker.sock" "$@"; }
sdc_nested_ctl() { "$SDC_NESTED_SUPERVISORCTL" -c "$SDC_NESTED_SUPERVISOR_CONFIG" "$@"; }
sdc_nested_service_state() {
    local output name state
    output=$(sdc_nested_ctl status "$1" 2>/dev/null || true)
    read -r name state _ <<< "$output"
    printf '%s\n' "${state:-UNKNOWN}"
}

sdc_nested_storage() {
    local staging
    [[ $SDC_NESTED_STORAGE_FILE == /* && ! -L $SDC_NESTED_STORAGE_FILE ]] || {
        printf '%s\n' '存储文件必须是绝对路径，且不能是符号链接。' >&2; return 1;
    }
    if [[ -e $SDC_NESTED_STORAGE_FILE ]]; then
        [[ -f $SDC_NESTED_STORAGE_FILE && $(blkid -p -s TYPE -o value "$SDC_NESTED_STORAGE_FILE") == ext4 ]] || {
            printf '%s\n' '已有文件不是 ext4 存储镜像，未修改该文件。' >&2; return 1;
        }
        return
    fi
    mkdir -p "$(dirname "$SDC_NESTED_STORAGE_FILE")"
    staging=$(mktemp "$(dirname "$SDC_NESTED_STORAGE_FILE")/.sdc-storage.XXXXXXXX")
    if ! (truncate -s "${SDC_NESTED_STORAGE_GIB}G" "$staging" && mkfs.ext4 -q -F "$staging"); then
        rm -f -- "$staging"
        return 1
    fi
    chmod 0600 "$staging"
    mv -n -- "$staging" "$SDC_NESTED_STORAGE_FILE"
    if [[ -e $staging ]]; then
        rm -f -- "$staging"
        printf '%s\n' '存储文件已被其他进程创建，请重新检查。' >&2
        return 1
    fi
}

sdc_nested_select_supervisor() {
    if [[ -f /base/mambaforge/etc/supervisord.conf && -x /base/mambaforge/bin/supervisorctl ]]; then
        SDC_NESTED_SUPERVISOR_CONFIG=/base/mambaforge/etc/supervisord.conf
        SDC_NESTED_SUPERVISORCTL=/base/mambaforge/bin/supervisorctl
        SDC_NESTED_SUPERVISORD=/base/mambaforge/bin/supervisord
        SDC_NESTED_PROGRAM_DIR=/root/supervisord
        SDC_NESTED_PRIVATE_SUPERVISOR=0
    else
        SDC_NESTED_SUPERVISOR_CONFIG="$SDC_NESTED_ETC/supervisord.conf"
        SDC_NESTED_SUPERVISORCTL=$(command -v supervisorctl)
        SDC_NESTED_SUPERVISORD=$(command -v supervisord)
        SDC_NESTED_PROGRAM_DIR="$SDC_NESTED_ETC/programs"
        SDC_NESTED_PRIVATE_SUPERVISOR=1
    fi
}

sdc_nested_ensure_supervisor() {
    mkdir -p "$SDC_NESTED_RUN" "$SDC_NESTED_LOG"
    chmod 0700 "$SDC_NESTED_RUN"
    if sdc_nested_ctl pid >/dev/null 2>&1; then return; fi
    if [[ $SDC_NESTED_PRIVATE_SUPERVISOR == 1 ]]; then
        "$SDC_NESTED_SUPERVISORD" -c "$SDC_NESTED_SUPERVISOR_CONFIG"
    else
        printf '%s\n' '平台 Supervisor 未运行，请先恢复平台服务。' >&2
        return 1
    fi
}

sdc_nested_program() {
    local name=$1 action=$2 priority=$3
    cat <<EOF
[program:$name]
command=$SDC_NESTED_INSTALL/nested/runtime.sh $action
environment=SDC_NESTED_CONFIG="$SDC_NESTED_CONFIG",PATH="/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin"
priority=$priority
autostart=true
autorestart=true
startsecs=3
startretries=3
stopsignal=TERM
stopwaitsecs=40
stopasgroup=true
killasgroup=true
stdout_logfile=$SDC_NESTED_LOG/$name.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=3
redirect_stderr=true

EOF
}

sdc_nested_write_config() {
    local staging key
    mkdir -p "$SDC_NESTED_ETC" "$SDC_NESTED_PROGRAM_DIR" "$SDC_NESTED_LOG" "$(dirname "$SDC_NESTED_CONFIG")"
    staging=$(mktemp "$SDC_NESTED_ETC/.settings.XXXXXXXX")
    for key in SDC_NESTED_ETC SDC_NESTED_MOUNT SDC_NESTED_RUN SDC_NESTED_INSTALL SDC_NESTED_LOG \
               SDC_NESTED_CDI_DIR SDC_NESTED_DAEMON_CONFIG SDC_NESTED_BUILDKIT_ROOT \
               SDC_NESTED_BUILDKIT_SIZE SDC_NESTED_STORAGE_FILE SDC_NESTED_STORAGE_GIB \
               SDC_NESTED_BUILDKIT_BIN SDC_NESTED_SUPERVISOR_CONFIG SDC_NESTED_SUPERVISORCTL \
               SDC_NESTED_SUPERVISORD SDC_NESTED_PROGRAM_DIR SDC_NESTED_PRIVATE_SUPERVISOR; do
        printf '%s=%q\n' "$key" "${!key}" >> "$staging"
    done
    chmod 0600 "$staging"
    mv -f -- "$staging" "$SDC_NESTED_CONFIG"
    mkdir -p "$(dirname "$SDC_NESTED_DAEMON_CONFIG")"
    staging=$(mktemp "$(dirname "$SDC_NESTED_DAEMON_CONFIG")/.sdc-daemon.XXXXXXXX")
    if [[ -f $SDC_NESTED_DAEMON_CONFIG ]]; then
        cp -p "$SDC_NESTED_DAEMON_CONFIG" "$SDC_NESTED_ETC/daemon.before-setup.json"
    else
        printf '{}\n' > "$SDC_NESTED_DAEMON_CONFIG"
    fi
    jq --arg root "$SDC_NESTED_MOUNT/docker" --arg run "$SDC_NESTED_RUN" '
      . + {"data-root":$root, "exec-root":($run+"/exec"),
           "hosts":[("unix://"+$run+"/docker.sock")], "pidfile":($run+"/dockerd.pid"),
           "rootless":true, "storage-driver":"fuse-overlayfs", "default-runtime":"nvidia",
           "bridge":"none", "iptables":false, "ip6tables":false,
           "ip-forward":false, "ip-masq":false,
           "exec-opts":["native.cgroupdriver=cgroupfs"], "log-driver":"local",
           "log-opts":{"max-file":"3","max-size":"10m"}}
      | .runtimes.nvidia = {"path":"nvidia-container-runtime","args":[]}
    ' "$SDC_NESTED_DAEMON_CONFIG" > "$staging"
    chmod 0644 "$staging"
    mv -f -- "$staging" "$SDC_NESTED_DAEMON_CONFIG"
    nvidia-ctk config --set nvidia-container-runtime.mode=cdi --set nvidia-container-cli.no-cgroups=true --in-place
    if [[ ! -f $SDC_NESTED_ETC/buildkit.toml ]]; then
        printf '[registry."docker.io"]\n  mirrors = ["docker.m.daocloud.io"]\n' > "$SDC_NESTED_ETC/buildkit.toml"
    fi
    sdc_nested_program sdc-docker-storage storage 10 > "$SDC_NESTED_PROGRAM_DIR/sdc-docker-storage.conf"
    sdc_nested_program sdc-docker daemon 20 > "$SDC_NESTED_PROGRAM_DIR/sdc-docker.conf"
    if [[ $SDC_NESTED_PRIVATE_SUPERVISOR == 1 ]]; then
        cat > "$SDC_NESTED_SUPERVISOR_CONFIG" <<EOF
[unix_http_server]
file=$SDC_NESTED_RUN/supervisor.sock
chmod=0700
[supervisord]
logfile=$SDC_NESTED_LOG/sdc-supervisor.log
pidfile=$SDC_NESTED_RUN/supervisor.pid
childlogdir=$SDC_NESTED_LOG
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://$SDC_NESTED_RUN/supervisor.sock
[include]
files=$SDC_NESTED_PROGRAM_DIR/*.conf
EOF
    fi
}

sdc_nested_start() {
    local name attempt ready
    sdc_nested_ensure_supervisor
    for name in sdc-docker-storage sdc-docker; do
        case $(sdc_nested_service_state "$name") in
            RUNNING|STARTING|BACKOFF) ;;
            STOPPED|EXITED|FATAL) sdc_nested_ctl start "$name" ;;
            *) printf '无法读取服务状态：%s\n' "$name" >&2; return 1 ;;
        esac
    done
    for ((attempt=0; attempt<30; attempt++)); do
        ready=1
        for name in sdc-docker-storage sdc-docker; do
            [[ $(sdc_nested_service_state "$name") == RUNNING ]] || ready=0
        done
        if [[ $ready == 1 && -S $SDC_NESTED_RUN/buildkit.sock ]] && sdc_nested_docker info >/dev/null 2>&1; then break; fi
        sleep 1
    done
    [[ $ready == 1 ]] || { printf '%s\n' '服务尚未就绪，请查看 status 和 logs。' >&2; return 1; }
    sdc_nested_docker info >/dev/null
    [[ -S $SDC_NESTED_RUN/buildkit.sock ]] || { printf '%s\n' 'BuildKit 未就绪，请查看 logs buildkit。' >&2; return 1; }
    nvidia-ctk cdi list | grep -Fx 'nvidia.com/gpu=all'
    if sdc_nested_docker buildx inspect sdc-nested >/dev/null 2>&1; then
        [[ $(sdc_nested_docker buildx inspect sdc-nested --format '{{.Driver}}') == remote ]] || {
            printf '%s\n' 'sdc-nested 构建器名称已被其他驱动占用。' >&2; return 1;
        }
        [[ $(sdc_nested_docker buildx inspect sdc-nested --format '{{range .Nodes}}{{.Endpoint}}{{end}}') == "unix://$SDC_NESTED_RUN/buildkit.sock" ]] || {
            printf '%s\n' 'sdc-nested 构建器指向其他地址。' >&2; return 1;
        }
        sdc_nested_docker buildx use sdc-nested
    else
        sdc_nested_docker buildx create --name sdc-nested --driver remote --use "unix://$SDC_NESTED_RUN/buildkit.sock"
    fi
    sdc_nested_docker buildx inspect sdc-nested --bootstrap
}

sdc_nested_stop() {
    local name
    for name in sdc-docker; do
        if [[ $(sdc_nested_service_state "$name") =~ ^(RUNNING|STARTING|BACKOFF)$ ]]; then
            sdc_nested_ctl stop "$name"
        fi
    done
    # Keep the mounted data filesystem available; never remove images or volumes.
}

sdc_nested_retire_proxy_services() {
    local name state
    for name in sdc-docker-proxy-socket sdc-docker-http-proxy; do
        state=$(sdc_nested_service_state "$name")
        if [[ $state =~ ^(RUNNING|STARTING|BACKOFF)$ ]]; then sdc_nested_ctl stop "$name"; fi
        if [[ $state =~ ^(RUNNING|STARTING|BACKOFF|STOPPED|EXITED|FATAL)$ ]]; then sdc_nested_ctl remove "$name"; fi
    done
}

sdc_nested_validate_existing_daemon() {
    if [[ -f $SDC_NESTED_DAEMON_CONFIG ]]; then
        [[ $(jq -r '."data-root" // empty' "$SDC_NESTED_DAEMON_CONFIG") == "$SDC_NESTED_MOUNT/docker" ]] || {
            printf '%s\n' '已有 Docker 配置使用其他存储目录，未覆盖。' >&2; return 1;
        }
    fi
}

sdc_nested_release_storage() {
    if [[ $(sdc_nested_service_state sdc-docker-storage) =~ ^(RUNNING|STARTING|BACKOFF)$ ]]; then
        sdc_nested_ctl stop sdc-docker-storage
    fi
    if mountpoint -q "$SDC_NESTED_MOUNT"; then
        if command -v fusermount3 >/dev/null; then
            fusermount3 -u "$SDC_NESTED_MOUNT"
        else
            fusermount -u "$SDC_NESTED_MOUNT"
        fi
    fi
}

sdc_nested_install_file() {
    local source=$1 destination=$2
    if [[ ! $source -ef $destination ]]; then install -m 0755 "$source" "$destination"; fi
}

sdc_nested_setup() {
    local name backup source_file
    [[ -c /dev/fuse && -r /dev/fuse && -w /dev/fuse ]] || { printf '%s\n' '需要平台开放可读写的 /dev/fuse。' >&2; return 1; }
    nvidia-smi >/dev/null
    if mountpoint -q "$SDC_NESTED_MOUNT"; then
        [[ $(findmnt -n -o SOURCE --target "$SDC_NESTED_MOUNT") == "$SDC_NESTED_STORAGE_FILE" ]] || {
            printf '%s\n' '存储挂载点已被其他文件系统使用。' >&2; return 1;
        }
    fi
    if [[ -n $(sdc_nested_docker ps -q 2>/dev/null) ]]; then
        printf '%s\n' '请先停止项目容器，再从外层 SSH 终端运行 setup。' >&2
        return 1
    fi
    source "$sdc_nested_dir/nested/install-tools.sh"
    sdc_nested_install_tools
    sdc_nested_validate_existing_daemon
    sdc_nested_select_supervisor
    if sdc_nested_ctl pid >/dev/null 2>&1; then
        sdc_nested_stop
        sdc_nested_retire_proxy_services
        sdc_nested_release_storage
    fi
    sdc_nested_storage
    mkdir -p "$SDC_NESTED_RUN" "$SDC_NESTED_MOUNT" "$SDC_NESTED_INSTALL/nested" "$SDC_NESTED_ETC"
    chmod 0700 "$SDC_NESTED_RUN"
    backup=$(mktemp -d "$SDC_NESTED_ETC/setup-backup.XXXXXXXX")
    for source_file in "$SDC_NESTED_CONFIG" "$SDC_NESTED_DAEMON_CONFIG" \
                       "$SDC_NESTED_PROGRAM_DIR/sdc-docker.conf" "$SDC_NESTED_PROGRAM_DIR/sdc-docker-proxy.conf" \
                       "$SDC_NESTED_PROGRAM_DIR/sdc-docker-storage.conf"; do
        [[ ! -f $source_file ]] || cp -p "$source_file" "$backup/$(basename "$source_file")"
    done
    for source_file in "$SDC_NESTED_PROGRAM_DIR/sdc-docker-proxy.conf" "$SDC_NESTED_ETC/tinyproxy.conf"; do
        [[ ! -e $source_file ]] || mv -f -- "$source_file" "$backup/"
    done
    for name in nested-docker.sh install-docker.sh; do
        sdc_nested_install_file "$sdc_nested_dir/$name" "$SDC_NESTED_INSTALL/$name"
    done
    for name in runtime.sh cdi.sh install-tools.sh; do
        sdc_nested_install_file "$sdc_nested_dir/nested/$name" "$SDC_NESTED_INSTALL/nested/$name"
    done
    sdc_nested_write_config
    cat > /usr/local/bin/sdc-docker-service <<EOF
#!/bin/sh
if [ "\$#" -eq 0 ]; then set -- status; fi
exec /bin/bash "$SDC_NESTED_INSTALL/nested-docker.sh" "\$@"
EOF
    chmod 0755 /usr/local/bin/sdc-docker-service
    sdc_nested_ensure_supervisor
    sdc_nested_ctl reread
    sdc_nested_ctl update sdc-docker-storage sdc-docker
    sdc_nested_start
    printf '%s\n' 'Docker 和 NVIDIA 已就绪。VS Code 选择“SDC NVIDIA（容器内 Docker）”。'
}

sdc_nested_main() {
    local action=${1:-setup} selector
    if (($#)); then shift; fi
    if [[ $action == -h || $action == --help ]]; then sdc_nested_usage; return; fi
    case $action in setup|start|stop|restart|status|logs) ;; *) sdc_nested_usage >&2; return 2 ;; esac
    [[ $(id -u) == 0 ]] || { printf '%s\n' '请在外层 SSH 环境以 root 执行。' >&2; return 1; }
    sdc_nested_defaults
    if [[ $action == setup ]]; then
        while (($#)); do
            case $1 in
                --storage-file) [[ $# -ge 2 ]] || return 2; SDC_NESTED_STORAGE_FILE=$(realpath -m "$2"); shift 2 ;;
                --storage-gib) [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || return 2; SDC_NESTED_STORAGE_GIB=$2; shift 2 ;;
                *) sdc_nested_usage >&2; return 2 ;;
            esac
        done
        mkdir -p "$SDC_NESTED_RUN"
        exec 9>"$SDC_NESTED_RUN/setup.lock"
        flock -n 9 || { printf '%s\n' '另一个 setup 正在运行。' >&2; return 1; }
        sdc_nested_setup
        return
    fi
    [[ -f $SDC_NESTED_CONFIG ]] || { printf '%s\n' '请先执行 nested-docker.sh setup。' >&2; return 1; }
    if [[ $action != logs && $# -gt 0 ]]; then sdc_nested_usage >&2; return 2; fi
    case $action in
        start) sdc_nested_start ;;
        stop) sdc_nested_stop ;;
        restart) sdc_nested_stop; sdc_nested_start ;;
        status)
            sdc_nested_ctl status sdc-docker-storage sdc-docker
            sdc_nested_docker info --format '{{.ServerVersion}}'
            df -h "$SDC_NESTED_MOUNT"
            ;;
        logs)
            selector=${1:-docker}
            case $selector in docker) selector=sdc-docker ;; buildkit) selector=sdc-buildkit ;; storage) selector=sdc-docker-storage ;; *) return 2 ;; esac
            tail -n 80 "$SDC_NESTED_LOG/$selector.log"
            ;;
    esac
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then sdc_nested_main "$@"; fi
