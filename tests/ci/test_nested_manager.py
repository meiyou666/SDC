"""Exercise the nested service manager without starting services or mounting filesystems."""

import os
from pathlib import Path
import shlex
import socket
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
MANAGER = ROOT / "environment/docker/nested-docker.sh"


def run_manager(tmp_path, script):
    env = os.environ.copy()
    env.update(
        SDC_NESTED_CONFIG=str(tmp_path / "settings.conf"),
        SDC_NESTED_ETC=str(tmp_path / "etc"),
        SDC_NESTED_RUN=str(tmp_path),
        SDC_NESTED_LOG=str(tmp_path / "log"),
        SDC_NESTED_STORAGE_FILE=str(tmp_path / "storage.ext4"),
        SDC_NESTED_STORAGE_GIB="1",
        EVENTS=str(tmp_path / "events"),
    )
    return subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(MANAGER))}\n{script}"],
        env=env, text=True, capture_output=True, check=False,
    )


@pytest.mark.parametrize("arguments", ["unknown", "setup --unknown", "setup --storage-gib 0", "setup --storage-file"])
def test_invalid_cli_never_reaches_setup(tmp_path, arguments):
    result = run_manager(tmp_path, f'''
        id() {{ echo 0; }}
        sdc_nested_setup() {{ touch "$EVENTS"; }}
        sdc_nested_main {arguments}
    ''')
    assert result.returncode == 2, result.stderr
    assert not (tmp_path / "events").exists()
    assert not (tmp_path / "setup.lock").exists()


@pytest.mark.parametrize("filesystem, expected", [("ext4", 0), ("xfs", 1)])
def test_existing_storage_is_never_formatted(tmp_path, filesystem, expected):
    storage = tmp_path / "storage.ext4"
    original = b"existing filesystem contents\x00"
    storage.write_bytes(original)
    result = run_manager(tmp_path, f'''
        blkid() {{ echo {filesystem}; }}
        truncate() {{ echo truncate >> "$EVENTS"; return 99; }}
        mkfs.ext4() {{ echo format >> "$EVENTS"; return 99; }}
        sdc_nested_storage
    ''')
    assert result.returncode == expected, result.stderr
    assert storage.read_bytes() == original
    assert not (tmp_path / "events").exists()


@pytest.mark.parametrize("failure", ["truncate", "mkfs.ext4"])
def test_failed_new_storage_does_not_publish_partial_image(tmp_path, failure):
    result = run_manager(tmp_path, f'''
        truncate() {{ printf partial > "${{@: -1}}"; [[ {shlex.quote(failure)} != truncate ]]; }}
        mkfs.ext4() {{ return 1; }}
        sdc_nested_storage
    ''')
    assert result.returncode != 0
    assert not (tmp_path / "storage.ext4").exists()
    assert not list(tmp_path.glob(".sdc-storage.*"))


def test_concurrent_creation_preserves_winning_storage(tmp_path):
    result = run_manager(tmp_path, '''
        truncate() { printf partial > "${@: -1}"; }
        mkfs.ext4() { printf winner > "$SDC_NESTED_STORAGE_FILE"; }
        sdc_nested_storage
    ''')
    assert result.returncode != 0
    assert (tmp_path / "storage.ext4").read_text() == "winner"
    assert not list(tmp_path.glob(".sdc-storage.*"))


@pytest.mark.parametrize("running", ["", "sdc-docker-storage", "all"])
def test_start_orders_dependencies_and_skips_running_services(tmp_path, running):
    services = ["sdc-docker-storage", "sdc-docker"]
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(tmp_path / "buildkit.sock"))
        result = run_manager(tmp_path, f'''
            sdc_nested_ensure_supervisor() {{ :; }}
            sdc_nested_ctl() {{
                if [[ $1 == status ]]; then
                    if [[ {shlex.quote(running)} == all || $2 == {shlex.quote(running)} || -f $SDC_NESTED_RUN/$2.started ]]; then
                        echo "$2 RUNNING pid 1, uptime 1:00:00"
                    else echo "$2 STOPPED"; fi
                else
                    echo "$*" >> "$EVENTS"
                    touch "$SDC_NESTED_RUN/$2.started"
                fi
            }}
            sdc_nested_docker() {{
                if [[ $* == 'buildx inspect sdc-nested' ]]; then return 1; fi
            }}
            nvidia-ctk() {{ echo nvidia.com/gpu=all; }}
            sleep() {{ return 99; }}
            sdc_nested_start
        ''')
    assert result.returncode == 0, result.stderr
    events = (tmp_path / "events").read_text().splitlines() if (tmp_path / "events").exists() else []
    assert events == [f"start {name}" for name in services if running != "all" and name != running]


def test_stop_preserves_storage_and_stops_consumers_first(tmp_path):
    storage = tmp_path / "storage.ext4"
    storage.write_bytes(b"keep images and volumes")
    result = run_manager(tmp_path, '''
        sdc_nested_ctl() {
            if [[ $1 == status ]]; then echo "$2 RUNNING pid 1, uptime 1:00:00";
            else echo "$*" >> "$EVENTS"; fi
        }
        rm() { echo rm >> "$EVENTS"; return 99; }
        truncate() { echo truncate >> "$EVENTS"; return 99; }
        mkfs.ext4() { echo format >> "$EVENTS"; return 99; }
        sdc_nested_stop
    ''')
    assert result.returncode == 0, result.stderr
    assert storage.read_bytes() == b"keep images and volumes"
    assert (tmp_path / "events").read_text().splitlines() == [
        "stop sdc-docker",
    ]


@pytest.mark.parametrize("private, alive, expected", [(1, False, 0), (0, False, 1), (1, True, 0), (0, True, 0)])
def test_supervisor_only_launches_private_instance_when_missing(tmp_path, private, alive, expected):
    result = run_manager(tmp_path, f'''
        SDC_NESTED_PRIVATE_SUPERVISOR={private}
        SDC_NESTED_SUPERVISORD=mock_supervisord
        SDC_NESTED_SUPERVISOR_CONFIG="$SDC_NESTED_CONFIG"
        sdc_nested_ctl() {{ return {0 if alive else 1}; }}
        mock_supervisord() {{ echo "$*" >> "$EVENTS"; }}
        sdc_nested_ensure_supervisor
    ''')
    assert result.returncode == expected, result.stderr
    assert (tmp_path / "events").exists() == (private == 1 and not alive)


@pytest.mark.parametrize("private", [0, 1])
def test_configuration_preserves_daemon_extensions_and_platform_supervisor(tmp_path, private):
    import json

    daemon = tmp_path / "daemon.json"
    original = {"registry-mirrors": ["https://mirror.example"], "runtimes": {"custom": {"path": "/custom"}}}
    daemon.write_text(json.dumps(original))
    supervisor = tmp_path / "supervisord.conf"
    supervisor.write_text("platform managed configuration\n")
    result = run_manager(tmp_path, f'''
        sdc_nested_defaults
        SDC_NESTED_MOUNT="$SDC_NESTED_RUN/mount"
        SDC_NESTED_INSTALL="$SDC_NESTED_RUN/install"
        SDC_NESTED_LOG="$SDC_NESTED_RUN/log"
        SDC_NESTED_CDI_DIR="$SDC_NESTED_RUN/cdi"
        SDC_NESTED_DAEMON_CONFIG="$SDC_NESTED_RUN/daemon.json"
        SDC_NESTED_BUILDKIT_ROOT="$SDC_NESTED_RUN/cache"
        SDC_NESTED_BUILDKIT_BIN=/mock/buildkitd
        SDC_NESTED_SUPERVISOR_CONFIG="$SDC_NESTED_RUN/supervisord.conf"
        SDC_NESTED_SUPERVISORCTL=/mock/supervisorctl
        SDC_NESTED_SUPERVISORD=/mock/supervisord
        SDC_NESTED_PROGRAM_DIR="$SDC_NESTED_RUN/programs"
        SDC_NESTED_PRIVATE_SUPERVISOR={private}
        nvidia-ctk() {{ echo "$*" >> "$EVENTS"; }}
        sdc_nested_write_config
        source "$SDC_NESTED_CONFIG"
        [[ $SDC_NESTED_STORAGE_FILE == "$SDC_NESTED_RUN/storage.ext4" ]]
    ''')
    assert result.returncode == 0, result.stderr
    configured = json.loads(daemon.read_text())
    assert configured["registry-mirrors"] == original["registry-mirrors"]
    assert configured["runtimes"]["custom"] == original["runtimes"]["custom"]
    assert configured["data-root"] == str(tmp_path / "mount/docker")
    assert configured["hosts"] == [f"unix://{tmp_path}/docker.sock"]
    assert configured["rootless"] is True
    assert configured["storage-driver"] == "fuse-overlayfs"
    assert configured["bridge"] == "none"
    for key in ["iptables", "ip6tables", "ip-forward", "ip-masq"]:
        assert configured[key] is False
    assert json.loads((tmp_path / "etc/daemon.before-setup.json").read_text()) == original
    assert (tmp_path / "settings.conf").stat().st_mode & 0o777 == 0o600
    if private:
        assert f"files={tmp_path}/programs/*.conf" in supervisor.read_text()
    else:
        assert supervisor.read_text() == "platform managed configuration\n"
    assert "nvidia-container-runtime.mode=cdi" in (tmp_path / "events").read_text()
    programs = "\n".join(path.read_text() for path in (tmp_path / "programs").glob("*.conf"))
    for service in ["sdc-docker-storage", "sdc-docker"]:
        assert f"[program:{service}]" in programs
    assert "proxy" not in programs
    assert not (tmp_path / "etc/tinyproxy.conf").exists()


@pytest.mark.parametrize("contents", ['{"data-root":"/foreign/docker"}', '{}', 'invalid json'])
def test_foreign_or_invalid_daemon_configuration_is_rejected_unchanged(tmp_path, contents):
    daemon = tmp_path / "daemon.json"
    daemon.write_text(contents)
    result = run_manager(tmp_path, '''
        SDC_NESTED_DAEMON_CONFIG="$SDC_NESTED_RUN/daemon.json"
        SDC_NESTED_MOUNT="$SDC_NESTED_RUN/mount"
        sdc_nested_validate_existing_daemon
    ''')
    assert result.returncode != 0
    assert daemon.read_text() == contents


def test_matching_daemon_storage_is_accepted(tmp_path):
    import json

    daemon = tmp_path / "daemon.json"
    daemon.write_text(json.dumps({"data-root": str(tmp_path / "mount/docker")}))
    result = run_manager(tmp_path, '''
        SDC_NESTED_DAEMON_CONFIG="$SDC_NESTED_RUN/daemon.json"
        SDC_NESTED_MOUNT="$SDC_NESTED_RUN/mount"
        sdc_nested_validate_existing_daemon
    ''')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mounted", [True, False])
def test_setup_storage_release_stops_service_before_unmount(tmp_path, mounted):
    storage = tmp_path / "storage.ext4"
    storage.write_bytes(b"persistent data")
    result = run_manager(tmp_path, f'''
        SDC_NESTED_MOUNT="$SDC_NESTED_RUN/mount"
        sdc_nested_ctl() {{
            if [[ $1 == status ]]; then echo "$2 RUNNING pid 1, uptime 1:00:00";
            else echo "$*" >> "$EVENTS"; fi
        }}
        mountpoint() {{ return {0 if mounted else 1}; }}
        fusermount3() {{ echo "unmount $*" >> "$EVENTS"; }}
        fusermount() {{ echo "unmount $*" >> "$EVENTS"; }}
        sdc_nested_release_storage
    ''')
    assert result.returncode == 0, result.stderr
    expected = ["stop sdc-docker-storage"]
    if mounted:
        expected.append(f"unmount -u {tmp_path}/mount")
    assert (tmp_path / "events").read_text().splitlines() == expected
    assert storage.read_bytes() == b"persistent data"


@pytest.mark.parametrize("initial_state", ["STARTING", "BACKOFF"])
def test_start_waits_for_autostart_without_duplicate_start(tmp_path, initial_state):
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(tmp_path / "buildkit.sock"))
        result = run_manager(tmp_path, f'''
            sdc_nested_ensure_supervisor() {{ :; }}
            sdc_nested_ctl() {{
                if [[ $1 != status ]]; then echo "$*" >> "$EVENTS"; return 99; fi
                if [[ -e $SDC_NESTED_RUN/ready ]]; then echo "$2 RUNNING pid 1";
                else echo "$2 {initial_state}"; fi
            }}
            sleep() {{ echo wait >> "$EVENTS"; touch "$SDC_NESTED_RUN/ready"; }}
            sdc_nested_docker() {{
                [[ -e $SDC_NESTED_RUN/ready ]] || return 99
                if [[ $* == 'buildx inspect sdc-nested' ]]; then return 1; fi
            }}
            nvidia-ctk() {{ echo nvidia.com/gpu=all; }}
            sdc_nested_start
        ''')
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "events").read_text().splitlines() == ["wait"]


def test_supervisor_recreates_volatile_run_directory_before_launch(tmp_path):
    result = run_manager(tmp_path, '''
        SDC_NESTED_RUN="$SDC_NESTED_RUN/rebooted-run"
        SDC_NESTED_PRIVATE_SUPERVISOR=1
        SDC_NESTED_SUPERVISORD=mock_supervisord
        SDC_NESTED_SUPERVISOR_CONFIG="$SDC_NESTED_CONFIG"
        sdc_nested_ctl() { return 1; }
        mock_supervisord() {
            [[ -d $SDC_NESTED_RUN && -d $SDC_NESTED_LOG ]]
            [[ $(stat -c %a "$SDC_NESTED_RUN") == 700 ]]
            echo launched > "$EVENTS"
        }
        sdc_nested_ensure_supervisor
    ''')
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "events").read_text() == "launched\n"
    assert (tmp_path / "rebooted-run").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("state", ["RUNNING", "STOPPED", "MISSING"])
def test_migration_retires_only_existing_proxy_services(tmp_path, state):
    result = run_manager(tmp_path, f'''
        sdc_nested_ctl() {{
            if [[ $1 == status ]]; then
                if [[ {state} == MISSING ]]; then echo "$2: ERROR (no such process)"; return 3; fi
                echo "$2 {state}"
            else echo "$*" >> "$EVENTS"; fi
        }}
        sdc_nested_retire_proxy_services
    ''')
    assert result.returncode == 0, result.stderr
    expected = []
    for service in ["sdc-docker-proxy-socket", "sdc-docker-http-proxy"]:
        if state == "RUNNING":
            expected.append(f"stop {service}")
        if state != "MISSING":
            expected.append(f"remove {service}")
    events = (tmp_path / "events").read_text().splitlines() if (tmp_path / "events").exists() else []
    assert events == expected
