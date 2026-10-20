import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "environment/docker/nested/install-tools.sh"
DOCKER = ROOT / "environment/docker/install-docker.sh"


def run_bash(script, **env):
    return subprocess.run(
        ["bash", "-euc", script], text=True, capture_output=True,
        env={**os.environ, **{key: str(value) for key, value in env.items()}},
    )


def test_install_only_reuses_docker_without_daemon_access():
    result = run_bash('''
source "$INSTALLER"
docker() { [[ $1 == compose || $1 == buildx ]]; }
dockerd() { return 0; }
rootlesskit() { return 0; }
sdc_root() { echo "unexpected privileged operation: $*" >&2; return 99; }
sdc_install_docker root install-only
''', INSTALLER=DOCKER)
    assert result.returncode == 0, result.stderr


def test_reuses_buildkit_from_path(tmp_path):
    binary = tmp_path / "buildkitd"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    result = run_bash('''
source "$INSTALLER"
id() { echo 0; }
sdc_nested_install_packages() { :; }
sdc_nested_install_buildkit() { echo unexpected-download >&2; return 99; }
sdc_nested_install_tools
printf '%s' "$SDC_NESTED_BUILDKIT_BIN"
''', INSTALLER=TOOLS, PATH=f"{tmp_path}:{os.environ['PATH']}")
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(binary)


def test_package_failure_restores_existing_policy(tmp_path):
    # Sandbox the policy and staging paths; fail before touching apt or Docker.
    policy = tmp_path / "policy-rc.d"
    target = tmp_path / "original-policy"
    target.write_text("original policy\n")
    policy.symlink_to(target)
    helper = tmp_path / "helper.sh"
    helper.write_text(TOOLS.read_text().replace(
        "policy=/usr/sbin/policy-rc.d", f"policy={policy}"
    ))
    result = run_bash('''
source "$INSTALLER"
dpkg-query() { echo missing; }
apt-get() { return 42; }
sdc_nested_install_packages
''', INSTALLER=helper)
    assert result.returncode == 42, result.stderr
    assert policy.is_symlink()
    assert policy.read_text() == "original policy\n"


def test_buildkit_invalid_checksum_stops_before_extracting():
    result = run_bash('''
source "$INSTALLER"
curl() { printf 'corrupt archive' > "${@: -1}"; }
tar() { echo unexpected-extract >&2; return 99; }
sdc_nested_install_buildkit
''', INSTALLER=TOOLS)
    assert result.returncode != 0
    assert "unexpected-extract" not in result.stderr
    assert "FAILED" in result.stdout
