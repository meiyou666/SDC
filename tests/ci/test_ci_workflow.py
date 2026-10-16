import os
from pathlib import Path
import shlex
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/environment-consistency.yml").read_text())


def cache_entries(value):
    return [dict(part.split("=", 1) for part in line.split(",")) for line in value.splitlines()]


def test_ci_builds_load_the_images_used_by_compose_with_reusable_cache():
    services = yaml.safe_load((ROOT / "compose.yaml").read_text())["services"]
    steps = workflow()["jobs"]["environment-consistency"]["steps"]
    builds = [step["with"] for step in steps if step.get("uses", "").startswith("docker/build-push-action@")]
    assert len(builds) == 2
    exported_scopes = set()
    for service in (services["tools"], services["dev"]):
        build, = [item for item in builds if item["target"] == service["build"]["target"]]
        assert build["context"] == service["build"]["context"]
        assert build["file"] == service["build"]["dockerfile"]
        assert service["image"] in build["tags"].splitlines()
        assert build["load"] is True
        exports = cache_entries(build["cache-to"])
        imports = cache_entries(build["cache-from"])
        assert all(item["type"] == "gha" and item["mode"] == "max" for item in exports)
        scopes = {item["scope"] for item in exports}
        assert scopes <= {item["scope"] for item in imports}
        assert not scopes & exported_scopes
        exported_scopes.update(scopes)
    dev, = [item for item in builds if item["target"] == services["dev"]["build"]["target"]]
    assert exported_scopes <= {item["scope"] for item in cache_entries(dev["cache-from"])}


def test_ci_executes_required_checks_without_rebuilding(tmp_path):
    config = workflow()
    steps = config["jobs"]["environment-consistency"]["steps"]
    commands = tmp_path / "commands"
    # Execute the workflow shell, intercepting Docker before any container or build runs.
    env = {**os.environ, "COMMAND_LOG": str(commands)}
    mock = 'docker() { printf "%s\\n" "$*" >> "$COMMAND_LOG"; }\n'
    for step in steps:
        if "run" in step:
            result = subprocess.run(["bash", "-eu", "-c", mock + step["run"]], env={**env, **{key: "fixture" for key in step.get("env", {})}}, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
    calls = [shlex.split(line) for line in commands.read_text().splitlines()]
    assert all("--build" not in call and "build" not in call for call in calls)
    assert any(call[-3:] == ["tools", "lock", "--check"] for call in calls)
    assert any("scripts/ci/check_environment.py" in call for call in calls)
    assert any(call[-3:] == ["dev", "bash", "scripts/check.sh"] for call in calls)
    assert any("environment/verify_environment.py" in call and "--smoke" in call for call in calls)
    assert config["concurrency"]["cancel-in-progress"] is True
    assert "github.event.pull_request.number" in config["concurrency"]["group"]


@pytest.mark.parametrize("arch,override,expected", [
    ("amd64", "", "mirrors.tuna.tsinghua.edu.cn/ubuntu"),
    ("arm64", "", "mirrors.tuna.tsinghua.edu.cn/ubuntu-ports"),
    ("amd64", "http://archive.ubuntu.com/ubuntu", "archive.ubuntu.com/ubuntu"),
    ("amd64", "https://archive.ubuntu.com/ubuntu", "archive.ubuntu.com/ubuntu"),
])
def test_apt_mirror_bootstraps_certificates_before_switching_to_https(tmp_path, arch, override, expected):
    sources = tmp_path / "sources.list"
    log = tmp_path / "apt.log"
    script = (ROOT / "environment/docker/install-system.sh").read_text().replace("/etc/apt/sources.list", shlex.quote(str(sources)))
    mock = '''
dpkg() { printf '%s\\n' "$TEST_ARCH"; }
apt-get() {
    printf '%s\\n' "$*" >> "$TEST_LOG"
    cat "$TEST_SOURCES" >> "$TEST_LOG"
}
rm() { :; }
'''
    env = {**os.environ, "SDC_APT_MIRROR": override, "TEST_ARCH": arch, "TEST_LOG": str(log), "TEST_SOURCES": str(sources)}
    result = subprocess.run(["bash", "-c", mock + script, "apt-test", "minimal"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    entries = [line.split() for line in sources.read_text().splitlines()]
    assert {entry[1] for entry in entries} == {f"https://{expected}"}
    assert {entry[2] for entry in entries} == {"jammy", "jammy-updates", "jammy-security"}
    bootstrap_scheme = "https" if override.startswith("https:") else "http"
    history = log.read_text()
    assert "ca-certificates" in history
    assert f"{bootstrap_scheme}://{expected}" in history
    if bootstrap_scheme == "http":
        assert f"https://{expected}" not in history


def test_shared_check_runner_keeps_locked_tests_and_lint(tmp_path):
    log = tmp_path / "checks.log"
    # Only replace accelerator initialization; run the actual launcher control flow.
    script = (ROOT / "scripts/check.sh").read_text().replace(
        "source /opt/sdc/activate-accelerator.sh", ":"
    )
    mock = '''
python() { printf 'python %s\\n' "$*" >> "$TEST_LOG"; }
uv() { printf 'uv %s\\n' "$*" >> "$TEST_LOG"; }
'''
    env = {**os.environ, "TEST_LOG": str(log), "SDC_RUNTIME": "development"}
    result = subprocess.run(["bash", "-c", mock + script, str(ROOT / "scripts/check.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [shlex.split(line) for line in log.read_text().splitlines()]
    for command in ("pytest", "ruff"):
        call, = [call for call in calls if command in call]
        assert call[:5] == ["uv", "run", "--locked", "--extra", "development"]
    assert any("--smoke" in call and "--project" in call for call in calls)
