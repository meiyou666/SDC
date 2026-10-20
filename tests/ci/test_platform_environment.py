import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
from test_container_environment import runtime


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("platform_config", ROOT / "environment/configure_platform.py")
config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(config)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("SDC_DEPLOYMENT", "platform")
    root = tmp_path / "project with spaces"
    for relative in (*config.SNAPSHOT_FILES, ".vscode/launch.json", "environment/activate.sh"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return root


def make_cuda(root, version):
    (root / "bin").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "include/cuda.h").write_text("#define CUDA_VERSION 12080\n")
    compiler = root / "bin/nvcc"
    compiler.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release 12.8, V{version}'\n")
    compiler.chmod(0o755)
    return root


def test_preinstalled_nvcc_matches_without_version_json(tmp_path):
    baseline = json.loads((ROOT / "environment/baseline.json").read_text())
    cuda = make_cuda(tmp_path / "cuda", "12.8.93")
    assert config.cuda_sdk(baseline, str(cuda)) == (cuda, "12.8.93")
    make_cuda(tmp_path / "wrong-cuda", "13.0.48")
    with pytest.raises(ValueError, match="Expected nvcc 12.8.93"):
        config.cuda_sdk(baseline, str(tmp_path / "wrong-cuda"))
    (cuda / "include/cuda.h").unlink()
    with pytest.raises(ValueError, match="headers"):
        config.cuda_sdk(baseline, str(cuda))


@pytest.mark.parametrize("version", ["22.04", "24.04"])
def test_platform_releases_do_not_relax_docker_baseline(project, version):
    release = {"ID": "ubuntu", "VERSION_ID": version}
    assert config.preflight(project, "development", release)["SDC_RUNTIME"] == "development"
    baseline = json.loads((project / "environment/baseline.json").read_text())
    runtime.check_distribution(baseline, release, platform_environment=True)
    if version == "24.04":
        with pytest.raises(ValueError, match="Distribution"):
            runtime.check_distribution(baseline, release)
    with pytest.raises(ValueError, match="Choose Ubuntu"):
        config.preflight(project, "development", {"ID": "debian", "VERSION_ID": "12"})


def test_configuration_and_activation_keep_project_environment_separate(project, monkeypatch):
    cuda = make_cuda(project / "preinstalled cuda", "12.8.93")
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    config.configure(project, "nvidia")
    result = subprocess.run(
        ["bash", "-euc", 'source "$1"; printf "%s\\n" "$SDC_RUNTIME" "$UV_PROJECT_ENVIRONMENT" "$UV_PYTHON" "$PYTHONPATH" "$CUDA_HOME" "$CUTLASS_PATH"', "activation", str(project / "environment/activate.sh")],
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": "/foreign/venv", "PYTHONPATH": "/foreign/source"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["nvidia", str(project / ".venv"), str(project / ".venv/bin/python"),
                                          str(project / "src"), str(cuda), str(project / ".sdc/cutlass")]
    for relative in config.SNAPSHOT_FILES:
        assert (project / relative).read_bytes() == (project / ".sdc/project" / relative).read_bytes()


def test_workspace_sets_debug_environment_without_exporting_api_keys(project, monkeypatch):
    config.configure(project, "development")
    monkeypatch.setenv("OPENAI_API_KEY", "never-export-this")
    monkeypatch.setenv("SDC_RUNTIME", "development")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(project / ".venv"))
    output = config.workspace(project)
    data = json.loads(output.read_text())
    assert "never-export-this" not in output.read_text()
    assert data["settings"]["python.defaultInterpreterPath"] == str(project / ".venv/bin/python")
    assert data["launch"]["configurations"][0]["env"]["SDC_RUNTIME"] == "development"
    assert data["launch"]["configurations"][0]["python"] == str(project / ".venv/bin/python")
    assert json.loads((project / ".sdc/environment.json").read_text())["os"]["ID"] == "ubuntu"


def test_wrong_interpreter_and_changed_lock_are_rejected(project, monkeypatch):
    state = config.configure(project, "development")
    state["runtime"] = "development"
    with pytest.raises(ValueError, match="Activate"):
        runtime.verify_platform_environment(project, state)
    monkeypatch.setattr(runtime.sys, "prefix", str(project / ".venv"))
    monkeypatch.setenv("SDC_RUNTIME", "development")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: None)
    runtime.verify_platform_environment(project, state)
    (project / "uv.lock").write_text("changed lock\n")
    with pytest.raises(ValueError, match="uv.lock differs"):
        runtime.verify_platform_environment(project, state)


def test_failed_platform_preflight_never_installs_or_runs_docker(project):
    setup = project / "environment/setup-platform.sh"
    shutil.copyfile(ROOT / "environment/setup-platform.sh", setup)
    binary = project / "bin"
    binary.mkdir()
    for name, body in {"python3": "exit 42", "curl": "echo UNEXPECTED_INSTALL; exit 99", "docker": "echo UNEXPECTED_DOCKER; exit 99"}.items():
        path = binary / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    result = subprocess.run(["bash", str(setup), "nvidia"], env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}, capture_output=True, text=True)
    assert result.returncode == 42, result.stderr
    assert "UNEXPECTED" not in result.stdout
    assert not (project / ".sdc").exists()


def test_cutlass_backend_uses_platform_snapshot_and_requires_its_interpreter(project, monkeypatch):
    from cutlass_gemm import backend

    config.configure(project, "development")
    monkeypatch.setattr(backend, "ROOT", project)
    with pytest.raises(RuntimeError, match=".venv"):
        backend.environment_settings_path()
    monkeypatch.setattr(backend.sys, "prefix", str(project / ".venv"))
    assert backend.environment_settings_path() == project / ".sdc/project/environment/gemm.json"
    monkeypatch.setenv("SDC_DEPLOYMENT", "docker")
    assert backend.environment_settings_path() == Path("/opt/sdc/gemm.json")


@pytest.mark.parametrize("preinstalled", [True, False])
@pytest.mark.parametrize("python_mirror", ["", "https://mirror.example/python"])
def test_setup_uses_locked_profile_and_preserves_personal_env(project, preinstalled, python_mirror):
    for relative in ("environment/setup-platform.sh", "environment/configure_platform.py", "environment/verify_environment.py", ".env.example"):
        shutil.copyfile(ROOT / relative, project / relative)
    (project / ".env").write_text("OPENAI_API_KEY=keep-personal-value\n")
    binary = project / ".sdc/bin"
    binary.mkdir(parents=True)
    uv = binary / "uv" if preinstalled else project / "uv-wheel-executable"
    version = json.loads((project / "environment/baseline.json").read_text())["uv"]
    uv.write_text(f'''#!/usr/bin/env bash
set -eu
echo "$*" >> "$TEST_LOG"
case $1 in
  --version) echo 'uv {version} (x86_64-unknown-linux-gnu)' ;;
  python) printf 'python-mirror=%s\\npython-bin=%s\\n' "$UV_PYTHON_INSTALL_MIRROR" "$UV_PYTHON_BIN_DIR" >> "$TEST_LOG" ;;
  sync)
    mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
    cp "$TEST_PYTHON_WRAPPER" "$UV_PROJECT_ENVIRONMENT/bin/python"
    chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"
    ;;
  *) exit 99 ;;
esac
''')
    uv.chmod(0o755)
    bootstrap = binary / "python3"
    bootstrap.write_text(f'''#!/usr/bin/env bash
set -eu
if [[ ${{1:-}} == -m && ${{2:-}} == pip ]]; then
    echo "pip $*" >> "$TEST_LOG"
    if [[ ${{3:-}} == --version ]]; then echo 'pip fixture'; exit 0; fi
    while (($#)); do
        if [[ $1 == --target ]]; then target=$2; shift 2; else shift; fi
    done
    mkdir -p "$target/bin"
    cp "$TEST_UV_WHEEL" "$target/bin/uv"
    cp "$TEST_UV_WHEEL" "$target/bin/uvx"
    exit 0
fi
exec {shlex.quote(sys.executable)} "$@"
''')
    bootstrap.chmod(0o755)
    wrapper = project / "python-wrapper"
    # Run the actual configurator, but replace library/device validation since this
    # test deliberately avoids downloading a second PyTorch environment.
    wrapper.write_text(f'''#!/usr/bin/env bash
if [[ $1 == environment/verify_environment.py ]]; then
    echo "verify $*" >> "$TEST_LOG"
    exit 0
fi
exec {shlex.quote(sys.executable)} "$@"
''')
    log = project / "commands.log"
    result = subprocess.run(["bash", str(project / "environment/setup-platform.sh"), "development"],
                            env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "TEST_LOG": str(log),
                                 "TEST_PYTHON_WRAPPER": str(wrapper), "TEST_UV_WHEEL": str(uv),
                                 "UV_PYTHON_INSTALL_MIRROR": python_mirror}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "sync --locked --all-groups --extra development --no-build" in log.read_text()
    assert "verify environment/verify_environment.py --smoke" in log.read_text()
    expected_mirror = python_mirror or "https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone"
    assert f"python-mirror={expected_mirror}" in log.read_text()
    assert f"python-bin={binary}" in log.read_text()
    assert (project / ".sdc/SDC-platform.code-workspace").is_file()
    assert (project / ".env").read_text() == "OPENAI_API_KEY=keep-personal-value\n"
    if preinstalled:
        assert "pip " not in log.read_text()
    else:
        assert "--index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" in log.read_text()
        assert "--no-deps --only-binary=:all:" in log.read_text()
        assert f"uv=={version}" in log.read_text()
        assert (binary / "uv").is_file()
        assert (binary / "uvx").is_file()
        assert not list((project / ".sdc").glob("uv-download.*"))


def test_cutlass_install_is_repeatable_and_rejects_modified_sources(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run(["git", "init", str(origin)], check=True, capture_output=True)
    (origin / "README").write_text("official fixture\n")
    subprocess.run(["git", "-C", str(origin), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(origin), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(origin), "rev-parse", "HEAD"], text=True).strip()
    manifest = tmp_path / "gemm.json"
    manifest.write_text(json.dumps({"cutlass_commit": commit}))
    target = tmp_path / "cutlass"
    env = {**os.environ, "SDC_GEMM_CONFIG": str(manifest), "CUTLASS_PATH": str(target), "GIT_CONFIG_COUNT": "1",
           "GIT_CONFIG_KEY_0": f"url.{origin.as_uri()}.insteadOf", "GIT_CONFIG_VALUE_0": "https://github.com/NVIDIA/cutlass.git", "GIT_ALLOW_PROTOCOL": "file"}
    command = ["bash", str(ROOT / "environment/docker/install-cutlass.sh")]
    for _ in range(2):
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    (target / "README").write_text("local change\n")
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert (target / "README").read_text() == "local change\n"
