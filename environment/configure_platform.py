"""Configure a project virtual environment using an existing platform SDK."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys


SNAPSHOT_FILES = ("pyproject.toml", "uv.lock", ".python-version", "environment/baseline.json", "environment/gemm.json")


def cuda_sdk(baseline, cuda_home=None):
    compiler = shutil.which("nvcc")
    root = Path(cuda_home or (str(Path(compiler).resolve().parents[1]) if compiler else "/usr/local/cuda-12.8"))
    output = subprocess.check_output([str(root / "bin/nvcc"), "--version"], text=True)
    expected = baseline["training_profiles"]["nvidia"]["nvcc"]
    found = re.search(r"\bV(\d+\.\d+\.\d+)\b", output)
    if not found or found[1] != expected:
        raise ValueError(f"Expected nvcc {expected}; found: {output.strip()}")
    if not (root / "include/cuda.h").is_file():
        raise ValueError(f"CUDA development headers are missing from {root}.")
    return root.resolve(), expected


def cann_sdk(baseline):
    expected = baseline["training_profiles"]["ascend"]["cann"]
    for root in (Path("/usr/local/Ascend/cann"), Path("/usr/local/Ascend/ascend-toolkit")):
        for metadata in (root / "latest/version.cfg", root / "version.cfg"):
            if metadata.is_file() and re.search(rf"(?m)^\s*(?:version|Version)\s*=\s*{re.escape(expected)}\s*$", metadata.read_text()):
                script = root / "set_env.sh"
                if script.is_file():
                    return script
    raise ValueError(f"Expected a preinstalled CANN {expected} SDK with version.cfg and set_env.sh.")


def preflight(project, runtime, release=None):
    baseline = json.loads((project / "environment/baseline.json").read_text())
    release = release or platform.freedesktop_os_release()
    if release.get("ID") != baseline["os_id"] or release.get("VERSION_ID") not in baseline["platform_os_versions"]:
        raise ValueError(f"Choose Ubuntu {' or '.join(baseline['platform_os_versions'])} for the platform environment.")
    settings = {"SDC_RUNTIME": runtime, "SDC_DEPLOYMENT": "platform"}
    if runtime == "nvidia":
        root, _ = cuda_sdk(baseline, os.environ.get("CUDA_HOME"))
        settings.update(CUDA_HOME=str(root), CUTLASS_PATH=str(project / ".sdc/cutlass"))
    elif runtime == "ascend":
        settings["SDC_CANN_ENV"] = str(cann_sdk(baseline))
    return settings


def configure(project, runtime):
    settings = preflight(project, runtime)
    directory = project / ".sdc"
    directory.mkdir(exist_ok=True)
    for relative in SNAPSHOT_FILES:
        target = directory / "project" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, target)
    (directory / "config.sh").write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()))
    (directory / "environment.json").write_text(json.dumps({"runtime": runtime, **settings}, indent=2) + "\n")
    return settings


def workspace(project):
    # Only export toolchain settings; never copy the session's API keys or tokens.
    keys = {"PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "VIRTUAL_ENV", "CUDA_HOME", "CUTLASS_PATH", "SDC_RUNTIME", "SDC_DEPLOYMENT",
            "UV_PROJECT_ENVIRONMENT", "UV_PYTHON", "UV_PYTHON_DOWNLOADS", "PYTHONNOUSERSITE", "PYTHONUNBUFFERED", "MPLBACKEND"}
    env = {key: value for key, value in os.environ.items() if key in keys or key.startswith(("ASCEND_", "CANN_"))}
    launch = json.loads((project / ".vscode/launch.json").read_text())
    for config in launch["configurations"]:
        config["name"] = "平台容器：按 YAML 调试"
        config["python"] = str(project / ".venv/bin/python")
        config["env"] = {**env, **config.get("env", {})}
    bashrc = project / ".sdc/bashrc"
    bashrc.write_text('if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi\n'
                      'export SDC_DEPLOYMENT=platform\n'
                      f'source {shlex.quote(str(project / "environment/activate.sh"))}\n')
    content = {
        "folders": [{"path": ".."}],
        "settings": {"python.defaultInterpreterPath": str(project / ".venv/bin/python"),
                     "terminal.integrated.env.linux": env, "python.terminal.activateEnvironment": False,
                     "terminal.integrated.profiles.linux": {"SDC": {"path": "/bin/bash", "args": ["--rcfile", str(bashrc)]}},
                     "terminal.integrated.defaultProfile.linux": "SDC"},
        "launch": launch,
    }
    target = project / ".sdc/SDC-platform.code-workspace"
    target.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    state_file = project / ".sdc/environment.json"
    state = json.loads(state_file.read_text())
    state.update(os=platform.freedesktop_os_release(), architecture=platform.machine(), python=sys.version)
    state["tools"] = {}
    commands = {"uv": ["uv", "--version"], "compiler": ["g++", "--version"]}
    if state["runtime"] == "nvidia":
        commands.update(nvcc=[str(Path(state["CUDA_HOME"]) / "bin/nvcc"), "--version"],
                        driver=["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    for name, command in commands.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            state["tools"][name] = {"exit_code": result.returncode, "output": result.stdout.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            state["tools"][name] = {"error": str(exc)}
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "configure", "workspace"))
    parser.add_argument("--profile", choices=("development", "nvidia", "ascend"), default="nvidia")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    try:
        if args.action == "workspace":
            print(workspace(project))
        elif args.action == "configure":
            configure(project, args.profile)
        else:
            preflight(project, args.profile)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Platform environment: {exc}\n")


if __name__ == "__main__":
    main()
