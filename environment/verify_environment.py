#!/usr/bin/env python3
"""Verify the running development environment and optionally exercise its libraries."""

import argparse
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def verify_baked_files(project, baked_root):
    for relative in ("pyproject.toml", "uv.lock", ".python-version", "environment/baseline.json"):
        supplied, baked = project / relative, baked_root / relative
        if supplied.resolve() != baked.resolve() and supplied.read_bytes() != baked.read_bytes():
            raise ValueError(f"{supplied.name} differs from the files used to build this image.")


def differences(baseline, python_version, system, machine, uv_version):
    errors = []
    if baseline.get("state") != "frozen" or baseline.get("profile") not in {"cpu-development", "shared-development"}:
        errors.append("Expected the frozen shared development profile.")
    if python_version != baseline.get("python"):
        errors.append(f"Python: expected {baseline.get('python')}, found {python_version}.")
    architecture = {"x86_64": "linux/amd64", "AMD64": "linux/amd64", "aarch64": "linux/arm64"}.get(machine)
    if system != "Linux" or architecture not in baseline.get("platforms", [baseline.get("platform")]):
        errors.append(f"Platform: unsupported {system}/{machine}.")
    if uv_version != baseline.get("uv"):
        errors.append(f"uv: expected {baseline.get('uv')}, found {uv_version}.")
    return errors


def check_uv_environment(project, runtime=None):
    command = ["uv", "sync", "--locked", "--check", "--offline", "--all-groups", "--project", str(project)]
    if runtime:
        command += ["--extra", runtime]
    result = subprocess.run(
        command,
        capture_output=True, text=True,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "uv environment check failed.")


def accelerator_smoke(runtime, baseline):
    if runtime not in baseline.get("training_profiles", {}):
        raise ValueError("Select the nvidia or ascend container for the device check.")
    import torch

    device_name = baseline["training_profiles"][runtime]["device"]
    backend = getattr(torch, device_name, None)
    if backend is None or not backend.is_available():
        raise ValueError(f"{device_name} device is unavailable. Check the host driver and container device mapping.")
    if runtime == "nvidia":
        from cuda.bindings import runtime as cudart

        status, cuda_runtime = cudart.cudaRuntimeGetVersion()
        if status != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot query the CUDA runtime version: {status}.")
        check_cuda_versions(baseline["training_profiles"][runtime], torch.version.cuda, cuda_runtime)
    a = torch.tensor([[1., 2.], [3., 4.]], device=device_name)
    result = a @ a
    backend.synchronize()
    torch.testing.assert_close(result.cpu(), torch.tensor([[7., 10.], [15., 22.]]))
    return {"device": device_name, "count": backend.device_count(), "name": backend.get_device_name(0)}


def check_distribution(baseline, release, platform_environment=False):
    expected = (baseline.get("os_id"), baseline.get("os_version"))
    actual = (release.get("ID"), release.get("VERSION_ID"))
    allowed = baseline.get("platform_os_versions", [expected[1]]) if platform_environment else [expected[1]]
    if actual[0] != expected[0] or actual[1] not in allowed:
        raise ValueError(f"Distribution: expected {expected[0]} {expected[1]}, found {actual[0]} {actual[1]}.")


def verify_platform_environment(project, state):
    if Path(sys.prefix).resolve() != (project / ".venv").resolve():
        raise ValueError("Activate this project's .venv with source environment/activate.sh.")
    if os.environ.get("SDC_RUNTIME") != state["runtime"]:
        raise ValueError("SDC_RUNTIME differs from the installed platform profile. Activate the project environment.")
    snapshot = project / ".sdc/project"
    verify_baked_files(project, snapshot)
    if (project / "environment/gemm.json").read_bytes() != (snapshot / "environment/gemm.json").read_bytes():
        raise ValueError("GEMM configuration changed. Run setup-platform.sh again.")
    subprocess.run([sys.executable, str(project / "environment/configure_platform.py"), "check", "--profile", state["runtime"]], check=True)
    if state["runtime"] == "nvidia":
        if os.environ.get("CUDA_HOME") != state["CUDA_HOME"] or os.environ.get("CUTLASS_PATH") != state["CUTLASS_PATH"]:
            raise ValueError("CUDA_HOME or CUTLASS_PATH differs from the configured platform environment.")
        expected = json.loads((project / "environment/gemm.json").read_text())["cutlass_commit"]
        actual = subprocess.check_output(["git", "-C", state["CUTLASS_PATH"], "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", state["CUTLASS_PATH"], "status", "--porcelain"], text=True).strip()
        if actual != expected or dirty:
            raise ValueError("CUTLASS source does not match the shared commit.")


def check_torch_version(baseline, runtime):
    if baseline.get("schema_version", 3) < 4:
        return
    profile = baseline["development_profile"] if runtime == "development" else baseline["training_profiles"][runtime]
    expected = profile["torch"]
    if importlib.metadata.version("torch") != expected:
        raise ValueError(f"The {runtime} profile requires torch {expected}.")
    if runtime == "nvidia":
        for package, key in (("nvidia-cudnn-cu12", "cudnn"), ("nvidia-cutlass-dsl", "cutlass_dsl"),
                             ("cuda-python", "cuda_python"), ("cuda-bindings", "cuda_bindings")):
            if importlib.metadata.version(package) != profile[key]:
                raise ValueError(f"The nvidia profile requires {package} {profile[key]}.")


def check_cuda_versions(profile, torch_cuda, runtime_cuda):
    major, minor = (int(part) for part in profile["cuda"].split(".")[:2])
    if torch_cuda != f"{major}.{minor}" or (runtime_cuda // 1000, runtime_cuda % 1000 // 10) != (major, minor):
        raise ValueError(f"Expected CUDA {major}.{minor}; found torch CUDA {torch_cuda}, runtime {runtime_cuda}.")


def smoke_test():
    import matplotlib
    import numpy as np
    import pandas as pd
    import scipy.linalg
    import yaml

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    np.testing.assert_array_equal(a @ a, [[7.0, 10.0], [15.0, 22.0]])
    b = np.array([5.0, 11.0])
    np.testing.assert_allclose(a @ scipy.linalg.solve(a, b), b)
    frame = pd.DataFrame({"value": [1, 2]})
    pd.testing.assert_frame_equal(pd.read_csv(io.StringIO(frame.to_csv(index=False))), frame)
    if yaml.safe_load("seed: 7") != {"seed": 7}:
        raise RuntimeError("YAML parsing failed.")
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    if not buffer.getvalue().startswith(b"\x89PNG"):
        raise RuntimeError("Headless figure rendering failed.")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=root)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--accelerator", action="store_true", help="Exercise the selected GPU/NPU")
    args = parser.parse_args()
    try:
        state_file = args.project / ".sdc/environment.json"
        native = state_file.is_file() and os.environ.get("SDC_DEPLOYMENT") != "docker"
        if native:
            verify_platform_environment(args.project, json.loads(state_file.read_text()))
        else:
            verify_baked_files(args.project, Path("/opt/sdc/project"))
        baseline = json.loads((args.project / "environment/baseline.json").read_text())
        release = platform.freedesktop_os_release()
        check_distribution(baseline, release, platform_environment=native)
        uv_version = subprocess.check_output(["uv", "--version"], text=True).split()[1]
        errors = differences(baseline, platform.python_version(), platform.system(), platform.machine(), uv_version)
        if errors:
            raise ValueError("\n".join(errors))
        runtime = os.environ.get("SDC_RUNTIME", "development")
        if runtime not in {"development", "nvidia", "ascend"}:
            raise ValueError(f"Unknown SDC_RUNTIME: {runtime}")
        check_uv_environment(args.project, runtime if baseline.get("schema_version", 3) >= 4 else None)
        check_torch_version(baseline, runtime)
        if args.smoke:
            smoke_test()
        accelerator = accelerator_smoke(runtime, baseline) if args.accelerator else "not checked"
    except (OSError, ValueError, ImportError, RuntimeError, AssertionError, subprocess.CalledProcessError) as exc:
        print(f"Environment verification failed:\n{exc}\nRe-run setup-platform.sh for a platform environment, or rebuild the project Docker image.", file=sys.stderr)
        return 1
    if not args.quiet:
        print(json.dumps({"profile": runtime, "deployment": "platform" if native else "docker", "os": release["ID"] + " " + release["VERSION_ID"], "python": baseline["python"], "uv": uv_version, "packages": len(list(importlib.metadata.distributions())), "smoke_test": "passed" if args.smoke else "not requested", "accelerator": accelerator}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
