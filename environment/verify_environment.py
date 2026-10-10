#!/usr/bin/env python3
"""Verify the running development environment and optionally exercise its libraries."""

import argparse
import importlib.metadata
import io
import json
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
    if baseline.get("state") != "frozen" or baseline.get("profile") != "cpu-development":
        errors.append("Expected the frozen cpu-development profile.")
    if python_version != baseline.get("python"):
        errors.append(f"Python: expected {baseline.get('python')}, found {python_version}.")
    if baseline.get("platform") != "linux/amd64" or system != "Linux" or machine not in {"x86_64", "AMD64"}:
        errors.append(f"Platform: expected linux/amd64, found {system}/{machine}.")
    if uv_version != baseline.get("uv"):
        errors.append(f"uv: expected {baseline.get('uv')}, found {uv_version}.")
    return errors


def check_uv_environment(project):
    result = subprocess.run(
        ["uv", "sync", "--locked", "--check", "--offline", "--all-groups", "--project", str(project)],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "uv environment check failed.")


def check_distribution(baseline, release):
    expected = (baseline.get("os_id"), baseline.get("os_version"))
    actual = (release.get("ID"), release.get("VERSION_ID"))
    if actual != expected:
        raise ValueError(f"Distribution: expected {expected[0]} {expected[1]}, found {actual[0]} {actual[1]}.")


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
    args = parser.parse_args()
    try:
        verify_baked_files(args.project, root)
        baseline = json.loads((args.project / "environment/baseline.json").read_text())
        check_distribution(baseline, platform.freedesktop_os_release())
        uv_version = subprocess.check_output(["uv", "--version"], text=True).split()[1]
        errors = differences(baseline, platform.python_version(), platform.system(), platform.machine(), uv_version)
        if errors:
            raise ValueError("\n".join(errors))
        check_uv_environment(args.project)
        if args.smoke:
            smoke_test()
    except (OSError, ValueError, ImportError, RuntimeError, AssertionError, subprocess.CalledProcessError) as exc:
        print(f"Environment verification failed:\n{exc}\nUse the shared container; rebuild it after environment updates.", file=sys.stderr)
        return 1
    if not args.quiet:
        print(json.dumps({"profile": baseline["profile"], "os": baseline["os_id"] + " " + baseline["os_version"], "python": baseline["python"], "uv": uv_version, "packages": len(list(importlib.metadata.distributions())), "smoke_test": "passed" if args.smoke else "not requested", "accelerator": "not configured"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
