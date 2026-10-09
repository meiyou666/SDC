#!/usr/bin/env python3
"""Verify the running development environment and optionally exercise its libraries."""

import argparse
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import re
import sys


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(text):
    packages = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)", line)
        if match:
            name, version = match.groups()
            packages[normalized(name)] = version
    if not packages:
        raise ValueError("The dependency lock is empty.")
    return packages


def verify_baked_files(baseline_path, lock_path, baked_root):
    for supplied, baked in ((baseline_path, baked_root / "baseline.json"), (lock_path, baked_root / "dependencies/requirements.lock")):
        if supplied.resolve() != baked.resolve() and supplied.read_bytes() != baked.read_bytes():
            raise ValueError(f"{supplied.name} differs from the files used to build this image.")


def differences(baseline, expected, installed, python_version, system, machine):
    errors = []
    if baseline.get("state") != "frozen" or baseline.get("profile") != "cpu-development":
        errors.append("Expected the frozen cpu-development profile.")
    if python_version != baseline.get("python"):
        errors.append(f"Python: expected {baseline.get('python')}, found {python_version}.")
    if baseline.get("platform") != "linux/amd64" or system != "Linux" or machine not in {"x86_64", "AMD64"}:
        errors.append(f"Platform: expected linux/amd64, found {system}/{machine}.")
    for name, version in expected.items():
        if installed.get(name) != version:
            errors.append(f"{name}: expected {version}, found {installed.get(name, 'not installed')}.")
    for name in sorted(set(installed) - set(expected)):
        errors.append(f"Unlocked installed dependency: {name}=={installed[name]}.")
    return errors


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
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=root / "baseline.json")
    parser.add_argument("--lock", type=Path, default=root / "dependencies/requirements.lock")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        verify_baked_files(args.baseline, args.lock, root)
        baseline = json.loads(args.baseline.read_text())
        expected = locked_versions(args.lock.read_text())
        installed = {
            normalized(dist.metadata["Name"]): dist.version
            for dist in importlib.metadata.distributions()
        }
        errors = differences(baseline, expected, installed, platform.python_version(), platform.system(), platform.machine())
        if errors:
            raise ValueError("\n".join(errors))
        if args.smoke:
            smoke_test()
    except (OSError, ValueError, ImportError, RuntimeError, AssertionError) as exc:
        print(f"Environment verification failed:\n{exc}\nUse the shared container; rebuild it after environment updates.", file=sys.stderr)
        return 1
    if not args.quiet:
        print(json.dumps({"profile": baseline["profile"], "python": baseline["python"], "packages": len(expected), "smoke_test": "passed" if args.smoke else "not requested", "accelerator": "not configured"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
