#!/usr/bin/env python3
"""Check the declared environment and visible Python imports without running experiments."""

import argparse
import ast
import json
import os
from pathlib import PurePosixPath
import re
import subprocess
import sys


BASELINE = "environment/baseline.json"
LOCKFILE = "environment/dependencies/requirements.lock"
INFRA = (".github/", "scripts/ci/", "tests/ci/")
SHARED_FILES = {"compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml", ".dockerignore", ".gitattributes", "AGENTS.md"}
CODE_SUFFIXES = {
    ".py", ".pyi", ".ipynb", ".sh", ".bash", ".bat", ".ps1",
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".f", ".f90",
}


class PolicyError(Exception):
    pass


class GitSnapshot:
    def __init__(self, ref):
        self.ref = ref
        raw = subprocess.check_output(["git", "ls-tree", "-rz", ref])
        self.files = {
            item.split(b"\t", 1)[1].decode("utf-8")
            for item in raw.split(b"\0") if item
        }

    def read(self, path):
        return subprocess.check_output(
            ["git", "show", f"{self.ref}:{path}"], stderr=subprocess.PIPE
        ).decode("utf-8")


def package_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def dependency_file(path):
    name = PurePosixPath(path).name.lower()
    return (
        name.startswith("dockerfile") or name.endswith(".dockerfile")
        or (name.startswith("requirements") and name.endswith((".txt", ".in", ".lock")))
        or name in {"pyproject.toml", "setup.py", "setup.cfg", "pipfile", "pipfile.lock", "poetry.lock", "uv.lock", "pdm.lock"}
        or (name.startswith(("environment", "conda")) and name.endswith((".yml", ".yaml")))
        or name in {"compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml", "devcontainer.json"}
    )


def experiment_code(path):
    return (
        not path.startswith(("archive/", "docs/", "environment/", *INFRA))
        and PurePosixPath(path).suffix.lower() in CODE_SUFFIXES
    )


def parse_lock(text):
    packages = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip().rstrip("\\").strip()
        if not line or line.startswith("--hash="):
            continue
        if line.startswith(("--index-url ", "--extra-index-url ", "--find-links ")):
            continue
        line = re.sub(r"\s+--hash=[A-Za-z0-9]+:[0-9a-fA-F]+", "", line).strip()
        match = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[A-Za-z0-9,._-]+\])?==([0-9][A-Za-z0-9.!+_-]*)",
            line,
        )
        if not match:
            raise PolicyError(f"{LOCKFILE}:{number}: use a resolved package==exact-version entry: {line}")
        name, version = package_name(match[1]), match[2]
        if name in packages and packages[name] != version:
            raise PolicyError(f"Conflicting locked versions for {name}.")
        packages[name] = version
    return packages


def imports_in(source, path):
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise PolicyError(f"Cannot parse {path}:{exc.lineno}: {exc.msg}") from exc
    imports, dynamic = set(), []
    importlib_names = {"importlib"}
    loader_names = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            importlib_names.update(alias.asname or alias.name for alias in node.names if alias.name == "importlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib" and node.level == 0:
            loader_names.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            target = node.func
            loader = (
                isinstance(target, ast.Name) and target.id in loader_names
            ) or (
                isinstance(target, ast.Attribute) and target.attr == "import_module"
                and isinstance(target.value, ast.Name) and target.value.id in importlib_names
            )
            if loader:
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    name = node.args[0].value
                    if not name.startswith("."):
                        imports.add(name.split(".", 1)[0])
                else:
                    dynamic.append(f"{path}:{node.lineno}")
    return imports, dynamic


def check_cpu_container(snapshot, baseline, locked):
    if baseline.get("profile") != "cpu-development" or baseline.get("platform") != "linux/amd64":
        raise PolicyError("Use the shared cpu-development profile on linux/amd64.")
    image = baseline.get("base_image", "")
    if not re.fullmatch(r"\S+@sha256:[0-9a-fA-F]{64}", image):
        raise PolicyError("The base image must include an immutable @sha256 digest.")
    accelerator = baseline.get("accelerator", {})
    if not isinstance(accelerator, dict) or accelerator.get("state") != "pending" or accelerator.get("backend") != "ascend":
        raise PolicyError("The CPU development profile must keep the Ascend platform pending.")
    if any(accelerator.get(key) is not None for key in ("device", "driver", "firmware", "cann", "pytorch", "torch_npu")):
        raise PolicyError("Do not declare unverified accelerator versions in the CPU development profile.")

    dockerfile = "environment/docker/Dockerfile"
    devcontainer = ".devcontainer/devcontainer.json"
    for path in (dockerfile, devcontainer, "compose.yaml", ".dockerignore", "environment/verify_environment.py", "environment/docker/entrypoint.sh"):
        if path not in snapshot.files:
            raise PolicyError(f"Missing shared container file: {path}.")
    source = snapshot.read(dockerfile)
    if re.findall(r"(?im)^FROM\s+(\S+)\s*$", source) != [image]:
        raise PolicyError("Dockerfile FROM must match the baseline's immutable base_image.")
    if "--require-hashes" not in source or "--no-deps" not in source:
        raise PolicyError("Install the resolved lock with --require-hashes and --no-deps.")
    try:
        dev = json.loads(snapshot.read(devcontainer))
    except ValueError as exc:
        raise PolicyError(f"Invalid Dev Container configuration: {exc}") from exc
    if not isinstance(dev, dict) or dev.get("dockerComposeFile") != "../compose.yaml" or dev.get("service") != "dev":
        raise PolicyError("Dev Container must use the shared Compose dev service.")

    hashes, current = set(), None
    for raw in snapshot.read(LOCKFILE).splitlines():
        line = raw.split("#", 1)[0].strip()
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)==", line)
        if match:
            current = package_name(match[1])
        if current and re.search(r"--hash=sha256:[0-9a-fA-F]{64}(?:\s|$)", line):
            hashes.add(current)
    missing = sorted(set(locked) - hashes)
    if missing:
        raise PolicyError("Every locked package needs a SHA256 artifact hash: " + ", ".join(missing))


def check(snapshot, changed, author, owner):
    owner_change = bool(owner) and author.casefold() == owner.casefold()
    protected = [
        path for path in changed
        if not path.startswith("archive/") and (
            path.startswith(("environment/", "configs/training/", ".devcontainer/"))
            or path in SHARED_FILES or dependency_file(path)
        )
    ]
    if protected and not owner_change:
        raise PolicyError("Only the repository owner may change the shared environment or training baseline: " + ", ".join(sorted(protected)))

    if BASELINE not in snapshot.files:
        raise PolicyError(f"Missing {BASELINE}; ask the maintainer to configure the shared baseline.")
    try:
        baseline = json.loads(snapshot.read(BASELINE))
    except (ValueError, UnicodeError) as exc:
        raise PolicyError(f"Invalid {BASELINE}: {exc}") from exc
    if not isinstance(baseline, dict) or baseline.get("schema_version") not in {1, 2}:
        raise PolicyError("Unsupported environment baseline schema.")
    state = baseline.get("state")
    code = sorted(path for path in snapshot.files if experiment_code(path))
    if state == "pending":
        if code:
            raise PolicyError("The shared environment is not frozen. Experimental code must wait for the owner to configure it: " + ", ".join(code[:10]))
        return {"state": "pending", "message": "Structure/documentation changes only; no experimental environment has been frozen.", "python_files": 0, "dynamic_imports": []}
    if state != "frozen":
        raise PolicyError("Environment state must be pending or frozen.")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[A-Za-z0-9.+-]+)?", baseline.get("python", "")):
        raise PolicyError("Set the exact Python version, including the patch version.")
    if baseline["schema_version"] == 1:
        if not re.fullmatch(r"\S+@sha256:[0-9a-fA-F]{64}", baseline.get("image", "")):
            raise PolicyError("The frozen image must include an immutable @sha256 digest.")
        if not re.fullmatch(r"\d+[A-Za-z0-9_.+-]*", baseline.get("cann", "")):
            raise PolicyError("Set an explicit CANN version.")
    if LOCKFILE not in snapshot.files:
        raise PolicyError(f"Missing {LOCKFILE}.")
    locked = parse_lock(snapshot.read(LOCKFILE))
    if not locked:
        raise PolicyError("The frozen dependency lock is empty.")
    if baseline["schema_version"] == 2:
        check_cpu_container(snapshot, baseline, locked)

    extra = sorted(
        path for path in snapshot.files
        if not path.startswith(("archive/", "environment/", *INFRA))
        and path not in {"compose.yaml", ".devcontainer/devcontainer.json"}
        and dependency_file(path)
    )
    if extra:
        raise PolicyError("Use the shared environment instead of separate dependency/Docker files: " + ", ".join(extra))
    aliases = baseline.get("import_map", {})
    if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in aliases.items()):
        raise PolicyError("import_map must map Python module names to locked package names.")
    for module, distribution in aliases.items():
        if package_name(distribution) not in locked:
            raise PolicyError(f"import_map entry {module} points to unlocked package {distribution}.")

    local_modules = set()
    for path in code:
        p = PurePosixPath(path)
        if p.suffix == ".py":
            local_modules.add(p.stem)
            local_modules.add(p.parts[0])
            if p.parts[0] == "src" and len(p.parts) > 1:
                local_modules.add(p.parts[1].removesuffix(".py"))
    standard = set(sys.stdlib_module_names) | set(sys.builtin_module_names) | {"__future__"}
    missing, dynamic, count = [], [], 0
    for path in code:
        if not path.endswith(".py"):
            continue
        count += 1
        imports, unresolved = imports_in(snapshot.read(path), path)
        dynamic.extend(unresolved)
        for module in sorted(imports):
            if module in standard or module in local_modules:
                continue
            distribution = aliases.get(module, module)
            if package_name(distribution) not in locked:
                missing.append(f"{path}: {module}")
    if missing:
        raise PolicyError("Imports outside the shared dependency lock: " + "; ".join(missing))
    return {"state": "frozen", "message": "Declared environment and visible Python imports match the shared baseline.", "python_files": count, "dynamic_imports": dynamic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--author", required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    base = args.base
    if not base or set(base) == {"0"}:
        base = subprocess.check_output(["git", "rev-parse", f"{args.head}^"], text=True).strip()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, args.head]
    ).decode("utf-8").split("\0")
    try:
        result = check(GitSnapshot(args.head), [p for p in changed if p], args.author, args.owner)
    except PolicyError as exc:
        print(f"Environment policy failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### Environment consistency\n\nState: **{result['state']}**\n\n{result['message']}\n")
            if result["dynamic_imports"]:
                f.write("\nDynamic imports need manual confirmation:\n" + "\n".join(f"- `{p}`" for p in result["dynamic_imports"]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
