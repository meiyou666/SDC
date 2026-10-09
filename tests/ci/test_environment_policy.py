import importlib.util
import json
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[2] / "scripts/ci/check_environment.py"
spec = importlib.util.spec_from_file_location("environment_policy", path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class Snapshot:
    def __init__(self, files):
        self.content = files
        self.files = set(files)

    def read(self, path):
        return self.content[path]


def uv_lock(packages=None):
    packages = packages if packages is not None else [("numpy", "2.4.0"), ("torch-npu", "2.5.1"), ("PyYAML", "6.0.3")]
    text = 'version = 1\n[[package]]\nname = "sdc"\nversion = "0.1.0"\nsource = { virtual = "." }\n'
    for name, version in packages:
        text += f'\n[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ registry = "https://pypi.org/simple" }}\nwheels = [{{ url = "https://example.invalid/pkg.whl", hash = "sha256:{"b" * 64}" }}]\n'
    return text


def snapshot(state="frozen", extra=None, **baseline_changes):
    baseline = {
        "schema_version": 3,
        "state": state,
        "profile": "cpu-development",
        "platform": "linux/amd64",
        "base_image": "python:3.11.17-bookworm@sha256:" + "a" * 64,
        "uv_image": "ghcr.io/astral-sh/uv:0.12.24@sha256:" + "c" * 64,
        "python": "3.11.17",
        "uv": "0.12.24",
        "accelerator": {"backend": "ascend", "state": "pending"},
        "import_map": {},
    }
    baseline.update(baseline_changes)
    files = {
        policy.BASELINE: json.dumps(baseline),
        policy.LOCKFILE: uv_lock(),
        policy.PROJECT: '[project]\nname = "sdc"\nversion = "0.1.0"\nrequires-python = ">=3.11,<3.12"\n[tool.uv]\npackage = false\nrequired-version = "==' + baseline['uv'] + '"\n',
        ".python-version": baseline["python"] + "\n",
        "environment/docker/Dockerfile": f"FROM {baseline['uv_image']} AS uv\nFROM {baseline['base_image']} AS tooling\nFROM tooling AS development\nRUN uv sync --locked --all-groups\n",
        ".devcontainer/devcontainer.json": json.dumps({"dockerComposeFile": "../compose.yaml", "service": "dev"}),
        "compose.yaml": "services: {}",
        ".dockerignore": "archive/",
        "environment/verify_environment.py": "",
        "environment/docker/entrypoint.sh": "",
        "dev.cmd": "",
        "dev.sh": "",
    }
    files.update(extra or {})
    return Snapshot(files)


class EnvironmentPolicyTests(unittest.TestCase):
    def run_check(self, state, changed=(), author="member"):
        return policy.check(state, list(changed), author, "owner")

    def test_pending_allows_docs_and_ci_infrastructure(self):
        state = snapshot("pending", {"docs/note.md": "note", "scripts/ci/check.py": "import anything", "archive/old.py": "import anything"})
        self.assertEqual(self.run_check(state, ["docs/note.md"])["state"], "pending")

    def test_pending_blocks_python_experiments(self):
        with self.assertRaisesRegex(policy.PolicyError, "not frozen"):
            self.run_check(snapshot("pending", {"src/example.py": "pass"}))

    def test_pending_blocks_native_and_shell_experiments(self):
        for name in ["src/op.cpp", "src/op.cu", "scripts/run.sh", "experiments/probe.ipynb"]:
            with self.subTest(name=name), self.assertRaisesRegex(policy.PolicyError, "not frozen"):
                self.run_check(snapshot("pending", {name: "content"}))

    def test_member_cannot_change_environment_even_with_frozen_head(self):
        with self.assertRaisesRegex(policy.PolicyError, "Only the repository owner"):
            self.run_check(snapshot(), [policy.BASELINE])

    def test_member_cannot_delete_lock_or_change_training_baseline(self):
        for name in [policy.LOCKFILE, "configs/training/default.yaml", "requirements-extra.txt", "experiments/Dockerfile"]:
            with self.subTest(name=name), self.assertRaisesRegex(policy.PolicyError, "Only the repository owner"):
                self.run_check(snapshot(), [name])

    def test_owner_can_prepare_pending_environment(self):
        self.assertEqual(self.run_check(snapshot("pending"), [policy.BASELINE], "owner")["state"], "pending")

    def test_owner_can_freeze_valid_environment(self):
        self.assertEqual(self.run_check(snapshot(), [policy.BASELINE, policy.LOCKFILE], "owner")["state"], "frozen")

    def test_owner_does_not_skip_validity_checks(self):
        with self.assertRaisesRegex(policy.PolicyError, "immutable"):
            self.run_check(snapshot(base_image="example.invalid/sdc:latest"), [policy.BASELINE], "owner")

    def test_missing_or_invalid_baseline_fails(self):
        for files in [{}, {policy.BASELINE: "bad json"}, {policy.BASELINE: "[]"}]:
            with self.subTest(files=files), self.assertRaises(policy.PolicyError):
                self.run_check(Snapshot(files))

    def test_versions_are_explicit(self):
        for updates in [{"python": "3.11"}, {"uv": "latest"}, {"state": "disabled"}]:
            with self.subTest(updates=updates), self.assertRaises(policy.PolicyError):
                self.run_check(snapshot(**updates))

    def test_unpinned_conflicting_or_empty_locks_fail(self):
        for lock in ["numpy>=2", uv_lock([("numpy", "2.*")]), uv_lock([("numpy", "2.0"), ("numpy", "2.1")]), uv_lock([])]:
            with self.subTest(lock=lock), self.assertRaises(policy.PolicyError):
                self.run_check(snapshot(extra={policy.LOCKFILE: lock}))

    def test_native_lock_skips_virtual_project_and_normalizes_names(self):
        self.assertEqual(policy.parse_lock(uv_lock()), {"numpy": "2.4.0", "torch-npu": "2.5.1", "pyyaml": "6.0.3"})

    def test_root_uv_files_are_owner_managed(self):
        for name in ["uv.lock", "pyproject.toml", ".python-version", "dev.cmd", "dev.sh"]:
            with self.subTest(name=name), self.assertRaisesRegex(policy.PolicyError, "Only the repository owner"):
                self.run_check(snapshot(), [name])

    def test_python_pin_must_match_baseline(self):
        with self.assertRaisesRegex(policy.PolicyError, ".python-version"):
            self.run_check(snapshot(extra={".python-version": "3.12.0\n"}))

    def test_uv_version_pin_must_match_baseline(self):
        state = snapshot()
        state.content[policy.PROJECT] = state.content[policy.PROJECT].replace("0.12.24", "0.12.23")
        with self.assertRaisesRegex(policy.PolicyError, "required-version"):
            self.run_check(state)

    def test_stdlib_locked_and_local_imports_pass(self):
        state = snapshot(extra={"src/training/__init__.py": "", "src/main.py": "import json\nimport numpy\nimport torch_npu\nfrom training import core\n"})
        self.assertEqual(self.run_check(state)["python_files"], 2)

    def test_unlocked_import_fails(self):
        with self.assertRaisesRegex(policy.PolicyError, "Imports outside"):
            self.run_check(snapshot(extra={"src/main.py": "import requests"}))

    def test_distribution_alias_is_checked_against_lock(self):
        state = snapshot(extra={"src/main.py": "import yaml"}, import_map={"yaml": "PyYAML"})
        self.run_check(state)
        with self.assertRaisesRegex(policy.PolicyError, "unlocked package"):
            self.run_check(snapshot(import_map={"yaml": "not-installed"}))

    def test_constant_dynamic_import_is_checked(self):
        with self.assertRaisesRegex(policy.PolicyError, "Imports outside"):
            self.run_check(snapshot(extra={"src/main.py": "import importlib\nimportlib.import_module('requests')"}))

    def test_variable_dynamic_import_is_reported(self):
        result = self.run_check(snapshot(extra={"src/main.py": "import importlib\nimportlib.import_module(module_name)"}))
        self.assertEqual(result["dynamic_imports"], ["src/main.py:2"])

    def test_aliased_dynamic_imports_are_checked(self):
        for code in ["import importlib as il\nil.import_module('requests')", "from importlib import import_module as load\nload('requests')"]:
            with self.subTest(code=code), self.assertRaisesRegex(policy.PolicyError, "Imports outside"):
                self.run_check(snapshot(extra={"src/main.py": code}))

    def test_extra_dependency_files_are_rejected(self):
        with self.assertRaisesRegex(policy.PolicyError, "separate dependency"):
            self.run_check(snapshot(extra={"experiments/requirements.txt": "requests==2.32.0"}))

    def test_archived_imports_and_dependency_files_are_ignored(self):
        self.run_check(snapshot(extra={"archive/old.py": "invalid python", "archive/Dockerfile": "FROM old"}))

    def test_syntax_error_does_not_silently_skip_imports(self):
        with self.assertRaisesRegex(policy.PolicyError, "Cannot parse"):
            self.run_check(snapshot(extra={"src/main.py": "import ("}))


if __name__ == "__main__":
    unittest.main()
