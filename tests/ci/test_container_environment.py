import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from test_environment_policy import policy, snapshot, uv_lock


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("container_runtime", ROOT / "environment/verify_environment.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def cpu_snapshot(**updates):
    state = snapshot(**updates)
    state.content[policy.LOCKFILE] = uv_lock([("numpy", "2.4.0")])
    return state


class ContainerPolicyTests(unittest.TestCase):
    def check(self, state, changed=(), author="member"):
        return policy.check(state, list(changed), author, "owner")

    def test_cpu_code_is_allowed_before_accelerator_is_available(self):
        state = cpu_snapshot(extra={"src/example.py": "import numpy\n"})
        self.assertEqual(self.check(state)["python_files"], 1)

    def test_member_cannot_change_shared_container_or_agent_rules(self):
        for path in ["compose.yaml", ".devcontainer/devcontainer.json", ".dockerignore", "AGENTS.md"]:
            with self.subTest(path=path), self.assertRaisesRegex(policy.PolicyError, "Only the repository owner"):
                self.check(cpu_snapshot(), [path])

    def test_owner_can_update_container(self):
        self.assertEqual(self.check(cpu_snapshot(), ["compose.yaml"], "owner")["state"], "frozen")

    def test_no_placeholder_accelerator_versions(self):
        state = cpu_snapshot(accelerator={"backend": "ascend", "state": "pending", "cann": "9.0.0"})
        with self.assertRaisesRegex(policy.PolicyError, "unverified accelerator"):
            self.check(state)

    def test_cpu_profile_cannot_claim_accelerator_is_frozen(self):
        with self.assertRaisesRegex(policy.PolicyError, "Ascend platform pending"):
            self.check(cpu_snapshot(accelerator={"backend": "ascend", "state": "frozen"}))

    def test_docker_base_must_match_manifest(self):
        state = cpu_snapshot(extra={"environment/docker/Dockerfile": "FROM python:latest\n"})
        with self.assertRaisesRegex(policy.PolicyError, "Dockerfile FROM"):
            self.check(state)

    def test_dependency_artifact_hash_is_required(self):
        state = cpu_snapshot()
        state.content[policy.LOCKFILE] = state.content[policy.LOCKFILE].replace("b" * 64, "abcd")
        with self.assertRaisesRegex(policy.PolicyError, "SHA256"):
            self.check(state)

    def test_devcontainer_cannot_use_a_separate_environment(self):
        state = cpu_snapshot(extra={".devcontainer/devcontainer.json": json.dumps({"image": "python:latest"})})
        with self.assertRaisesRegex(policy.PolicyError, "shared Dockerfile"):
            self.check(state)

    def test_devcontainer_cannot_change_dockerfile_context_or_target(self):
        for key, value in [
            ("dockerfile", "../experiments/Dockerfile"),
            ("context", "../experiments"),
            ("target", "tooling"),
        ]:
            with self.subTest(key=key):
                state = cpu_snapshot()
                config = json.loads(state.read(".devcontainer/devcontainer.json"))
                config["build"][key] = value
                state.content[".devcontainer/devcontainer.json"] = json.dumps(config)
                with self.assertRaisesRegex(policy.PolicyError, "shared Dockerfile"):
                    self.check(state)

    def test_missing_runtime_checker_is_rejected(self):
        state = cpu_snapshot()
        state.files.remove("environment/verify_environment.py")
        with self.assertRaisesRegex(policy.PolicyError, "Missing shared container"):
            self.check(state)

    def test_member_cannot_add_alternative_compose_file(self):
        with self.assertRaisesRegex(policy.PolicyError, "Only the repository owner"):
            self.check(cpu_snapshot(), ["experiments/compose.yaml"])

    def test_import_of_unconfigured_npu_library_is_rejected(self):
        state = cpu_snapshot(extra={"src/example.py": "import torch_npu\n"})
        with self.assertRaisesRegex(policy.PolicyError, "Imports outside"):
            self.check(state)


class RuntimeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads(cpu_snapshot().read(policy.BASELINE))

    def errors(self, python="3.11.17", system="Linux", machine="x86_64", uv="0.12.24"):
        return runtime.differences(self.baseline, python, system, machine, uv)

    def test_matching_runtime_passes(self):
        self.assertEqual(self.errors(), [])

    def test_python_patch_version_is_checked(self):
        self.assertTrue(any("Python:" in x for x in self.errors(python="3.11.5")))

    def test_architecture_is_checked(self):
        self.assertTrue(any("Platform:" in x for x in self.errors(machine="aarch64")))

    def test_uv_version_is_checked(self):
        self.assertTrue(any("uv:" in x for x in self.errors(uv="0.12.23")))

    def test_native_uv_failure_is_propagated(self):
        result = subprocess.CompletedProcess([], 1, "", "Environment is not synchronized")
        with mock.patch.object(runtime.subprocess, "run", return_value=result) as run:
            with self.assertRaisesRegex(ValueError, "not synchronized"):
                runtime.check_uv_environment(Path("/project"))
            self.assertIn("--locked", run.call_args.args[0])
            self.assertIn("--check", run.call_args.args[0])

    def test_changed_image_or_lock_requires_rebuild(self):
        for changed in ["environment/baseline.json", "uv.lock", "pyproject.toml", ".python-version"]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                roots = [Path(directory) / "current", Path(directory) / "baked"]
                for root in roots:
                    (root / "environment").mkdir(parents=True)
                    for name in ["environment/baseline.json", "uv.lock", "pyproject.toml", ".python-version"]:
                        (root / name).write_text("same\n")
                (roots[0] / changed).write_text("changed\n")
                with self.assertRaisesRegex(ValueError, "files used to build"):
                    runtime.verify_baked_files(*roots)

    def test_uv_rejects_stale_project_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "pyproject.toml"
            project.write_text('[project]\nname = "probe"\nversion = "0.1.0"\nrequires-python = ">=3.11,<3.12"\ndependencies = []\n[tool.uv]\npackage = false\n')
            env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(root / ".venv"))
            command = ["uv", "lock", "--offline", "--project", str(root)]
            subprocess.run(command, env=env, check=True, capture_output=True, text=True)
            subprocess.run(command + ["--check"], env=env, check=True, capture_output=True, text=True)
            project.write_text(project.read_text().replace("dependencies = []", 'dependencies = ["numpy==2.4.6"]'))
            result = subprocess.run(command + ["--check"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
