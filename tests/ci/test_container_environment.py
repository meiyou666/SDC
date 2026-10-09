import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load("container_policy", ROOT / "scripts/ci/check_environment.py")
runtime = load("container_runtime", ROOT / "environment/verify_environment.py")


class Snapshot:
    def __init__(self, content):
        self.content = content
        self.files = set(content)

    def read(self, path):
        return self.content[path]


def cpu_snapshot(**updates):
    baseline = {
        "schema_version": 2,
        "state": "frozen",
        "profile": "cpu-development",
        "platform": "linux/amd64",
        "python": "3.11.17",
        "base_image": "python:3.11.17-bookworm@sha256:" + "a" * 64,
        "accelerator": {"backend": "ascend", "state": "pending"},
        "import_map": {},
    }
    baseline.update(updates)
    content = {
        policy.BASELINE: json.dumps(baseline),
        policy.LOCKFILE: "numpy==2.4.0 \\\n    --hash=sha256:" + "b" * 64 + "\n",
        "environment/docker/Dockerfile": f"FROM {baseline['base_image']}\nRUN pip install --require-hashes --no-deps\n",
        ".devcontainer/devcontainer.json": json.dumps({"dockerComposeFile": "../compose.yaml", "service": "dev"}),
        "compose.yaml": "services: {}",
        ".dockerignore": "archive/",
        "environment/verify_environment.py": "",
        "environment/docker/entrypoint.sh": "",
    }
    return Snapshot(content)


class ContainerPolicyTests(unittest.TestCase):
    def check(self, state, changed=(), author="member"):
        return policy.check(state, list(changed), author, "owner")

    def test_cpu_code_is_allowed_before_accelerator_is_available(self):
        state = cpu_snapshot()
        state.content["src/example.py"] = "import numpy\n"
        state.files.add("src/example.py")
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
        state = cpu_snapshot()
        state.content["environment/docker/Dockerfile"] = "FROM python:latest\n"
        with self.assertRaisesRegex(policy.PolicyError, "Dockerfile FROM"):
            self.check(state)

    def test_dependency_artifact_hash_is_required(self):
        for value in ["numpy==2.4.0\n", "numpy==2.4.0 --hash=sha256:abcd\n"]:
            state = cpu_snapshot()
            state.content[policy.LOCKFILE] = value
            with self.subTest(value=value), self.assertRaisesRegex(policy.PolicyError, "SHA256"):
                self.check(state)

    def test_devcontainer_cannot_use_a_separate_environment(self):
        state = cpu_snapshot()
        state.content[".devcontainer/devcontainer.json"] = json.dumps({"image": "python:latest"})
        with self.assertRaisesRegex(policy.PolicyError, "shared Compose"):
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
        state = cpu_snapshot()
        state.content["src/example.py"] = "import torch_npu\n"
        state.files.add("src/example.py")
        with self.assertRaisesRegex(policy.PolicyError, "Imports outside"):
            self.check(state)


class RuntimeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads(cpu_snapshot().read(policy.BASELINE))

    def errors(self, installed, python="3.11.17", system="Linux", machine="x86_64"):
        return runtime.differences(self.baseline, {"numpy": "2.4.0"}, installed, python, system, machine)

    def test_matching_runtime_passes(self):
        self.assertEqual(self.errors({"numpy": "2.4.0"}), [])

    def test_version_drift_is_reported(self):
        self.assertTrue(any("expected 2.4.0" in x for x in self.errors({"numpy": "2.3.0"})))

    def test_extra_and_missing_packages_are_reported(self):
        errors = self.errors({"requests": "2.32.0"})
        self.assertTrue(any("not installed" in x for x in errors))
        self.assertTrue(any("Unlocked installed" in x for x in errors))

    def test_python_patch_version_is_checked(self):
        self.assertTrue(any("Python:" in x for x in self.errors({"numpy": "2.4.0"}, python="3.11.5")))

    def test_architecture_is_checked(self):
        self.assertTrue(any("Platform:" in x for x in self.errors({"numpy": "2.4.0"}, machine="aarch64")))

    def test_hashed_lock_parsing_and_distribution_names(self):
        text = "# generated\nPyYAML==6.0.3 \\\n    --hash=sha256:" + "a" * 64 + "\nnumpy==2.4.0\n"
        self.assertEqual(runtime.locked_versions(text), {"pyyaml": "6.0.3", "numpy": "2.4.0"})

    def test_changed_base_image_manifest_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baked = root / "baked"
            (baked / "dependencies").mkdir(parents=True)
            (baked / "baseline.json").write_text('{"base_image": "old"}')
            (baked / "dependencies/requirements.lock").write_text("numpy==2.4.0\n")
            current = root / "baseline.json"
            current.write_text('{"base_image": "new"}')
            with self.assertRaisesRegex(ValueError, "files used to build"):
                runtime.verify_baked_files(current, baked / "dependencies/requirements.lock", baked)


if __name__ == "__main__":
    unittest.main()
