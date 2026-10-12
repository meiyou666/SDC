import json
from pathlib import Path
import subprocess
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
import pytest
import yaml

from test_environment_policy import policy, uv_lock


ROOT = Path(__file__).resolve().parents[2]


def exported(extra, machine):
    result = subprocess.run(
        ["uv", "export", "--locked", "--offline", "--extra", extra, "--no-hashes", "--format", "requirements-txt"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    marker_env = {**default_environment(), "platform_machine": machine, "sys_platform": "linux"}
    packages = {}
    for line in result.stdout.splitlines():
        if not line.strip() or line.startswith(("#", "-", " ")):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate(marker_env):
            versions = [specifier.version for specifier in requirement.specifier if specifier.operator == "=="]
            assert len(versions) == 1, line
            packages[requirement.name] = versions[0]
    return packages


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_accelerator_profiles_have_identical_common_packages(machine):
    gpu, npu = exported("nvidia", machine), exported("ascend", machine)
    assert gpu["torch"] == "2.7.1+cu128"
    assert npu["torch"] == "2.7.1+cpu"
    assert npu["torch-npu"] == "2.7.1.post8"
    for name in set(gpu) | set(npu):
        if name in {"torch", "torch-npu", "triton"} or name.startswith("nvidia-"):
            continue
        assert gpu.get(name) == npu.get(name), name


def test_only_hardware_builds_may_differ_in_lock():
    assert policy.parse_lock(uv_lock([("torch", "2.7.1+cpu"), ("torch", "2.7.1+cu128")]))["torch"] == "2.7.1"
    with pytest.raises(policy.PolicyError, match="Conflicting"):
        policy.parse_lock(uv_lock([("torch", "2.7.1+cpu"), ("torch", "2.8.0+cu128")]))
    with pytest.raises(policy.PolicyError, match="Conflicting"):
        policy.parse_lock(uv_lock([("numpy", "2.1.0"), ("numpy", "2.2.0")]))


def test_compose_and_devcontainers_use_the_shared_build():
    services = yaml.safe_load((ROOT / "compose.yaml").read_text())["services"]
    for profile in ("nvidia", "ascend"):
        assert services[profile]["build"]["dockerfile"] == "environment/docker/Dockerfile"
        assert services[profile]["build"]["target"] == profile
        dev = json.loads((ROOT / f".devcontainer/{profile}/devcontainer.json").read_text())
        assert dev["build"]["target"] == profile
        assert dev["build"]["dockerfile"] == "../../environment/docker/Dockerfile"
    assert services["nvidia"]["gpus"] == "all"
    assert "/dev/davinci0:/dev/davinci0" in services["ascend"]["devices"]
    assert not services["ascend"].get("privileged", False)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    conflict = project["tool"]["uv"]["conflicts"][0]
    assert {entry["extra"] for entry in conflict} == {"development", "nvidia", "ascend"}
