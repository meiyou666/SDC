import json
from pathlib import Path
import subprocess
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
import pytest
import yaml

from test_environment_policy import Snapshot, policy, uv_lock


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def training_snapshot():
    files = [
        policy.BASELINE, policy.PROJECT, policy.LOCKFILE, ".python-version", "compose.yaml", ".dockerignore", "environment/gemm.json",
        "environment/docker/Dockerfile", "environment/docker/install-python.sh",
        "environment/docker/entrypoint.sh", "environment/verify_environment.py", "dev.cmd", "dev.sh",
        ".devcontainer/devcontainer.json", *policy.TRAINING_DEVCONTAINERS,
    ]
    return Snapshot({path: (ROOT / path).read_text() for path in files})


def check_snapshot(snapshot):
    return policy.check(snapshot, [], "member", "owner")


def change_baseline(snapshot, change):
    baseline = json.loads(snapshot.read(policy.BASELINE))
    change(baseline)
    snapshot.content[policy.BASELINE] = json.dumps(baseline)


def test_schema4_shared_configuration_passes(training_snapshot):
    assert json.loads(training_snapshot.read(policy.BASELINE))["schema_version"] == 4
    assert check_snapshot(training_snapshot)["state"] == "frozen"


@pytest.mark.parametrize("path", policy.TRAINING_DEVCONTAINERS)
def test_schema4_requires_training_devcontainers(training_snapshot, path):
    training_snapshot.files.remove(path)
    with pytest.raises(policy.PolicyError, match="Missing"):
        check_snapshot(training_snapshot)


@pytest.mark.parametrize("path", policy.TRAINING_DEVCONTAINERS)
@pytest.mark.parametrize("change", ["image", "dockerfile", "context", "target"])
def test_training_devcontainers_reject_separate_environments(training_snapshot, path, change):
    config = json.loads(training_snapshot.read(path))
    if change == "image":
        config.pop("build")
        config["image"] = "python:latest"
    else:
        config["build"][change] = {
            "dockerfile": "../../experiments/Dockerfile",
            "context": "../../experiments",
            "target": "development",
        }[change]
    training_snapshot.content[path] = json.dumps(config)
    with pytest.raises(policy.PolicyError, match="shared Dockerfile"):
        check_snapshot(training_snapshot)


@pytest.mark.parametrize("profile", ["nvidia", "ascend"])
def test_schema4_requires_both_training_profiles(training_snapshot, profile):
    change_baseline(training_snapshot, lambda b: b["training_profiles"].pop(profile))
    with pytest.raises(policy.PolicyError, match="NVIDIA and Ascend"):
        check_snapshot(training_snapshot)


def test_schema4_requires_development_profile(training_snapshot):
    change_baseline(training_snapshot, lambda b: b.pop("development_profile"))
    with pytest.raises(policy.PolicyError, match="development profile"):
        check_snapshot(training_snapshot)


def test_schema4_rejects_early_cann_selector(training_snapshot):
    filename = "environment/docker/Dockerfile"
    source = training_snapshot.read(filename)
    selector = "FROM cann_${ASCEND_SOC} AS cann_sdk\n"
    training_snapshot.content[filename] = selector + source.replace(selector, "")
    with pytest.raises(policy.PolicyError, match="before selecting"):
        check_snapshot(training_snapshot)


@pytest.mark.parametrize("sdk", ["nvidia", "910b", "a3", "950"])
def test_schema4_requires_sdk_digests(training_snapshot, sdk):
    def unpin(baseline):
        if sdk == "nvidia":
            baseline["training_profiles"][sdk]["sdk_image"] = "nvidia/cuda:latest"
        else:
            baseline["training_profiles"]["ascend"]["sdk_images"][sdk] = "ascendai/cann:latest"
    change_baseline(training_snapshot, unpin)
    with pytest.raises(policy.PolicyError, match="SDK image by digest"):
        check_snapshot(training_snapshot)


def test_schema4_rejects_incomplete_cann_image_set(training_snapshot):
    change_baseline(training_snapshot, lambda b: b["training_profiles"]["ascend"]["sdk_images"].pop("950"))
    with pytest.raises(policy.PolicyError, match="910b, a3 and 950"):
        check_snapshot(training_snapshot)


@pytest.mark.parametrize("profile", ["development", "nvidia", "ascend"])
def test_schema4_requires_common_torch_version(training_snapshot, profile):
    def alter(baseline):
        selected = baseline["development_profile"] if profile == "development" else baseline["training_profiles"][profile]
        # Raise the major version so this profile differs whatever is pinned.
        version, _, local = selected["torch"].partition("+")
        major, rest = version.split(".", 1)
        selected["torch"] = f"{int(major) + 1}.{rest}" + (f"+{local}" if local else "")
    change_baseline(training_snapshot, alter)
    with pytest.raises(policy.PolicyError, match="same upstream"):
        check_snapshot(training_snapshot)


def test_schema4_checks_exact_torch_build_in_lock(training_snapshot):
    lock = tomllib.loads(training_snapshot.read(policy.LOCKFILE))
    baseline = json.loads(training_snapshot.read(policy.BASELINE))
    cuda_build = baseline["training_profiles"]["nvidia"]["torch"]
    packages = [(item["name"], item["version"]) for item in lock["package"]
                if "virtual" not in item["source"] and not (item["name"] == "torch" and item["version"] == cuda_build)]
    training_snapshot.content[policy.LOCKFILE] = uv_lock(packages)
    with pytest.raises(policy.PolicyError, match="exact torch build"):
        check_snapshot(training_snapshot)


@pytest.mark.parametrize("profile", ["development", "nvidia", "ascend"])
def test_schema4_checks_torch_extra(training_snapshot, profile):
    source = training_snapshot.read(policy.PROJECT)
    training_snapshot.content[policy.PROJECT] = source.replace(f'{profile} = ["torch==', f'{profile} = ["torch>=')
    with pytest.raises(policy.PolicyError, match=f"{profile} extra"):
        check_snapshot(training_snapshot)


def test_schema4_checks_torch_npu_version(training_snapshot):
    change_baseline(training_snapshot, lambda b: b["training_profiles"]["ascend"].update(torch_npu="2.9.0"))
    with pytest.raises(policy.PolicyError, match="locked torch_npu"):
        check_snapshot(training_snapshot)


@pytest.mark.parametrize("change", ["add", "replace"])
def test_schema4_rejects_docker_sdk_changes(training_snapshot, change):
    filename = "environment/docker/Dockerfile"
    source = training_snapshot.read(filename)
    if change == "add":
        source += "\nFROM ubuntu:24.04 AS unexpected\n"
    else:
        baseline = json.loads(training_snapshot.read(policy.BASELINE))
        source = source.replace(baseline["training_profiles"]["nvidia"]["sdk_image"], "nvidia/cuda:latest")
    training_snapshot.content[filename] = source
    with pytest.raises(policy.PolicyError, match="Dockerfile FROM"):
        check_snapshot(training_snapshot)


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
    gpu, npu, dev = exported("nvidia", machine), exported("ascend", machine), exported("development", machine)
    baseline = json.loads((ROOT / policy.BASELINE).read_text())
    assert gpu["torch"] == baseline["training_profiles"]["nvidia"]["torch"]
    assert npu["torch"] == baseline["training_profiles"]["ascend"]["torch"]
    assert dev["torch"] == baseline["development_profile"]["torch"]
    assert npu["torch-npu"] == baseline["training_profiles"]["ascend"]["torch_npu"]
    dsl_dependencies = {"nvidia-cutlass-dsl", "cuda-python", "cuda-bindings"}
    for name in dsl_dependencies:
        assert name in gpu
        assert name not in npu and name not in dev
    for name in set(gpu) | set(npu):
        if name in {"torch", "torch-npu", "triton"} or name.startswith("nvidia-"):
            continue
        if name in dsl_dependencies:
            continue
        assert gpu.get(name) == npu.get(name), name
        assert gpu.get(name) == dev.get(name), name


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
    for path, profile in policy.TRAINING_DEVCONTAINERS.items():
        dev = json.loads((ROOT / path).read_text())
        assert dev["build"]["target"] == profile
        assert dev["build"]["dockerfile"] == "../../environment/docker/Dockerfile"
    base = json.loads((ROOT / ".devcontainer/nvidia/devcontainer.json").read_text())
    nested = json.loads((ROOT / ".devcontainer/nvidia-nested/devcontainer.json").read_text())
    for key in ("build", "workspaceMount", "workspaceFolder", "customizations", "forwardPorts",
                "portsAttributes", "postCreateCommand", "postStartCommand"):
        assert nested[key] == base[key]
    assert services["nvidia"]["gpus"] == "all"
    assert "/dev/davinci0:/dev/davinci0" in services["ascend"]["devices"]
    assert not services["ascend"].get("privileged", False)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    conflict = project["tool"]["uv"]["conflicts"][0]
    assert {entry["extra"] for entry in conflict} == {"development", "nvidia", "ascend"}



def test_cutlass_dsl_matches_official_source_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    baseline = json.loads((ROOT / policy.BASELINE).read_text())
    gemm = json.loads((ROOT / "environment/gemm.json").read_text())
    profile = baseline["training_profiles"]["nvidia"]
    assert profile["cutlass_dsl"] == gemm["cutlass_version"] == "4.2.1"
    assert "nvidia-cutlass-dsl==4.2.1" in project["project"]["optional-dependencies"]["nvidia"]
    assert baseline["import_map"]["cutlass"] == "nvidia-cutlass-dsl"
    assert baseline["import_map"]["cuda"] == "cuda-bindings"


def test_policy_rejects_mismatched_cutlass_source(training_snapshot):
    gemm = json.loads(training_snapshot.read("environment/gemm.json"))
    gemm["cutlass_version"] = "4.3.0"
    training_snapshot.content["environment/gemm.json"] = json.dumps(gemm)
    with pytest.raises(policy.PolicyError, match="source versions must match"):
        check_snapshot(training_snapshot)


def test_policy_rejects_mismatched_cutlass_package(training_snapshot):
    change_baseline(training_snapshot, lambda b: b["training_profiles"]["nvidia"].update(cutlass_dsl="4.3.0"))
    with pytest.raises(policy.PolicyError, match="CUTLASS DSL version"):
        check_snapshot(training_snapshot)


@pytest.mark.parametrize("profile", ["common", "development", "nvidia", "ascend"])
def test_installer_uses_locked_binary_distributions(tmp_path, profile):
    script = (ROOT / "environment/docker/install-python.sh").read_text()
    venv = tmp_path / "venv"
    venv.mkdir()
    script = script.replace("/opt/venv", str(venv))
    command = 'uv() { printf "%s\\n" "$@"; }\n' + script
    result = subprocess.run(["bash", "-c", command, "installer", profile], capture_output=True, text=True, check=True)
    arguments = result.stdout.splitlines()
    assert "--locked" in arguments
    assert "--no-build" in arguments
    assert "--no-build-isolation-package" not in arguments
    assert ("--extra" in arguments) == (profile != "common")


def test_git_distributions_have_no_policy_exception():
    text = 'version = 1\n[[package]]\nname = "example"\nversion = "1.0.0"\nsource = { git = "https://example.invalid/repo.git#' + "a" * 40 + '" }\n'
    with pytest.raises(policy.PolicyError, match="registry packages"):
        policy.parse_lock(text)
