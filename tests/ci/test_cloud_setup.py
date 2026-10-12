import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cloud_config", ROOT / "environment/docker/configure-cloud.py")
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)
SETUP = ROOT / "environment/docker/setup-cloud.sh"


@pytest.fixture
def project(tmp_path):
    for relative in [".devcontainer/devcontainer.json", ".devcontainer/nvidia/devcontainer.json", ".devcontainer/ascend/devcontainer.json", ".env.example"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


@pytest.mark.parametrize("platform", ["linux/amd64", "linux/arm64"])
@pytest.mark.parametrize("profile,soc", [("dev", ""), ("nvidia", ""), ("ascend", "910b"), ("ascend", "a3"), ("ascend", "950")])
def test_generated_configuration_uses_shared_target_and_host_parameters(project, platform, profile, soc):
    original = {path: path.read_bytes() for path in (project / ".devcontainer").rglob("*.json")}
    output = cloud.configure(project, profile, platform, uid=1000, soc=soc, device_gid=1800)
    config = json.loads(output.read_text())
    assert output == project / f".devcontainer/cloud-{profile}/devcontainer.json"
    assert (output.parent / config["build"]["dockerfile"]).resolve() == project / "environment/docker/Dockerfile"
    assert (output.parent / config["build"]["context"]).resolve() == project
    assert config["build"]["target"] == ("development" if profile == "dev" else profile)
    assert f"--platform={platform}" in config["build"]["options"]
    assert f"--platform={platform}" in config["runArgs"]
    assert config["remoteUser"] == "sdc"
    if profile == "nvidia":
        assert "--gpus=all" in config["runArgs"]
        assert config["containerEnv"]["CUDA_VISIBLE_DEVICES"] == "0"
    elif profile == "ascend":
        assert config["build"]["args"]["ASCEND_SOC"] == soc
        assert "--group-add=1800" in config["runArgs"]
        assert "--device=/dev/davinci0" in config["runArgs"]
        assert any("source=/usr/local/Ascend/driver" in value for value in config["mounts"])
    assert all(path.read_bytes() == data for path, data in original.items())
    assert (project / ".env").stat().st_mode & 0o777 == 0o600


def test_root_session_has_writable_tools_and_persistent_personal_state(project):
    config = json.loads(cloud.configure(project, "dev", "linux/amd64", uid=0).read_text())
    assert config["containerUser"] == config["remoteUser"] == "root"
    assert any("target=/root/.codex" in value for value in config["mounts"])
    assert any("target=/root/.local" in value for value in config["mounts"])
    assert config["containerEnv"]["HF_HOME"] == "/root/.cache/huggingface"
    assert config["remoteEnv"]["PATH"].startswith("/root/.local/bin:")


def test_rerun_preserves_api_settings_and_other_profiles(project):
    (project / ".env").write_text("OPENAI_API_KEY=local-placeholder\n")
    first = cloud.configure(project, "dev", "linux/amd64", uid=1000)
    original = first.read_bytes()
    second = cloud.configure(project, "ascend", "linux/arm64", uid=1000, soc="a3", device_gid=1200)
    cloud.configure(project, "ascend", "linux/arm64", uid=1000, soc="950", device_gid=1600)
    assert first.read_bytes() == original
    assert (project / ".env").read_text() == "OPENAI_API_KEY=local-placeholder\n"
    assert json.loads(second.read_text())["build"]["args"]["ASCEND_SOC"] == "950"


def test_generated_configuration_does_not_follow_external_directory(project, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (project / ".devcontainer/cloud-dev").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside this repository"):
        cloud.configure(project, "dev", "linux/amd64", uid=1000)
    assert not list(outside.iterdir())


def shell(body, *args, script=SETUP):
    # SSH accounts have a passwd entry; a CI container's numeric UID may not.
    identity = 'id() { if [[ $# == 1 && $1 == -un ]]; then printf "sdc\\n"; else command id "$@"; fi; }\n'
    return subprocess.run(["bash", "-c", 'source "$1"\nshift\n' + identity + body, "cloud-test", str(script), *args], text=True, capture_output=True, env={**os.environ, "SUDO_USER": ""})


@pytest.mark.parametrize("args", [("invalid",), ("dev", "nvidia"), ("ascend",), ("dev", "--soc", "a3"), ("ascend", "--soc", "bad"), ("ascend", "--soc")])
def test_invalid_selection_stops_before_installation(args):
    result = shell('sdc_install_docker() { echo UNEXPECTED_INSTALL; exit 91; }\nsdc_setup_cloud "$@"', *args)
    assert result.returncode == 2
    assert "UNEXPECTED_INSTALL" not in result.stdout


@pytest.mark.parametrize("profile,target,arch,platform", [("dev", "development", "aarch64", "linux/arm64"), ("nvidia", "nvidia", "x86_64", "linux/amd64")])
def test_setup_builds_selected_image_and_runs_generator(profile, target, arch, platform):
    body = '''
sdc_install_docker() { echo INSTALL_DOCKER; }
sdc_install_nvidia_runtime() { echo INSTALL_NVIDIA_RUNTIME; }
nvidia-smi() { return 0; }
docker() { return 0; }
uname() { printf '%s\\n' "$TEST_ARCH"; }
sdc_docker() { printf 'DOCKER'; printf ' <%s>' "$@"; printf '\\n'; }
TEST_ARCH=$2
sdc_setup_cloud "$1"
'''
    result = shell(body, profile, arch)
    assert result.returncode == 0, result.stderr
    assert f"<--target> <{target}>" in result.stdout
    assert f"<--platform> <{platform}>" in result.stdout
    assert f"<sdc-{profile}:py311>" in result.stdout
    assert "<--entrypoint> <python>" in result.stdout
    assert "<--profile> <" + profile + ">" in result.stdout
    assert ("INSTALL_NVIDIA_RUNTIME" in result.stdout) == (profile == "nvidia")


def test_missing_gpu_driver_stops_before_installation():
    result = shell('nvidia-smi() { return 1; }\nsdc_install_docker() { echo UNEXPECTED_INSTALL; }\nsdc_setup_cloud nvidia')
    assert result.returncode == 1
    assert "NVIDIA 驱动" in result.stderr
    assert "UNEXPECTED_INSTALL" not in result.stdout


def test_failed_installation_does_not_start_build():
    result = shell('sdc_install_docker() { return 17; }\nsdc_docker() { echo UNEXPECTED_BUILD; }\nsdc_setup_cloud dev')
    assert result.returncode == 17
    assert "UNEXPECTED_BUILD" not in result.stdout


def test_existing_nvidia_runtime_is_reused():
    result = shell('sdc_docker() { printf \'{"nvidia":{}}\\n\'; }\nsdc_root() { echo UNEXPECTED_CHANGE; return 99; }\nsdc_install_nvidia_runtime')
    assert result.returncode == 0
    assert "UNEXPECTED_CHANGE" not in result.stdout


def test_busy_docker_is_not_restarted_for_toolkit_installation():
    result = shell('''
sdc_docker() { if [[ $1 == info ]]; then echo '{}'; else echo active-container; fi; }
sdc_root() { echo UNEXPECTED_CHANGE; return 99; }
sdc_install_nvidia_runtime
''')
    assert result.returncode == 1
    assert "当前有容器运行" in result.stderr
    assert "UNEXPECTED_CHANGE" not in result.stdout


def test_healthy_docker_is_not_reinstalled_or_restarted():
    result = shell('''
docker() { return 0; }
dockerd() { return 0; }
sdc_root() { if [[ $1 != docker ]]; then echo UNEXPECTED_CHANGE; return 99; fi; }
sdc_install_docker root
''', script=ROOT / "environment/docker/install-docker.sh")
    assert result.returncode == 0, result.stderr
    assert "UNEXPECTED_CHANGE" not in result.stdout


def test_accessible_docker_is_reused_without_sudo():
    result = shell('''
docker() { return 0; }
dockerd() { return 0; }
sdc_root() { echo UNEXPECTED_SUDO; return 99; }
sdc_install_docker "$(id -un)"
''', script=ROOT / "environment/docker/install-docker.sh")
    assert result.returncode == 0, result.stderr
    assert "UNEXPECTED_SUDO" not in result.stdout


def test_fresh_install_configures_tsinghua_source_and_required_packages():
    result = shell('''
installed=0
repository_file=$(mktemp)
trap 'rm -f "$repository_file"' EXIT
command() { if [[ $1 == -v && ( $2 == docker || $2 == dockerd ) ]]; then return 1; fi; builtin command "$@"; }
docker() { [[ $installed == 1 ]]; }
curl() { while (($#)); do if [[ $1 == -o ]]; then printf key > "$2"; return; fi; shift; done; }
sdc_root() {
    printf 'ROOT'; printf ' <%s>' "$@"; printf '\\n'
    case $1 in
        apt-get) if [[ " $* " == *' docker-ce '* ]]; then installed=1; fi ;;
        install) : ;;
        tee) cat > "$repository_file" ;;
        docker) shift; docker "$@" ;;
        *) return 99 ;;
    esac
}
sdc_install_docker root
cat "$repository_file"
''', script=ROOT / "environment/docker/install-docker.sh")
    assert result.returncode == 0, result.stderr
    assert "https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu" in result.stdout
    assert "<docker-ce> <docker-ce-cli> <containerd.io> <docker-compose-plugin> <docker-buildx-plugin>" in result.stdout


def test_failed_key_download_stops_before_writing_repository():
    result = shell('''
command() { return 1; }
docker() { return 1; }
curl() { return 7; }
sdc_root() { if [[ $1 == tee ]]; then echo UNEXPECTED_REPOSITORY; fi; }
sdc_install_docker root
''', script=ROOT / "environment/docker/install-docker.sh")
    assert result.returncode == 1
    assert "UNEXPECTED_REPOSITORY" not in result.stdout


def test_toolkit_is_configured_before_restarting_empty_docker():
    result = shell('''
configured=0
nvidia-ctk() { return 0; }
sdc_docker() {
    if [[ $1 == info ]]; then
        if [[ $configured == 1 ]]; then echo '{"nvidia":{}}'; else echo '{}'; fi
    fi
}
sdc_root() {
    printf 'ROOT'; printf ' <%s>' "$@"; printf '\\n'
    case $1 in
        nvidia-ctk) configured=1 ;;
        systemctl|service) [[ $configured == 1 ]] ;;
        *) return 99 ;;
    esac
}
sdc_install_nvidia_runtime
''')
    assert result.returncode == 0, result.stderr
    assert "<nvidia-ctk> <runtime> <configure> <--runtime=docker>" in result.stdout
    assert "<restart>" in result.stdout
