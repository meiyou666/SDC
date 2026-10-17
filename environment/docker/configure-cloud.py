"""Create the server-local VS Code configuration from a shared Dev Container."""

import argparse
import json
import os
from pathlib import Path
import tempfile


def configure(project, profile, platform, uid, soc="", device_gid=0):
    templates = {
        "dev": (".devcontainer/devcontainer.json", "普通开发"),
        "nvidia": (".devcontainer/nvidia/devcontainer.json", "NVIDIA"),
        "ascend": (".devcontainer/ascend/devcontainer.json", "Ascend"),
    }
    template, label = templates[profile]
    config = json.loads((project / template).read_text())
    config["name"] = f"SDC 云端（{label}）"
    config["build"]["dockerfile"] = "../../environment/docker/Dockerfile"
    config["build"]["context"] = "../.."
    options = config["build"].get("options", [])
    config["build"]["options"] = [x for x in options if not x.startswith("--platform=")] + [f"--platform={platform}"]
    config["runArgs"] = [x for x in config["runArgs"] if not x.startswith(("--platform=", "--group-add="))]
    config["runArgs"].append(f"--platform={platform}")
    if profile == "nvidia":
        config["containerEnv"]["CUDA_VISIBLE_DEVICES"] = "0"
    elif profile == "ascend":
        if soc not in {"910b", "a3", "950"}:
            raise ValueError("Choose the Ascend chip: 910b, a3, or 950.")
        config["build"]["args"]["ASCEND_SOC"] = soc
        config["containerEnv"]["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        config["runArgs"].append(f"--group-add={device_gid}")
    if uid == 0:
        config["containerUser"] = "root"
        config["remoteUser"] = "root"
        # Keep the shared npm prefix and PATH, including accelerator SDK paths.
        config["mounts"] = [
            x if "target=/home/sdc/.local," in x else x.replace("target=/home/sdc/", "target=/root/")
            for x in config["mounts"]
        ]
        config["containerEnv"].update({
            "NPM_CONFIG_CACHE": "/root/.cache/npm",
            "HF_HOME": "/root/.cache/huggingface",
        })

    directory = project / f".devcontainer/cloud-{profile}"
    if directory.is_symlink():
        raise ValueError("The cloud configuration directory must be inside this repository.")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "devcontainer.json"
    if destination.is_symlink():
        raise ValueError("The cloud configuration must be a regular file.")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
        staging = Path(stream.name)
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)
    environment = project / ".env"
    if not environment.exists():
        try:
            descriptor = os.open(environment, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "w") as stream:
                stream.write((project / ".env.example").read_text())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("dev", "nvidia", "ascend"), required=True)
    parser.add_argument("--platform", choices=("linux/amd64", "linux/arm64"), required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--soc", default="")
    parser.add_argument("--device-gid", type=int, default=0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    path = configure(project, args.profile, args.platform, args.uid, args.soc, args.device_gid)
    print(f"VS Code 云端入口：{path.relative_to(project)}")


if __name__ == "__main__":
    main()
