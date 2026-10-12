# 云服务器开发与实验

使用 **本机 VS Code → Remote-SSH → 服务器 Docker → 项目容器**。代码、Python、数据和 GPU / NPU 均在服务器上。本机只需 VS Code 与 SSH 客户端。

## 准备服务器

选择可运行 Docker 的 Ubuntu 服务器。通过 VS Code 的 **Remote-SSH: Connect to Host** 连接后，在服务器终端执行：

```bash
git clone https://github.com/meiyou666/SDC.git
cd SDC
cp .env.example .env
```

在服务器的 `.env` 中填写个人 API 密钥。Docker Engine 与 Compose 已安装时，执行 `docker info` 和 `docker compose version` 检查连接；首次安装时复用仓库的 Ubuntu 安装脚本：

```bash
bash environment/docker/install-wsl-docker.sh
```

该脚本使用清华源。安装后断开并重新连接 SSH，使 Docker 用户组生效。

## 配置计算设备

### NVIDIA GPU

服务器驱动由云平台或维护人配置，先运行 `nvidia-smi`。Docker 通过 NVIDIA Container Toolkit 使用显卡；尚未安装 Toolkit 时，在服务器执行：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
  sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

首次配置在启动实验前完成。之后选择 **SDC NVIDIA**，CUDA 与 Python 依赖由项目镜像安装。

### 昇腾 NPU

服务器驱动和固件由云平台或维护人配置，先运行 `npu-smi info`。选择 **SDC Ascend**，CANN 与 `torch_npu` 由项目镜像安装。

维护人按服务器型号统一设置 [.devcontainer/ascend/devcontainer.json](../.devcontainer/ascend/devcontainer.json) 的芯片参数、设备映射和设备组，并同步 [compose.yaml](../compose.yaml)。现有配置映射第 0 张 NPU 与管理设备，并挂载 `/usr/local/Ascend/driver`。命令行的芯片与多卡设置见[训练说明](training.md#启动容器)。

## 打开容器与运行

在 Remote-SSH 窗口打开服务器上的 `SDC` 目录，按 `Ctrl+Shift+P`，执行 **Dev Containers: Reopen in Container**。按用途选择 SDC、SDC NVIDIA 或 SDC Ascend；切换方式与本机相同。

在训练容器终端执行：

```bash
python environment/verify_environment.py --accelerator
bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline
```

模型与预处理数据保存在服务器仓库的 `data/`，检查点与结果在 `results/`。本机和云端是两份工作副本，代码通过个人分支同步。

长时间训练可改用独立后台容器，关闭 VS Code 后仍继续运行。在**服务器宿主机的仓库目录**通过 Compose 准备模型并启动训练，使两步共用下载缓存。以下为 NVIDIA 入口：

```bash
docker compose --profile nvidia run --build --rm --user "$(id -u):$(id -g)" nvidia \
  bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
docker compose --profile nvidia run -d --name sdc-qwen3-nvidia \
  --user "$(id -u):$(id -g)" nvidia \
  bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline
docker logs -f sdc-qwen3-nvidia
```

`Ctrl+C` 退出日志查看，训练继续运行。昇腾将命令中的 profile 与服务名 `nvidia` 替换为 `ascend`，并使用独立容器名。每次新任务使用不同的容器名和输出目录；断点续训使用[训练说明](training.md#准备与训练)中的 `--resume latest`。

## 远程 device 与镜像复用

标准 `torch.device` 接受设备类型与编号，例如 `cuda:0`；加载 `torch_npu` 后使用 `npu:0`。它表示当前进程可访问的设备，不接收远程服务器地址。PyTorch RPC 可以向远程进程发送计算任务，但服务器仍需要运行框架、驱动与工作进程，也需要相应的程序组织。

本项目采用 Remote-SSH 与服务器容器。训练代码在服务器内使用本地设备编号，实验环境沿用同一 Dockerfile 和锁文件。

要减少重复构建，可构建一次镜像再传到服务器。两端使用相同的仓库提交与 CPU 架构。以下以 NVIDIA 镜像为例，在构建机器的仓库目录执行：

```bash
docker compose --profile nvidia build nvidia
docker image save -o ../sdc-nvidia.tar sdc-nvidia:py311
scp -P 22 ../sdc-nvidia.tar 用户名@服务器地址:~/
```

在服务器载入镜像后，使用已有镜像启动：

```bash
docker image load -i ~/sdc-nvidia.tar
cd ~/SDC
docker compose --profile nvidia run --rm --user "$(id -u):$(id -g)" nvidia bash
```

服务器仍需 Docker 与设备驱动，Python 依赖和 SDK 随镜像复用。`docker save` 保存镜像；仓库、数据和个人 `.env` 分别在服务器准备。上述复用入口是 Compose；VS Code 的三个配置仍按各自 Dockerfile 目标构建。

资料：[VS Code 远程容器](https://code.visualstudio.com/remote/advancedcontainers/develop-remote-host) · [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) · [PyTorch 2.7.1 device 定义](https://github.com/pytorch/pytorch/blob/v2.7.1/docs/source/tensor_attributes.rst#torchdevice) · [PyTorch RPC](https://github.com/pytorch/pytorch/blob/v2.7.1/docs/source/rpc.rst)
