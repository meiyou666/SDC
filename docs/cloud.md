# 云端开发

本机 VS Code 通过 SSH 连接云服务器，再进入服务器上的 Docker 容器。Docker、Python、模型与数据均位于服务器。

## 1. 在本机 VS Code 连接服务器

安装 **Remote-SSH**、**Dev Containers** 和 **Container Tools** 扩展。

1. 按 `Ctrl+Shift+P`，执行 **Remote-SSH: Add New SSH Host...**。
2. 输入云平台提供的连接命令，例如 `ssh 用户名@服务器地址 -p 22`，保存到 SSH 配置。
3. 执行 **Remote-SSH: Connect to Host...**，选择刚添加的服务器，按提示输入密码或使用 SSH 密钥。
4. 左下角显示 `SSH: 服务器` 后，通过 **终端 → 新建终端** 打开远端终端。

**接下来的安装命令都在这个远端终端执行。** 本机使用 VS Code 和 SSH，服务器使用 Ubuntu。GPU / NPU 驱动由云平台或维护人配置。

## 2. 在服务器克隆仓库

```bash
git clone https://github.com/meiyou666/SDC.git
cd SDC
```

若提示未安装 Git，先执行 `sudo apt-get update` 和 `sudo apt-get install -y git`；root 用户省略 `sudo`。

## 3. 安装 Docker 并构建环境

```bash
bash environment/docker/setup-cloud.sh
```

按提示选择：

| 选择 | 构建内容 | VS Code 中的云端入口 |
| --- | --- | --- |
| 1：普通开发 | 公共开发环境 | SDC 云端（普通开发） |
| 2：NVIDIA | CUDA 与 GPU 训练环境 | SDC 云端（NVIDIA） |
| 3：昇腾 | CANN 与 NPU 训练环境 | SDC 云端（Ascend） |

脚本依次完成：

1. 安装或复用 Docker Engine、Compose 和 Buildx。Docker 安装包使用清华源。
2. 根据硬件配置容器访问。NVIDIA 配置 Container Toolkit；昇腾读取第 0 张卡的设备组，并设置芯片与驱动挂载。
3. 使用项目 Dockerfile 构建所选镜像。
4. 在 `.devcontainer/cloud-*/devcontainer.json` 保存服务器专用入口，自动创建 `.env`。

也可直接指定环境：

```bash
bash environment/docker/setup-cloud.sh dev
bash environment/docker/setup-cloud.sh nvidia
bash environment/docker/setup-cloud.sh ascend --soc 910b
```

昇腾 A2 的 `--soc` 为 `910b`，A3 为 `a3`，950 为 `950`。脚本根据服务器自动选择 x86-64 或 ARM64；已有 `.env` 的内容会保留。重新运行同一命令可继续构建。

## 4. 在 VS Code 进入远端容器

1. 如果脚本提示用户组已更新，执行 **Remote-SSH: Kill VS Code Server on Host**，选择这台服务器。随后重新连接，使 Docker 用户组生效。
2. 在 **SSH 窗口**中选择 **文件 → 打开文件夹**，打开刚克隆的服务器 `SDC` 目录。
3. 打开服务器上的 `.env`，填写自己的 `OPENAI_API_KEY`。
4. 按 `Ctrl+Shift+P`，执行 **Dev Containers: Reopen in Container**。
5. 选择脚本生成的 **SDC 云端（…）** 入口。VS Code 复用构建缓存，启动容器并安装 Python、Codex 等扩展。

左下角变为容器名称后，新建终端：

```bash
pwd
command -v python
python environment/verify_environment.py --smoke
```

目录应为 `/workspace/SDC`，Python 应为 `/opt/venv/bin/python`。此后编辑、调试和运行都使用服务器上的容器环境，代码保存在服务器仓库。

GPU / NPU 容器再执行设备检查：

```bash
python environment/verify_environment.py --accelerator
```

训练入口见[公共训练任务](training.md#准备与训练)。

## 5. 后续打开、切换和更新

- **再次开发**：通过 Remote-SSH 打开服务器仓库，再进入对应云端容器；恢复之前的容器窗口也可继续使用。
- **选择其他环境**：执行 **Dev Containers: Reopen Folder in SSH**，在服务器终端重跑初始化脚本，选择另一环境；再执行 **Reopen in Container**。各环境的云端入口分别保存。
- **切换已构建容器**：在容器窗口执行 **Dev Containers: Switch Container**，选择对应云端入口。
- **更新项目环境**：同步仓库后运行初始化脚本，再执行 **Dev Containers: Rebuild Container**。

服务器专用入口只记录架构、芯片和设备参数，沿用项目 Dockerfile 与锁文件。它保存在所在服务器，不提交 Git。

长时间训练的后台运行与断点续训见[训练说明](training.md#后台训练)。

资料：[VS Code 远程容器](https://code.visualstudio.com/remote/advancedcontainers/develop-remote-host) · [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
