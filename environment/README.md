# 环境搭建与日常使用

完成本教程后，你可以在 VS Code 中进入项目容器，使用统一的 Python 环境开发、检查和运行实验。选择一种搭建方式即可。

| 开发位置 | Docker 所在位置 | VS Code 打开的仓库 |
| --- | --- | --- |
| [本地 Windows](#本地-windows) | Docker Desktop | Windows 中的仓库 |
| [本地 WSL](#本地-wsl) | Ubuntu 中的 Docker Engine | WSL 文件系统中的仓库 |
| [云服务器](#云服务器) | 服务器中的 Docker Engine | Remote-SSH 打开的服务器仓库 |

项目代码、依赖安装、测试与实验均在容器中执行。宿主机用于编辑、Git 操作和启动 Docker。训练步骤见[训练教程](../docs/training.md)，CUTLASS 构建与接入见[GEMM 教程](../docs/gemm.md)。

## 本地 Windows

1. 安装 [Git for Windows](https://git-scm.com/downloads/win)，在准备保存项目的目录打开 **Windows PowerShell**，先克隆仓库：

   ```powershell
   git clone https://github.com/meiyou666/SDC.git
   cd SDC
   ```

2. 下载最新的 [Docker Desktop Windows 安装程序](https://docs.docker.com/desktop/setup/install/windows-install/)。本路线使用 **Docker VMM（Beta）**，无需配置 WSL。在**安装包所在目录的 PowerShell**执行：

   ```powershell
   Start-Process '.\Docker Desktop Installer.exe' -Wait -ArgumentList 'install','--user','--backend=docker-vmm'
   ```

3. 启动 Docker Desktop。在 **Settings → Resources** 为虚拟机分配至少 4 GB 内存；在 **General → Virtual Machine Manager** 选择 **Docker VMM**。在 **Resources → File sharing** 添加仓库目录，点击 **Apply & restart**。设置说明见[官方 VMM 文档](https://docs.docker.com/desktop/features/vmm/)。
4. 重新打开终端和 VS Code。安装 **Dev Containers** 和 **Container Tools** 扩展，打开仓库中的 `SDC.code-workspace`。
5. 按 `Ctrl+Shift+P`，执行 **Dev Containers: Reopen in Container**，选择 **SDC**。首次进入会构建镜像并安装 VS Code 扩展。
6. 按[确认环境可用](#确认环境可用)检查容器。

## 本地 WSL

1. 尚未安装 WSL 时，在**管理员 PowerShell**执行：

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

   按提示重启并完成 Ubuntu 用户初始化。随后在 **PowerShell** 进入 Ubuntu：

   ```powershell
   wsl -d Ubuntu-24.04
   ```

2. 在 **Ubuntu 终端**克隆仓库，放在 WSL 自己的文件系统中：

   ```bash
   git clone https://github.com/meiyou666/SDC.git ~/SDC
   cd ~/SDC
   ```

   使用 `~/SDC`，避免将项目放在 `/mnt/c/...`。跨文件系统访问会拖慢依赖安装、数据处理和结果写入。

3. 在同一 **Ubuntu 终端、仓库根目录**安装 Docker Engine、Compose 和 Buildx：

   ```bash
   bash environment/docker/install-docker.sh
   ```

   脚本使用清华 Docker CE 源。若提示用户组更新，执行 `exit` 返回 PowerShell，再执行 `wsl -d Ubuntu-24.04` 重新登录。

4. 在 VS Code 安装 **WSL**、**Dev Containers** 和 **Container Tools** 扩展。通过 WSL 扩展连接 Ubuntu，打开 `~/SDC/SDC.code-workspace`。
5. 执行 **Dev Containers: Reopen in Container**，选择 **SDC**。首次进入会构建镜像。随后按[确认环境可用](#确认环境可用)检查。

## 云服务器

连接顺序为：**本机 VS Code → Remote-SSH → 服务器仓库 → 远端容器**。服务器使用 Ubuntu；GPU / NPU 驱动由云平台或维护人准备。

### 连接服务器并克隆仓库

在**本机 VS Code**安装 **Remote-SSH**、**Dev Containers** 和 **Container Tools** 扩展。

1. 执行 **Remote-SSH: Add New SSH Host...**，输入云平台提供的命令，例如 `ssh 用户名@服务器地址 -p 22`。
2. 执行 **Remote-SSH: Connect to Host...**，选择服务器并完成认证。
3. 左下角显示 `SSH: 服务器` 后，新建终端。以下命令在这个**服务器终端**执行：

   ```bash
   git clone https://github.com/meiyou666/SDC.git
   cd SDC
   ```

   若没有 Git，先执行 `sudo apt-get update` 和 `sudo apt-get install -y git`；root 用户省略 `sudo`。

### 初始化服务器环境

在 **Remote-SSH 的服务器终端、仓库根目录**执行：

```bash
bash environment/docker/setup-cloud.sh
```

按提示选择普通开发、NVIDIA 或昇腾。脚本安装或复用 Docker、Compose 和 Buildx，构建镜像，再生成服务器专用的 VS Code 入口。

| 选择 | VS Code 入口 | 服务器准备 |
| --- | --- | --- |
| 1：普通开发 | SDC 云端（普通开发） | Ubuntu |
| 2：NVIDIA | SDC 云端（NVIDIA） | `nvidia-smi` 可正常运行；脚本配置 Container Toolkit |
| 3：昇腾 | SDC 云端（Ascend） | `npu-smi info` 可正常运行，驱动和设备文件已就绪 |

也可在同一**服务器终端**直接指定环境，以下命令任选一条：

```bash
bash environment/docker/setup-cloud.sh dev
bash environment/docker/setup-cloud.sh nvidia
bash environment/docker/setup-cloud.sh ascend --soc 910b
```

昇腾 A2 选择 `910b`，A3 选择 `a3`，950 选择 `950`。脚本自动识别 x86-64 或 ARM64，并读取昇腾第 0 张卡的设备组。生成的 GPU / NPU 入口默认使用设备 0。

入口保存在 `.devcontainer/cloud-*/`，只在这台服务器使用，不提交 Git。脚本同时创建 `.env`，保留已有文件。需要 Codex 时填写个人 `OPENAI_API_KEY`，见[Codex 配置](#codex-配置)。

### 进入远端容器

1. 若脚本提示用户组更新，执行 **Remote-SSH: Kill VS Code Server on Host**，选择服务器，再重新连接。
2. 在 **SSH 窗口**执行 **文件 → 打开文件夹**，打开服务器上刚克隆的 `SDC` 目录。
3. 执行 **Dev Containers: Reopen in Container**，选择刚生成的 **SDC 云端（…）**。VS Code 使用构建缓存启动容器。
4. 左下角显示容器名称后，新建终端，按下一节确认环境可用。此后 Python、调试器和 Codex 均在服务器容器中运行。

## 确认环境可用

以下命令在 **VS Code 的容器终端、仓库根目录**执行：

```bash
pwd
command -v python
python environment/verify_environment.py --smoke
```

目录应为 `/workspace/SDC`，Python 应为 `/opt/venv/bin/python`，检查输出中的 `smoke_test` 应为 `passed`。

GPU / NPU 容器再执行设备检查，确认命令成功且 `accelerator` 字段显示设备信息：

```bash
python environment/verify_environment.py --accelerator
```

环境就绪后，可继续[训练教程](../docs/training.md)或[GEMM 教程](../docs/gemm.md)。

## 选择、切换与重建容器

执行 **Dev Containers: Reopen in Container** 时，可以看到以下三个公共入口。云端初始化后，还会显示 **SDC 云端（…）**。

![VS Code 容器选择列表：SDC、SDC Ascend、SDC NVIDIA](../docs/images/devcontainer-selection.png)

| 容器 | 用途 | 所需硬件 |
| --- | --- | --- |
| **SDC** | 日常开发、调试和单元测试 | 普通开发机，不用于训练 |
| **SDC NVIDIA** | NVIDIA 训练与实验，含 CUDA、CUTLASS 和 Transformer Engine | NVIDIA GPU 与可用驱动 |
| **SDC Ascend** | 昇腾训练与实验，含 CANN | 昇腾 NPU 与可用驱动、固件 |

本地日常开发选择 **SDC**；云端选择初始化脚本生成的入口。

- **再次打开**：恢复之前的容器窗口，或打开同一仓库后执行 **Reopen in Container**。
- **切换已有配置**：在容器窗口执行 **Dev Containers: Switch Container**，选择另一个配置。首次使用该配置时需要构建，之后使用已有镜像和构建缓存。
- **新增云端环境或更换昇腾芯片**：执行 **Dev Containers: Reopen Folder in SSH**，回到服务器终端重跑 `setup-cloud.sh`，然后进入对应云端入口。
- **代码改动**：仓库目录直接挂载到容器，无需重建。
- **Dockerfile、锁文件或容器配置更新**：执行 **Dev Containers: Rebuild Container**。云端先在服务器终端重跑初始化脚本，更新生成的入口。

关闭 Dev Container 窗口会停止对应容器。长时间训练按[训练教程](../docs/training.md)使用独立后台任务。

也可仅使用终端。在 **WSL 的仓库根目录**执行 `./dev.sh`；在 **Windows PowerShell 的仓库根目录**执行 `.\dev.cmd`。两者固定启动普通开发环境，每次检查构建并创建临时容器，退出后删除容器。GPU / NPU 启动命令见[训练教程](../docs/training.md)。

## Codex 配置

首次启动会从 `.env.example` 创建仓库根目录的 `.env`。使用 Codex 时，在 **VS Code 编辑器**中打开当前主机仓库的 `.env`，填写个人 `OPENAI_API_KEY`。云端填写服务器上的文件。服务地址已预填，`.env` 不提交 Git。

容器启动时尝试登录；失败不会阻止容器使用。修改密钥后，在**容器终端**执行，再重新打开 Codex 面板：

```bash
sh /opt/sdc/setup-codex.sh
```

可使用 VS Code 的 Codex 插件，或在**容器终端**运行 `codex`。首次初始化的模型和审查模型均为 `gpt-6-astra`，已有个人配置不会被覆盖。需要更新 CLI 时，在**容器终端**执行：

```bash
npm install -g @openai/codex@latest
```

个人插件、MCP、skills 和工具可自行配置。额外工具依赖放在个人目录或独立环境，不改项目的 `/opt/venv`。

## 文件保存与 SSH 日常操作

代码保存在宿主机仓库，模型和数据放在 `data/`，实验输出放在 `results/`。后两者不提交 Git。容器通过数据卷保存 Codex 配置、个人工具、SSH 配置与下载缓存；重建容器后仍可使用。Dev Container 与命令行入口使用各自的数据卷，个人配置不会在两种入口间自动同步。

通过 Remote-SSH 开发时，连接服务器和传输结果的命令在**本机终端**执行。将用户名、地址、端口和仓库路径换成实际值：

```bash
ssh -p 22 用户名@服务器地址
scp -P 22 用户名@服务器地址:~/SDC/results/summary.csv .
```

**本机 WSL / Linux 终端**还可增量下载结果目录：

```bash
rsync -avP -e "ssh -p 22" 用户名@服务器地址:~/SDC/results/ ./results/
```

SSH 连接配置和密钥放在发起连接一侧的 `~/.ssh`。如需新密钥，在该终端执行 `ssh-keygen -t ed25519`，再将公钥配置到目标服务器。容器也已安装 SSH、SCP、SFTP 和 rsync，可从容器连接其他服务器。

VS Code 配置了 6006（TensorBoard）和 8888（Jupyter）端口转发。服务在容器中启动后，通过 VS Code 的**端口**面板打开对应地址。

## 常见问题

| 问题 | 处理方式 |
| --- | --- |
| 找不到 Docker 或无法连接 | Windows 启动 Docker Desktop，并重开终端；WSL / 云端在宿主机执行 `docker info`，确认 Docker 服务与访问权限 |
| 安装后仍提示 Docker 权限不足 | WSL 重新登录；Remote-SSH 执行 **Kill VS Code Server on Host** 后重连 |
| Windows 找不到 Docker VMM 选项或安装参数不识别 | 更新到最新 Docker Desktop，并按官方 Windows 安装说明确认系统要求 |
| Windows 提示仓库目录未共享 | 在 Docker Desktop 的 **Resources → File sharing** 添加仓库目录后重试 |
| 初始化提示已有 Docker 软件包冲突 | 请管理员按脚本提示处理已有服务及冲突包，再重跑初始化脚本 |
| NVIDIA 初始化提示有容器运行 | 首次配置运行时需要重启 Docker，等已有任务结束后重跑脚本 |
| 云端容器入口未出现 | 确认打开的是服务器仓库；初始化必须完成镜像构建后才生成入口，失败后修复报错并重跑 |
| GPU / NPU 检查失败 | 先回到服务器终端检查 `nvidia-smi` 或 `npu-smi info`，再确认选中了对应云端容器 |
| 更新仓库后环境检查失败 | 重建容器；云端先重跑初始化脚本，再执行 **Rebuild Container** |
| Codex 登录失败 | 检查当前仓库 `.env` 的密钥与网络，在容器中重新运行登录脚本；无需重建镜像 |
| 下载镜像或依赖失败 | 检查当前 Docker 主机的网络或代理，恢复后重试；云端检查服务器网络 |

环境版本、依赖维护和 CI 说明见[维护参考](../.agent/reference.md)。返回[项目入口](../README.md)。
