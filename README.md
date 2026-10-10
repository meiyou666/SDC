# SDC

## 1. 配置开发容器

### 第一步：克隆仓库

在 PowerShell 中执行以下命令。未安装 Git 时，先安装 [Git for Windows](https://git-scm.com/downloads/win)。

```powershell
git clone https://github.com/meiyou666/SDC.git
cd SDC
Copy-Item .env.example .env
```

随后选择下面一种环境方案。两套方案均在 VS Code 中开发，共用 Dockerfile、Compose、`pyproject.toml` 和 `uv.lock`。VS Code 安装 **Dev Containers** 和 **Container Tools** 扩展。

在 `.env` 中填写 `OPENAI_API_KEY`，按需填写 `CODEX_MODEL`。API 地址统一为 `https://s2api.top`。`.env` 只保存在本机。

### 方案一：WSL

1. 在管理员 PowerShell 执行 `wsl --install -d Ubuntu-24.04`，按提示重启并完成 Ubuntu 用户初始化。
2. 在已克隆的仓库目录打开 PowerShell，执行 `wsl -d Ubuntu-24.04`。进入 Ubuntu 后，从清华源安装 Docker Engine 和 Compose：

   ```bash
   bash environment/docker/install-wsl-docker.sh
   ```

3. 执行 `exit` 返回 PowerShell，再执行 `wsl -d Ubuntu-24.04`，使 Docker 用户组设置生效。
4. VS Code 安装 **WSL** 扩展，连接 Ubuntu，打开仓库中的 [`SDC.code-workspace`](SDC.code-workspace)。
5. 执行 **Dev Containers: Reopen in Container**。终端启动命令为 `./dev.sh`。

### 方案二：Windows

1. 下载 [Docker Desktop 安装程序](https://docs.docker.com/desktop/setup/install/windows-install/)。在安装包所在目录打开 PowerShell，使用 Docker VMM 后端安装：

   ```powershell
   Start-Process '.\Docker Desktop Installer.exe' -Wait -ArgumentList 'install','--user','--backend=docker-vmm'
   ```

2. 安装完成后启动 Docker Desktop，重新打开 VS Code 和仓库目录中的 PowerShell。
3. 在 Docker Desktop 的 **Settings** 完成以下设置，然后点击 **Apply & restart**：

   - **General → Virtual Machine Manager**：选择 **Docker VMM**。
   - **Resources**：分配至少 4 GB 内存。
   - **Resources → File sharing**：添加仓库目录。

4. 在 Windows 的 VS Code 中打开 [`SDC.code-workspace`](SDC.code-workspace)，执行 **Dev Containers: Reopen in Container**。PowerShell 启动命令为 `.\dev.cmd`。

进入容器后，Python 使用 `/opt/venv/bin/python`，依赖由 `uv sync --locked` 安装。环境更新后执行 **Dev Containers: Rebuild Container**。详细设置见[环境说明](environment/README.md)。

Python 依赖和 WSL 下的 Docker 安装包使用清华源；容器基础镜像来自 Docker Hub 和 GHCR。

容器已安装 Node.js、npm 和 Codex CLI，VS Code 自动安装 Codex 插件。进入容器后可在侧栏使用 Codex，或在终端运行 `codex`。CLI 不锁版本，更新命令为 `npm install -g @openai/codex@latest`。修改 `.env` 后执行 **Dev Containers: Rebuild Container**。

配置入口：[Dockerfile](environment/docker/Dockerfile) · [Compose](compose.yaml) · [Dev Container](.devcontainer/devcontainer.json)。

## 2. 项目结构

```text
SDC/
├── environment/
│   ├── codex/                 # Codex 公共配置
│   ├── docker/
│   │   ├── Dockerfile             # 镜像构建
│   │   ├── entrypoint.sh          # 容器启动
│   │   ├── install-wsl-docker.sh  # WSL 安装 Docker，使用清华源
│   │   └── start-dev.ps1          # Windows 启动辅助
│   ├── baseline.json          # 环境基线
│   └── verify_environment.py  # 环境校验
├── compose.yaml               # 命令行容器入口
├── dev.cmd                    # Windows 启动脚本
├── dev.sh                     # WSL 启动脚本
├── pyproject.toml             # 项目依赖声明
├── uv.lock                    # uv 原生锁文件
├── .python-version            # Python 版本
├── .env.example               # 个人 API 配置模板
├── .env                       # 个人密钥，不提交
├── SDC.code-workspace          # VS Code 工作区入口
├── .vscode/extensions.json     # 推荐扩展
├── .devcontainer/             # VS Code 容器配置
├── src/
│   ├── algorithms/            # ABFT 算法
│   ├── evaluation/            # 注入、统计与计时
│   ├── training/              # 训练流程与 GEMM 采集
│   └── backends/ascend/        # 昇腾算子实现
├── configs/
│   ├── training/              # 公共训练配置
│   └── evaluation/            # 评测配置
├── docs/
│   ├── papers/                # 论文精读记录
│   └── reproduction/          # 复现与人工比对记录
├── experiments/               # 实验配置与小型摘要
├── scripts/                   # 运行脚本
├── tests/                     # 测试
├── data/                      # 本地模型与数据，不提交
├── results/                   # 本地实验输出，不提交
├── archive/                   # 旧实验归档
├── .github/
│   ├── workflows/             # CI
│   └── CODE_REVIEW.md         # 代码审查规范
└── AGENTS.md                  # Agent 开发规范
```

## 3. 协作流程

仓库拥有者可直接提交 `main`；其他成员按以下流程开发和合并。

以下命令在**宿主机的仓库目录**执行。

通过 VS Code、GitHub Desktop 或 Git 命令行提交。HTTPS 使用凭据管理器或个人访问令牌认证；使用 [SSH](https://docs.github.com/zh/authentication/connecting-to-github-with-ssh) 时，将远端设为 `git@github.com:meiyou666/SDC.git`。

**开始任务**：将 `name/task` 换成自己的分支名。

```bash
git switch main
git pull --ff-only
git switch -c name/task
```

**在容器内运行测试**：

```bash
uv run --locked pytest -q
uv run --locked ruff check .
```

**回到宿主机提交**：

```bash
git status
git add .
git commit -m "简述改动"
git push -u origin HEAD
```

在 GitHub 网页点击 **Compare & pull request**，简述改动和验证结果。Codex 自动审查；CI 通过且获得一名成员批准后合并。Issue 按需使用。
