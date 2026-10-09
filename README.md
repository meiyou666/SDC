# SDC

## 1. 配置开发容器

### VS Code 与 Docker 引擎

VS Code 安装 **Dev Containers** 和 **Container Tools** 扩展。扩展提供容器操作入口，后台仍需 Docker Engine。

Windows 使用 **WSL + Docker Engine** 或 **远程 Linux + Docker Engine**。前者通过 VS Code 的 WSL 扩展连接，后者通过 Remote - SSH 连接。仓库不依赖 Docker Desktop。

**Ubuntu 24.04 后端**

在所选的 WSL 或 Linux 主机中安装 Docker Engine：

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 git
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

注销并重新登录，使用户组生效。

### 获取仓库并启动

在所选的 WSL 或 Linux 主机上获取仓库：

```bash
git clone https://github.com/meiyou666/SDC.git
cd SDC
```

在已连接该主机的 VS Code 窗口中：

1. 打开 [`SDC.code-workspace`](SDC.code-workspace)。
2. 按 `Ctrl+Shift+P`，执行 **Dev Containers: Reopen in Container**。
3. 等待构建完成，在容器终端中开发；Python 自动使用 `/opt/venv/bin/python`。

依赖由 `pyproject.toml` 和 `uv.lock` 管理，构建使用 `uv sync --locked`。在容器内验证：

```bash
python environment/verify_environment.py --smoke
```

源码保存在所选主机的仓库目录。环境更新后执行 **Dev Containers: Rebuild Container**。

终端入口仍可使用：Linux/WSL 运行 `./dev.sh`；Windows 已配置 Docker CLI 和引擎连接时运行 `dev.cmd`。脚本会构建并进入容器。其他设置见[环境说明](environment/README.md)。

配置入口：[Dockerfile](environment/docker/Dockerfile) · [Compose](compose.yaml) · [Dev Container](.devcontainer/devcontainer.json)。

## 2. 项目结构

```text
SDC/
├── environment/
│   ├── docker/
│   │   ├── Dockerfile          # 镜像构建
│   │   ├── entrypoint.sh       # 容器启动
│   │   └── start-dev.ps1       # Windows 启动辅助
│   ├── baseline.json          # 环境基线
│   └── verify_environment.py  # 环境校验
├── compose.yaml               # 命令行容器入口
├── dev.cmd                    # Windows 启动脚本
├── dev.sh                     # Linux 启动脚本
├── pyproject.toml             # 项目依赖声明
├── uv.lock                    # uv 原生锁文件
├── .python-version            # Python 版本
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

**首次配置身份**：

```bash
git config user.name "你的名字"
git config user.email "你的 GitHub 邮箱"
```

GitHub CLI 可选。Windows 可用 Git for Windows 的凭据管理器登录，或使用 GitHub Desktop、VS Code。命令行 HTTPS 认证可使用个人访问令牌；也可[配置 SSH](https://docs.github.com/zh/authentication/connecting-to-github-with-ssh)，将远端改为 `git@github.com:meiyou666/SDC.git`。

**开始任务**：将 `name/task` 换成自己的分支名。

```bash
git switch main
git pull --ff-only
git switch -c name/task
```

**完成后在容器内验证**：

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
