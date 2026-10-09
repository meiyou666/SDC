# SDC

## 1. 配置开发容器

### 安装 Docker

按宿主机系统选择一组步骤。

**Windows**

1. 安装 [Git for Windows](https://git-scm.com/downloads/win) 和 [Docker Desktop](https://www.docker.com/products/docker-desktop/)。
2. 启动 Docker Desktop，使用 **Linux containers** 模式。
3. 后续直接在 PowerShell 操作，无需单独安装 Ubuntu。Docker Desktop 默认使用 WSL 2 后端；支持 Hyper-V 的系统也可选择全用户安装与 Hyper-V 后端。

**Ubuntu 24.04 主机**

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 git
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

注销并重新登录，使用户组生效。

### 获取仓库并启动

在 PowerShell 或 Linux 终端获取仓库：

```bash
git clone https://github.com/meiyou666/SDC.git
cd SDC
```

Windows 启动：

```powershell
.\dev.cmd
```

Linux 启动：

```bash
./dev.sh
```

脚本会检查 Docker、构建镜像并进入容器。依赖通过 `pyproject.toml` 和 `uv.lock` 管理，构建使用 `uv sync --locked`。在容器内验证：

```bash
python environment/verify_environment.py --smoke
```

输入 `exit` 退出容器，代码保存在本机仓库。环境更新后重新执行启动命令即可重建。

VS Code 安装 **Dev Containers** 扩展，打开仓库后执行 **Reopen in Container**。其他设置见[环境说明](environment/README.md)。

配置入口：[Dockerfile](environment/docker/Dockerfile) · [Compose](compose.yaml) · [Dev Container](.devcontainer/devcontainer.json)。

## 2. 项目结构

```text
SDC/
├── environment/
│   ├── docker/
│   │   ├── Dockerfile          # 镜像构建
│   │   └── entrypoint.sh       # 容器启动
│   ├── baseline.json          # 环境基线
│   └── verify_environment.py  # 环境校验
├── compose.yaml               # 命令行容器入口
├── dev.cmd                    # Windows 启动脚本
├── dev.sh                     # Linux 启动脚本
├── pyproject.toml             # 项目依赖声明
├── uv.lock                    # uv 原生锁文件
├── .python-version            # Python 版本
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
