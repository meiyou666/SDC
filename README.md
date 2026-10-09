# SDC

## 1. 配置开发容器

当前使用 CPU 开发容器。

### 安装 Docker

按宿主机系统选择一组步骤。

**Windows**

1. 在管理员 PowerShell 中安装 WSL，完成后重启并打开 Ubuntu，按提示创建用户：

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

2. 安装并启动 [Docker Desktop](https://www.docker.com/products/docker-desktop/)。在 Settings 中启用 **Use the WSL 2 based engine**，再到 **Resources → WSL Integration** 启用 Ubuntu-24.04。
3. 在 Ubuntu 终端安装 Git 和 GitHub CLI：

   ```bash
   sudo apt update
   sudo apt install -y git gh
   ```

**Ubuntu 24.04 主机**

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 git gh
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

注销并重新登录，使用户组生效。

### 获取仓库并启动

以下命令在 Ubuntu 或 WSL 的终端执行：

```bash
git clone https://github.com/meiyou666/SDC.git
cd SDC
docker compose run --build --rm dev
```

首次构建会自动安装固定版本的 Python 和依赖。进入容器后验证环境：

```bash
python environment/verify_environment.py --smoke
```

输入 `exit` 退出容器，代码保存在本机仓库。环境更新后重新执行启动命令即可重建。

使用 VS Code 时，安装 **Dev Containers** 扩展，打开仓库后执行 **Reopen in Container**；Windows 同时安装 **WSL** 扩展，通过 WSL 打开仓库。其他环境设置见[环境说明](environment/README.md)。

配置入口：[Dockerfile](environment/docker/Dockerfile) · [Compose](compose.yaml) · [Dev Container](.devcontainer/devcontainer.json)。

## 2. 项目结构

```text
SDC/
├── environment/
│   ├── docker/
│   │   ├── Dockerfile          # 镜像构建
│   │   └── entrypoint.sh       # 容器启动
│   ├── dependencies/          # 固定依赖
│   ├── baseline.json          # 环境基线
│   └── verify_environment.py  # 环境校验
├── compose.yaml               # 命令行容器入口
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
├── .github/                   # CI、审核规则与 PR 模板
└── AGENTS.md                  # Codex 开发与审查规则
```

## 3. 协作流程

仓库拥有者可直接提交 `main`；其他成员按以下流程开发和合并。

以下命令在**宿主机的仓库目录**执行。

**首次配置身份**：`gh auth login` 选择 GitHub.com、HTTPS，并通过浏览器登录。

```bash
gh auth login
gh auth setup-git
git config user.name "你的名字"
git config user.email "你的 GitHub 邮箱"
```

**开始任务**：将 `name/task` 换成自己的分支名。

```bash
git switch main
git pull --ff-only
git switch -c name/task
```

**完成后验证、提交并创建 PR**：

```bash
docker compose run --build --rm -T dev python -m pytest -q
docker compose run --rm -T dev ruff check .
git status
git add .
git commit -m "简述改动"
git push -u origin HEAD
gh pr create --base main
```

PR 简述改动和验证结果。Codex 自动审查；CI 通过且获得一名成员批准后合并。Issue 按需使用。

公共环境、CI 和训练配置由 `meiyou666` 维护，成员有调整需求先沟通。
