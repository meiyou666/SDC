# 公共开发环境

| 项目 | 设置 |
|---|---|
| 容器 | Ubuntu 24.04，Linux amd64，即 x86-64 |
| Python | uv 安装 3.11.17，固定在 `.python-version` |
| uv | 0.12.24，工具镜像固定 digest |
| Node.js、npm | 初始安装 24.21.0、11.19.0，供 AI 工具使用 |
| Codex CLI | 安装 latest，不锁版本 |
| Codex 插件 | Dev Container 自动安装 `openai.chatgpt` |
| Codex 初始模型、审查模型 | `gpt-6-astra` |
| Codex 服务地址 | `https://s2api.top/v1` |
| 依赖声明 | 根目录 `pyproject.toml` |
| 依赖锁定 | 根目录 `uv.lock` |
| Python 依赖源 | [清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple) |
| WSL Docker 安装包源 | [清华 Docker CE 镜像](https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu) |
| 容器基础镜像源 | Ubuntu 使用 Docker Hub；uv 使用 GHCR |
| Node.js、npm 软件源 | npmmirror |

## 启动与执行

按[项目 README](../README.md)选择 WSL 或 Windows 方案，打开 `SDC.code-workspace`，执行 **Dev Containers: Reopen in Container**。

终端启动：WSL 使用 `./dev.sh`，Windows PowerShell 使用 `.\dev.cmd`。

容器通过 `uv sync --locked --all-groups` 安装锁定的依赖。

容器内执行：

```bash
python environment/verify_environment.py --smoke
uv run --locked pytest -q
uv run --locked ruff check .
```

所有项目代码、依赖管理、测试与实验均在 Docker 中执行。宿主机用于编辑、Git 操作和启动容器。

## Codex 配置

首次启动从 `environment/docker/codex-default.toml` 初始化 Codex。模型和审查模型均为 `gpt-6-astra`，启用模型发现和 goals。已有个人配置不被覆盖，可自行增添插件、MCP 和 skills。

根目录 `.env` 已预填服务地址，只需填写个人 `OPENAI_API_KEY`。容器启动时自动完成 API 密钥登录。`.env.example` 为模板，`.env` 不提交。修改后重建 Dev Container，命令行方式重新运行启动脚本。

容器已启动时，也可执行 `sh /opt/sdc/setup-codex.sh` 重新读取 `.env` 并完成登录，然后重新打开 Codex 面板。

容器内执行 `codex`，或使用 VS Code 的 Codex 插件。CLI 更新命令：`npm install -g @openai/codex@latest`。

AI 工具、插件、MCP 和 skills 可自行配置和升级，不加入 CI 检查或测试。额外依赖放在个人目录或独立环境，不修改项目的 `/opt/venv`。两种启动入口均通过数据卷保存 `~/.codex`、`~/.local`、`~/.ssh` 和 `~/.cache`。

## 实验工具

| 用途 | 已安装工具 |
|---|---|
| 远程连接与传输 | SSH、SCP、SFTP、ssh-agent、rsync |
| 代码与大文件 | Git、Git LFS |
| C/C++ 编译 | GCC、G++、Make、CMake、Ninja、pkg-config |
| 调试与进程管理 | GDB、strace、tmux、htop、lsof、time |
| 下载与解压 | curl、wget、aria2、unzip、zip、xz、zstd |
| 文件与配置处理 | ripgrep、jq、less |
| 网络排查 | ip、ping、dig、nc |

系统软件包使用清华 Ubuntu 源。项目 Python 依赖仍由 `uv.lock` 管理。个人命令行工具可通过 `uv tool install 包名` 或 `npm install -g 包名` 安装到独立目录。

共享内存设为 2 GB。模型和数据放在 `data/`，实验输出放在 `results/`。`~/.cache` 持久化保存下载缓存，Hugging Face 缓存位于 `~/.cache/huggingface`。

## 连接远程服务器

在容器终端使用：

```bash
ssh -p 22 用户名@服务器地址
scp -P 22 results/summary.csv 用户名@服务器地址:~/
rsync -avP -e "ssh -p 22" 用户名@服务器地址:~/results/ ./results/
```

SSH 密钥和连接配置放在持久化目录 `~/.ssh`。需要新密钥时运行 `ssh-keygen -t ed25519`，将生成的公钥配置到服务器。

VS Code 自动转发容器的 6006 和 8888 端口。查看远程 TensorBoard 时可运行 `ssh -N -L 6006:127.0.0.1:6006 用户名@服务器地址`，然后在 VS Code 的端口面板打开 6006。

## 维护依赖

由仓库拥有者修改依赖。下面的命令在宿主机调用 Docker，uv 实际运行在 `tools` 容器内：

```bash
docker compose run --build --rm tools add --no-sync "包名==版本"
docker compose run --build --rm tools lock --check
```

开发工具使用 `add --dev --no-sync`。直接修改 `pyproject.toml` 后，执行：

```bash
docker compose run --build --rm tools lock
```

同时提交 `pyproject.toml` 和 `uv.lock`，再使用启动脚本重建开发容器。Python 版本调整时同步修改 `.python-version`、Dockerfile 和 `baseline.json`。导入名与发行包名不同的依赖，在 `import_map` 中登记。

WSL 运行维护命令时，在 `tools` 前添加 `--user "$(id -u):$(id -g)"`，以当前用户写入文件。`dev.sh` 已包含该设置。

## CI

GitHub Actions 在项目容器中运行锁文件检查、测试和 Ruff。

维护人也可手动运行 GitHub Actions 的 **Update uv lock**，下载生成的 `uv.lock` 后提交。切换依赖源时可填写 `index-url`，并同步修改 `pyproject.toml` 中的源地址。
