# 公共开发环境

| 项目 | 设置 |
|---|---|
| 容器 | Debian 12，Linux amd64 |
| Python | 3.11.17，固定在 `.python-version` |
| uv | 0.12.24，工具镜像固定 digest |
| Node.js、npm | 初始安装 24.21.0、11.19.0，供 AI 工具使用 |
| Codex CLI | 安装 latest，不锁版本 |
| Codex 插件 | Dev Container 自动安装 `openai.chatgpt` |
| Codex 服务地址 | `https://s2api.top` |
| 依赖声明 | 根目录 `pyproject.toml` |
| 依赖锁定 | 根目录 `uv.lock` |
| Python 依赖源 | [清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple) |
| WSL Docker 安装包源 | [清华 Docker CE 镜像](https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu) |
| 容器基础镜像源 | Python 使用 Docker Hub；uv 使用 GHCR |
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

Codex 使用安装后的默认配置。

根目录 `.env` 已预填服务地址，只需填写个人 `OPENAI_API_KEY`。容器启动时自动完成 API 密钥登录。`.env.example` 为模板，`.env` 不提交。修改后重建 Dev Container，命令行方式重新运行启动脚本。

容器内执行 `codex`，或使用 VS Code 的 Codex 插件。CLI 更新命令：`npm install -g @openai/codex@latest`。

AI 工具、插件、MCP 和 skills 可自行配置和升级，不加入 CI 检查或测试。额外依赖放在个人目录或独立环境，不修改项目的 `/opt/venv`。Dev Container 使用数据卷保留 `~/.codex` 和 `~/.local`。

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
