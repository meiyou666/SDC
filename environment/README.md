# 公共开发环境

| 项目 | 设置 |
|---|---|
| 容器 | Debian 12，Linux amd64 |
| Python | 3.11.17，固定在 `.python-version` |
| uv | 0.12.24，工具镜像固定 digest |
| 依赖声明 | 根目录 `pyproject.toml` |
| 依赖锁定 | 根目录 `uv.lock` |
| 后续平台 | 昇腾型号、驱动、固件、CANN、PyTorch 与 `torch_npu` 待统一配置 |

## 启动与执行

通过 VS Code 的 WSL 或 Remote - SSH 连接配置了 Docker Engine 的主机，打开 `SDC.code-workspace`，执行 **Dev Containers: Reopen in Container**。首次使用步骤见[项目 README](../README.md)。

终端入口：Linux/WSL 使用 `./dev.sh`；Windows 已配置 Docker CLI 和引擎连接时使用 `dev.cmd`。仓库不绑定 Docker Desktop。

容器通过 `uv sync --locked --all-groups` 安装依赖。启动时比较镜像与仓库配置，并通过 `uv sync --locked --check --offline --all-groups` 核对实际环境。

容器内执行：

```bash
python environment/verify_environment.py --smoke
uv run --locked pytest -q
uv run --locked ruff check .
```

所有项目代码、依赖管理、测试与实验均在 Docker 中执行。宿主机用于编辑、Git 操作和启动容器。

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

Linux 用户 UID 不是 1000 时，运行上述维护命令需在 `tools` 前加 `--user "$(id -u):$(id -g)"`。`dev.sh` 会自动匹配当前用户。

## CI

`Environment consistency` 在 Docker 内检查环境维护权限、导入依赖、原生锁文件与实际安装状态，并执行测试、绘图检查和 Ruff。锁文件与项目声明不一致时检查失败。

维护人也可手动运行 GitHub Actions 的 **Update uv lock**，下载生成的 `uv.lock` 后提交。
