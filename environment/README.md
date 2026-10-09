# 公共开发环境

当前统一的是 **CPU 开发容器**，用于论文算法复现、小规模数值验证、测试和数据分析。基础环境已冻结，昇腾平台配置保持待定。

| 项目 | 统一设置 |
|---|---|
| 系统 | Debian 12，Linux amd64 |
| Python | 3.11.17 |
| 数值计算 | NumPy、SciPy |
| 数据与绘图 | pandas、Matplotlib、PyYAML |
| 开发工具 | pytest、Ruff、uv、pip，镜像内提供 Git 和 C/C++ 编译工具 |
| 版本依据 | `baseline.json` 和 `dependencies/requirements.lock` |
| 后续配置 | NPU 型号、驱动、固件、CANN、PyTorch、`torch_npu` |

基础镜像固定到 SHA256 digest，全部 Python 依赖固定版本并校验下载文件的哈希。容器不会在启动时自动更新软件。PyTorch 与 `torch_npu` 待硬件和 CANN 版本确定后配套加入。

## 成员开始开发

先安装 Docker Desktop，或 Docker Engine 与 Compose v2。Windows 建议使用 Docker Desktop 的 WSL2 后端。宿主机无需另外安装 Python 或项目依赖。

### VS Code

安装 Dev Containers 扩展，打开仓库，执行 **Dev Containers: Reopen in Container**。首次会构建镜像；编辑器使用 `/opt/venv/bin/python`，源码仍保存在本地仓库。

### 命令行

在仓库根目录执行：

```bash
docker compose run --build --rm dev
```

进入容器后可以直接运行：

```bash
python environment/verify_environment.py --smoke
python -m pytest -q
ruff check .
```

也可以在宿主机运行一次性任务：

```bash
docker compose run --build --rm -T dev python -m pytest -q
```

容器默认以普通用户运行。Linux 宿主用户不是 UID 1000 时，命令行可使用 `docker compose run --build --rm --user "$(id -u):$(id -g)" -e HOME=/tmp dev`；VS Code 会自动匹配用户 UID。ARM 电脑使用相同的 amd64 容器，需要 Docker 支持架构模拟。

## 环境更新

成员在自己的分支开发，不自行安装或升级公共依赖。需要新依赖时交由维护人统一处理。拉取环境更新后重新构建容器；VS Code 使用 **Rebuild Container**。

维护人修改 `dependencies/requirements.in` 后，在现有公共容器中重新生成完整依赖锁：

```bash
uv pip compile environment/dependencies/requirements.in \
  --python-version 3.11.17 --python-platform x86_64-manylinux_2_36 \
  --no-python-downloads \
  --generate-hashes --only-binary :all: \
  --output-file environment/dependencies/requirements.lock
```

若导入名与发行包名不同，在 `baseline.json` 的 `import_map` 中补充映射。随后重新构建并运行环境验证、测试和 CI。Python 或基础镜像调整时，同时修改 `baseline.json` 与 Dockerfile。

## CI 与运行校验

`Environment consistency` 强制检查公共环境的维护权限、固定版本和代码导入，并实际构建容器，在容器内核对安装版本、运行基础数值与绘图检查、测试及 Ruff。每次启动容器还会将实际 Python 和依赖版本与仓库基线比对，发现旧镜像或额外安装的包时退出并提示重建。

拿到计算平台后，由维护人补充昇腾环境并完成设备验证。CPU 测试用于算法和工具的正确性检查，NPU 上的数值路径、计时和性能结果另行测量。
