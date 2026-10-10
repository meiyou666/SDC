# Agent 开发规范

## 项目约定

- 本仓库用于 ABFT 算法复现、统一评测和昇腾算子实现。
- README 和环境说明使用简洁中文，直接写配置步骤、项目结构与协作流程，不加入审计性说明或方法可行范围论述。
- 仓库拥有者可直接修改和提交 `main`，不要求新建分支或 PR。其他成员在个人分支开发，通过 CI 和审核后合并。Issue 按需使用。
- 项目公共环境、训练配置、CI 和本文件由 `meiyou666` 维护。项目实验不得另建依赖清单或换用其他镜像；新增实验依赖先与维护人沟通。维护人明确安排的环境调整可以直接执行。
- `archive/` 保留历史材料，日常开发不修改、不导入其中的旧实现。模型、数据和完整实验输出放在 `data/`、`results/`，不提交到 Git。

## 执行环境

- **项目代码、依赖解析与安装、测试、实验、绘图和基准测量必须在项目 Docker 容器中执行。禁止使用 Agent 所在的本地 Python、pip、uv、Conda 或其他本地虚拟环境代跑。**
- Agent 已在 Dev Container 内时，直接运行容器中的命令；在容器外时，通过 Docker 调用项目容器。
- 宿主机仅用于文件编辑、静态查看、Git/GitHub 操作以及 Docker 的安装和调用。
- 文档只保留 WSL 和 Windows 两套本机方案。两套均通过 VS Code Dev Container 开发，WSL 使用 `./dev.sh`，Windows 使用 `dev.cmd`。
- 容器不可用时，先修复 Docker，或在 GitHub Actions 中使用项目容器执行；不得回退到宿主机环境。
- 项目 Python 依赖由根目录 `pyproject.toml` 和 `uv.lock` 原生管理。使用 `uv sync --locked`、`uv run --locked`，不得另建实验 requirements 清单或自行升级项目依赖。环境维护使用 Compose 的 `tools` 服务。
- AI 工具的版本、插件、MCP 和 skills 不做统一限制，可自行安装、升级和配置，不加入 AI 工具的 CI 检查或测试。AI 扩展依赖使用个人目录或独立虚拟环境，不改动项目的 `/opt/venv`、依赖锁文件和训练配置。
- Codex 默认安装 latest，使用其默认设置。API 地址在 `.env.example` 中预填；个人密钥放在 `.env`，不得提交或输出密钥。
- 容器内验证：`python environment/verify_environment.py --smoke`。测试：`uv run --locked pytest -q`。代码检查：`uv run --locked ruff check .`。
- 根据改动运行相关检查。公共环境和 CI 改动须通过 `Environment consistency`；没有执行的检查不得写成通过。

## 算法与实验

- 先精读论文并复现核心机制，再做小规模验证，人工比对通过后扩大规模。AI 输出和测试通过均不能代替论文核对。
- 复现记录放在 `docs/reproduction/`，注明论文公式或算法位置、对应实现和小规模验证结果。机制改动应与原方法区分，不把辅助机制描述成主要创新。
- 比较方法须共享输入、故障事件和明确的精度配置。保留原方法机制，单因素实验另行说明。
- 修改测试应覆盖实际行为和失败情形，不为可逆的简单文档修改另写测试。

## Review guidelines

代码审查任务须阅读并遵循 [.github/CODE_REVIEW.md](.github/CODE_REVIEW.md)。本文件规定开发行为，审查标准单独维护在该文件中。
