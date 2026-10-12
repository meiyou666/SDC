# Agent 开发规范

## 项目背景与任务入口

本项目研究**面向低精度矩阵乘法的算法级容错评测与优化**，主线是统一评测、低精度 ABFT 算法优化和昇腾 NPU 算子融合。重点分析主计算误差如何进入校验残差、影响故障检出，并结合 FP8、FP4 的数据表示改进校验与判定，在真实训练中评价检测、恢复和开销。

- 开始任务时先阅读 [项目研究说明](docs/project.md)，了解研究问题、代表方法与实验口径，再阅读本次任务涉及的代码和配置。
- 统一评测与算法研究可以并行，使用共同的训练操作数、故障协议和指标定义。具体优化机制由论文复现与实验结果确定。
- 训练任务读取 [训练说明](docs/training.md)；云端部署读取 [云服务器使用说明](docs/cloud.md)。
- 撰写或修改研究报告前，读取团队共享的 [.agent/skills/revise-evidence-report/SKILL.md](.agent/skills/revise-evidence-report/SKILL.md)，按需加载其参考文件。报告使用简洁、清晰的学术中文。

## 项目约定

- 本仓库用于 ABFT 算法复现、统一评测和昇腾算子实现。
- README 和环境说明使用简洁中文，直接写配置步骤、项目结构与协作流程，不加入审计性说明或方法可行范围论述。
- 所有成员（包括仓库拥有者和 Agent）在个人分支开发，不直接提交或推送到 `main`。拥有者使用 `meiyou666/dev`。修改完成后创建以 `main` 为目标的 PR，通过 CI 和审核后合并；Agent 创建 PR 后等待用户确认合并。Issue 按需使用。
- 项目公共环境、训练配置、CI 和本文件由 `meiyou666` 维护。项目实验不得另建依赖清单或换用其他镜像；新增实验依赖先与维护人沟通。维护人明确安排的环境调整可以直接执行。
- `archive/` 保留历史材料，日常开发不修改、不导入其中的旧实现。模型、数据和完整实验输出放在 `data/`、`results/`，不提交到 Git。

## 执行环境

- **项目代码、依赖解析与安装、测试、实验、绘图和基准测量必须在项目 Docker 容器中执行。禁止使用 Agent 所在的本地 Python、pip、uv、Conda 或其他本地虚拟环境代跑。**
- Agent 已在 Dev Container 内时，直接运行容器中的命令；在容器外时，通过 Docker 调用项目容器。
- 宿主机可运行 AI 工具、编辑器、Git/GitHub 和 Docker；项目代码与实验仍在项目容器中执行。
- 环境文档提供 WSL、Windows 和云服务器三种入口。本机两套方案均通过 VS Code Dev Container 开发，WSL 使用 `./dev.sh`，Windows 使用 `dev.cmd`；云端通过 Remote-SSH 进入服务器上的项目容器。
- 容器不可用时，先修复 Docker，或在 GitHub Actions 中使用项目容器执行；不得回退到宿主机环境。
- 项目 Python 依赖由根目录 `pyproject.toml` 和 `uv.lock` 原生管理。使用 `uv sync --locked`、`uv run --locked`，不得另建实验 requirements 清单或自行升级项目依赖。环境维护使用 Compose 的 `tools` 服务。
- AI 工具的版本、插件、MCP 和 skills 不做统一限制，可自行安装、升级和配置，不加入 AI 工具的 CI 检查或测试。AI 扩展依赖使用个人目录或独立虚拟环境，不改动项目的 `/opt/venv`、依赖锁文件和训练配置。
- 团队共享 skills 放在 `.agent/skills/`，来源与使用入口见 [.agent/README.md](.agent/README.md)。其中的工具依赖与项目实验依赖分开管理；实验实现仍放在 `src/`、`experiments/` 等项目目录。
- Codex 每次安装使用 latest，初始模型和审查模型为 `gpt-6-astra`。首次启动写入初始配置，后续个人设置、插件、MCP 和 skills 自行维护，不覆盖已有配置。API 地址在 `.env.example` 中预填；个人密钥放在 `.env`，不得提交或输出密钥。
- NVIDIA 与昇腾训练镜像共用根目录 `uv.lock` 和 Dockerfile。公共 Python 库必须保持同版本，硬件差异只放在 `nvidia`、`ascend` extra 和对应 SDK 中。本地开发检查使用 `development` extra，不设置 CPU 训练任务。
- 容器内检查：`bash scripts/check.sh`。直接使用 uv 时，按 `SDC_RUNTIME` 指定 `--extra development`、`--extra nvidia` 或 `--extra ascend`。设备检查：`python environment/verify_environment.py --accelerator`。
- 根据改动运行相关检查。公共环境和 CI 改动须通过 `Environment consistency`；没有执行的检查不得写成通过。

## 算法与实验

- 先精读论文并复现核心机制，再做小规模验证，人工比对通过后扩大规模。AI 输出和测试通过均不能代替论文核对。
- 复现记录放在 `docs/reproduction/`，注明论文公式或算法位置、对应实现和小规模验证结果。机制改动应与原方法区分，不把辅助机制描述成主要创新。
- 比较方法须共享输入、故障事件和明确的精度配置。保留原方法机制，单因素实验另行说明。
- 修改测试应覆盖实际行为和失败情形，不为可逆的简单文档修改另写测试。

## Review guidelines

代码审查任务须阅读并遵循 [.github/CODE_REVIEW.md](.github/CODE_REVIEW.md)。本文件规定开发行为，审查标准单独维护在该文件中。
