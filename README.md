# SDC

团队协作开展 SDC 统一评测、算法研究与系统实现的仓库。

公共 CPU 开发容器已配置。团队使用相同的 Python、依赖和测试工具开展前期开发，昇腾平台和训练基准待硬件确定后统一补充。

## 开始开发

安装 Docker 后，在仓库根目录执行：

```bash
docker compose run --build --rm dev
```

也可以用 VS Code 的 **Dev Containers: Reopen in Container**。容器使用 Python 3.11.17，包含 NumPy、SciPy、pandas、Matplotlib 和测试工具。版本、使用方法及环境维护说明见 [`environment/README.md`](environment/README.md)。

## 目录

| 目录 | 用途 |
|---|---|
| `environment/docker/` | 公共 Docker 镜像及构建文件 |
| `environment/dependencies/` | 依赖清单与版本锁定 |
| `configs/training/` | 统一训练场景配置 |
| `configs/evaluation/` | 故障注入、指标与评测配置 |
| `src/training/` | 训练流程与 GEMM 采集、重放 |
| `src/algorithms/` | 人工确认后的基线与新算法 |
| `src/backends/ascend/` | 华为 NPU 算子与设备接口 |
| `src/evaluation/` | 公共注入、统计与计时逻辑 |
| `scripts/` | 统一运行入口与辅助脚本 |
| `tests/` | 新框架的正确性与集成测试 |
| `docs/papers/` | 论文精读记录 |
| `docs/reproduction/` | 论文与实现对应关系及人工比对记录 |
| `experiments/` | 实验说明、可复现配置与小型结果摘要 |
| `data/` | 本地模型与数据，默认不提交 |
| `results/` | 本地实验输出，默认不提交 |
| [`archive/`](archive/README.md) | 停止使用的旧复现实验材料 |

空目录中的 `.gitkeep` 仅用于保留目录结构。

## 协作方式

1. **各自在自己的分支开发。** 可以持续提交和推送，分支名称自行决定。
2. **做好后再合并。** 非仓库拥有者不得直接推送 `main`；完成一批改动后提一次 PR，通过 CI 和一次审核后合并。PR 简单说明改动与验证结果，Issue 按需使用。
3. **公共基础环境不自行改动。** `environment/` 和 `configs/training/` 由 [@meiyou666](https://github.com/meiyou666) 统一维护，有调整需要先沟通。

核心算法仍须精读论文、人工比对和小规模验证后再扩大实验。AI 可以辅助，不能代替人工确认。

大型模型、数据和逐次输出放在 `data/`、`results/`，不提交到 Git；仓库保留必要配置和小型结果摘要。

## 环境一致性检查

CI 检查环境与训练配置的维护权限、依赖锁定和 Python 导入，并实际构建公共容器，在其中核对安装版本、运行测试和基础数值检查。

CPU 开发环境已冻结，可以开始算法与评测工具开发。NPU 相关版本仍待配置；当前 CI 不执行 NPU 训练。Codex 的项目约定和自动审查重点见 [`AGENTS.md`](AGENTS.md)。

## 历史材料

旧的 `ATTNChecker/`、`FT2/`、`LLM training SDC/` 和编辑器配置已完整移入 [`archive/legacy_reproduction/`](archive/legacy_reproduction/)。旧文件内容未改写，旧复现结果不作为当前已确认的实验基线。

新工作在上述公共目录中开展。测试收集范围限定为 `tests/`，归档目录不纳入新框架测试。
