# SDC

团队协作开展 SDC 统一评测、算法研究与系统实现的仓库。

当前仅初始化目录与协作入口。公共环境版本、训练配置和算法实现由后续任务逐项补充。

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

1. 使用任务 Issue 明确负责人、复核人、交付内容和完成标准。
2. 每项任务在独立分支开展，通过 Pull Request 合并到 `main`。
3. 公共环境与统一训练配置由 [@meiyou666](https://github.com/meiyou666) 维护；成员通过 PR 提出调整。
4. 核心算法先精读论文、人工比对和小规模验证，再扩大实验。AI 可以辅助，不能代替人工确认。
5. 实验记录绑定代码 commit、镜像版本与配置。大型模型、数据及逐次输出保存在 `data/`、`results/`，仓库保留必要的复现信息与小型摘要。

任务模板与 PR 模板已放在 `.github/`。环境目录和训练配置目录的默认维护人由 `CODEOWNERS` 标明；仓库保护规则由维护人另行管理。

## 历史材料

旧的 `ATTNChecker/`、`FT2/`、`LLM training SDC/` 和编辑器配置已完整移入 [`archive/legacy_reproduction/`](archive/legacy_reproduction/)。旧文件内容未改写，旧复现结果不作为当前已确认的实验基线。

新工作在上述公共目录中开展。测试收集范围限定为 `tests/`，归档目录不纳入新框架测试。
