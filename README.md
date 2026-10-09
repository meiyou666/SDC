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

1. **各自在自己的分支开发。** 可以持续提交和推送，分支名称自行决定。
2. **做好后再合并。** 非仓库拥有者不得直接推送 `main`；完成一批改动后提一次 PR，通过 CI 和一次审核后合并。PR 简单说明改动与验证结果，Issue 按需使用。
3. **公共基础环境不自行改动。** `environment/` 和 `configs/training/` 由 [@meiyou666](https://github.com/meiyou666) 统一维护，有调整需要先沟通。

核心算法仍须精读论文、人工比对和小规模验证后再扩大实验。AI 可以辅助，不能代替人工确认。

大型模型、数据和逐次输出放在 `data/`、`results/`，不提交到 Git；仓库保留必要配置和小型结果摘要。

## 环境一致性检查

CI 检查公共环境与训练配置是否被擅自改动、依赖是否固定，以及 Python 显式导入是否属于标准库、本项目或锁定的依赖。

当前环境基线为 `pending`，允许文档和结构调整；实验代码需要先由维护人填写镜像与版本，并将基线设为 `frozen`。填写入口见 [`environment/README.md`](environment/README.md)。

CI 做配置和静态依赖检查，不在 GitHub 执行 NPU 训练，也不替代真实服务器上的驱动、固件和运行结果检查。

## 历史材料

旧的 `ATTNChecker/`、`FT2/`、`LLM training SDC/` 和编辑器配置已完整移入 [`archive/legacy_reproduction/`](archive/legacy_reproduction/)。旧文件内容未改写，旧复现结果不作为当前已确认的实验基线。

新工作在上述公共目录中开展。测试收集范围限定为 `tests/`，归档目录不纳入新框架测试。
