# 作者源码与本次复现代码的对应说明

## 基线

作者源码固定在 commit `90510aec5d26850739cf583c97c65cbd71256cb1`。`upstream_reference/` 保留与本报告两种模型直接相关的 evaluation 与 protected-model 文件；完整上游仓库地址见根目录 `README.md`。

## 公开 artifact 中影响复现的差异

1. 论文描述越界值裁剪到最近边界、NaN 置零；提交仓库中的 `TensorCompare.cu` 路径会把越界值直接置零。报告主实验选择论文的 `paper_clamp`，仓库置零只作为次级兼容模式。
2. 论文评估 1-bit、2-bit、EXP；公开脚本没有给出一套统一、可重放的 2-bit 与 exponent-only 生成器。本复现把 FP16 位编号和 XOR 语义显式化，并持久化每个 fault spec。
3. 论文的 offline bounds 来自训练集 20%；公开仓库未提供完整 profiling artifact。本次为控制 4090 预算，每个模型–数据集对固定抽取 200 个校准样本求全局 extrema，因此 offline 结果不是论文 20% 规模的逐点复现。
4. 公开 protected 脚本的 bounds 生命周期依赖全局列表，跨样本运行时存在泄漏风险。本引擎在每次 inference 开始时清空 online bounds，并检查每个关键 site 恰有一对边界。
5. 论文写每个输入 500 次故障注入，而公开循环与发布数据不能无歧义恢复全部 500 次。本复现明确冻结为每格 10 prompts × 40 faults，并保存全部 6,000 个唯一 fault specs。
6. 作者评分代码的 recall<1 分支出现 `re2 == 0`（比较而不是赋值）。本报告把“归一化 reference-token recall 是否等于 1”作为二元 Masked/SDC 主判据，`1 - mean recall` 仅保留为次级连续指标。
7. 最初复现汇总额外使用 protected-clean evaluability gate，导致 240 次已完成运行被排除；论文的 Masked/SDC 定义没有该门槛。`reanalyze_author_logic.py` 不修改原始运行，只对全部 18,000 次结果重新评分。

## 本次新增模块

| 文件 | 作用 | 关键审计点 |
|---|---|---|
| `reproduction/src/ft2_formal/adapters.py` | 枚举 OPT/Qwen 所有线性投影并标注关键层 | site 数、关键 site 数、模块路径、采样权重固定 |
| `reproduction/src/ft2_formal/manifest.py` | 生成可重放的故障清单 | SHA-256 派生种子、PCG64、layer/neuron/bit 全记录 |
| `reproduction/src/ft2_formal/engine.py` | hook 注入、首 token 边界、clamp/NaN 修正 | 每 inference 恰一次 XOR、每 step/site 恰一次 hook、边界重置 |
| `reproduction/src/ft2_formal/decoding.py` | 固定长度贪心解码 | QA 60 token、GSM8K 180 token、不按 EOS 提前停止 |
| `reproduction/src/ft2_formal/main18k.py` | 主 campaign、续跑、完成审计 | 5×10×3×40×3=18,000，150 clean controls |
| `reproduction/analysis/reanalyze_author_logic.py` | 作者逻辑重评分 | 只读原始 artifacts、全部运行进入分母 |
| `reproduction/experiments/run_scaling_ablation_v2.py` | factor 1.0–2.5 消融 | 同一 400 个冻结 EXP specs 跨档配对 |
| `reproduction/experiments/run_critical_layer_identification.py` | 逐投影 omission | 全保护对照，其余投影仍保护，McNemar 配对检验 |
| `experiments/run_overhead_benchmark.py` | 时间/显存微基准 | 同 prompts 成对交叉顺序、同步计时、warm-up 排除 |
| `analysis/make_report_figures.py` | 结果可视化 | 只从提交的 CSV/JSON 读取数值 |

配置中的 `author_repository`、`paper`、`reproduction_frozen`、`reproduction_defined` 标签用于区分来源；这比把无法从作者 artifact 恢复的细节默认为“作者设置”更利于审计。
