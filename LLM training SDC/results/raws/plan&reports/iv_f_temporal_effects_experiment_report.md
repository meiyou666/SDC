# IV-F Duration Effects 复现实验报告

## 1. 实验定位

本报告只记录 IV-F temporal effects 中的 **fault duration** 部分，即固定一次故障从 step 500 开始触发，比较持续 1、3、5 个 update step 时最终训练结果是否随持续时间延长而退化。原先报告中的旧 rate 实验已经移除；rate 部分使用新的 all-HMMA BP8 口径单独成文。

本轮 duration 实验是早期精简复现：3 个 seed、3 档 duration，共 9 组 FI run，配套 3 组同 seed baseline。该实验使用旧 IV-F 固定单条 SASS instruction 口径，因此仅用于 duration 子问题，不和新 rate all-HMMA 结果混合统计。

## 2. 实验配置

共同训练配置如下：

| 项目 | 配置 |
|---|---|
| Model | `llama_60m` |
| Seeds | 42、1337、351344 |
| Training updates | 1,000 |
| LR schedule | warmup 1,000 steps，cosine schedule span 100,000 steps |
| Max LR | 1e-3 |
| Optimizer | AdamW，weight decay 0.01 |
| Precision | bfloat16 |
| Max length | 256 |
| Micro batch size | 32 |
| Gradient accumulation | 16 |
| Effective batch size | 512 |
| Gradient clipping | global norm 1.0 |
| Final evaluation | step 1,000 后执行一次 |

Fault injection 配置如下：

| 项目 | 配置 |
|---|---|
| Opcode filter | `TARGET_OP=HMMA` |
| Kernel filter | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` |
| Instruction index | `TARGET_INSTR=312` |
| Location | backward |
| Target register | `TARGET_REGISTER=1` |
| SM / lane | `TARGET_SMID=0`，`TARGET_LANEID=0` |
| Bit position / mask | 13 / 8192 |
| Trigger schedule | 固定从 step 500 开始 |
| Duration values | 1、3、5 updates |
| Recompute | disabled |

## 3. 数据完整性

数据来源：

| 类型 | 路径 |
|---|---|
| 单 run 汇总 | `logs/iv_f/iv_f_summary.csv` |
| 条件聚合 | `logs/iv_f/iv_f_aggregate.csv` |
| Duration checkpoints | `checkpoints/iv_f/duration/` |
| Baseline checkpoints | `checkpoints/iv_f/baseline/` |

完整性检查：

| 项目 | 结果 |
|---|---:|
| Baseline runs | 3 / 3 complete |
| Duration FI runs | 9 / 9 complete |
| Final checkpoint | 全部有 `model_1000` |
| Final eval | 全部完成 |
| Nonfinite metric count | 全部为 0 |
| Duration=1 trigger window | step 500 |
| Duration=3 trigger window | steps 500-502 |
| Duration=5 trigger window | steps 500-504 |

## 4. Baseline

| Seed | Final eval loss |
|---:|---:|
| 42 | 4.315345287 |
| 1337 | 4.322783470 |
| 351344 | 4.315844536 |
| Mean ± population std | 4.317991098 ± 0.003394843 |

后续使用同 seed paired delta：

```text
paired delta = FI final eval loss - same-seed baseline final eval loss
```

## 5. Duration 结果

逐 run 结果如下：

| Duration | Seed | Final eval loss | Paired delta | Parameter L2 diff | Trigger window | Detected | Nonfinite |
|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 42 | 4.314446926 | -0.000898361 | 16.564802 | 500-500 | 0 | 0 |
| 1 | 1337 | 4.323482513 | +0.000699043 | 16.584884 | 500-500 | 0 | 0 |
| 1 | 351344 | 4.316143990 | +0.000299454 | 11.885071 | 500-500 | 0 | 0 |
| 3 | 42 | 4.315045834 | -0.000299454 | 16.344428 | 500-502 | 0 | 0 |
| 3 | 1337 | 4.323282719 | +0.000499249 | 17.595081 | 500-502 | 0 | 0 |
| 3 | 351344 | 4.315844536 | +0.000000000 | 17.436341 | 500-502 | 0 | 0 |
| 5 | 42 | 4.315045834 | -0.000299454 | 15.503151 | 500-504 | 0 | 0 |
| 5 | 1337 | 4.323582172 | +0.000798702 | 17.094664 | 500-504 | 0 | 0 |
| 5 | 351344 | 4.316742897 | +0.000898361 | 12.567746 | 500-504 | 0 | 0 |

按 duration 聚合：

| Duration | Mean final eval | Population std | Mean paired delta | Mean parameter L2 diff | Mean max grad pre | Mean max attn logits | Mean max Rt |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.318024476 | 0.003921108 | +0.000033379 | 15.011586 | 2.302083 | 44.333333 | 0.000656128 |
| 3 | 4.318057696 | 0.003709010 | +0.000066598 | 17.125283 | 2.302083 | 51.333333 | 0.000657399 |
| 5 | 4.318456968 | 0.003689697 | +0.000465870 | 15.055187 | 2.302083 | 43.750000 | 0.000657399 |

## 6. 现象分析

从 final eval loss 看，duration 的均值方向与论文一致：

```text
steps_1: +0.000033
steps_3: +0.000067
steps_5: +0.000466
```

也就是说，平均 paired delta 随 duration 从 1 到 5 增大。但这个趋势很弱：最大均值退化只有约 `4.66e-4`，明显小于 baseline 跨 seed 波动，也远小于论文 duration=5 大约 `+0.02` 的量级。

参数 L2 差异证明故障确实改变了训练轨迹。所有 duration FI run 的最终参数差都非零，范围为 `11.8851` 到 `17.5951`。不过参数差不随 duration 单调增加，duration=3 的均值最高，duration=5 反而回落。

训练期指标也没有显示论文描述的强 spike：各 duration 组的 `max_gradient_norm_pre` 均值都等于 baseline 的 `2.302083`，`max_rt` 与 baseline 也几乎一致。`max_attention_logits` 在 duration=3 时较高，但没有形成 1、3、5 的单调趋势。

检测器没有响应：9 个 duration run 的 detected count 全部为 0。这个结果不说明没有注入，因为最终参数差非零；它说明当前固定单条 instruction 的 duration 扰动没有达到 `rt` 检测阈值。

## 7. 结论

本轮 duration 实验工程执行完整，所有 run 都正常结束，trigger window 正确，且 FI run 均产生非零参数偏移。

科学结论上，只能给出弱支持：duration 从 1、3 到 5 时，平均 final eval loss delta 方向上递增，但幅度很小，参数差、gradient norm、attention logits 和 Rt 都没有形成稳定单调关系。因此本轮 duration 结果不能单独证明论文中“fault duration 越长，训练退化越强”的强结论，只能作为固定单指令口径下的弱趋势记录。

本结果不再与旧 rate 子实验合并成一个总判定。新的 rate 子实验已使用 `BP8 + bit13 + TARGET_INSTR=-1` all-HMMA 口径单独分析。

## 8. 证据文件

| 类型 | 路径 |
|---|---|
| Duration logs | `logs/iv_f/duration/` |
| Baseline logs | `logs/iv_f/baseline/` |
| Summary CSV | `logs/iv_f/iv_f_summary.csv` |
| Aggregate CSV | `logs/iv_f/iv_f_aggregate.csv` |
| Duration checkpoints | `checkpoints/iv_f/duration/` |
| Baseline checkpoints | `checkpoints/iv_f/baseline/` |
| Config | `scripts/configs/iv_f.sh` |
| Runner | `scripts/experiments/run_iv_f_temporal.sh` |
| Summary script | `scripts/experiments/summarize_iv_f.py` |
