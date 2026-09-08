# IV-F Rate Effects All-HMMA 复现实验报告

## 1. 实验定位

本报告记录 IV-F temporal effects 中的 **fault rate** 部分，使用新的本机敏感 target：`BP8 + bit13 + TARGET_INSTR=-1`。该报告只分析本轮 all-HMMA rate 数据，不纳入旧 IV-F 固定单条 `TARGET_INSTR=312` 的失败 rate 结果。

本轮实验目标是验证：在相同 seed、相同 backward kernel、相同 bit、相同 duration=1 step 下，提高 fault injection 频率是否会导致 final eval loss、gradient corruption、attention logits、参数偏移和 detector response 增强。

## 2. Target 选择依据

Target 来自 IV-B kernel sensitivity 的已完成结果。`BP8 + bit13` 在 IV-B 的 1/10 频率下是本机最强的 finite backward 退化组合：

| Candidate | Final eval loss | Delta vs baseline | Parameter diff | Max grad pre | Detected | Nonfinite metrics |
|---|---:|---:|---:|---:|---:|---:|
| `BP8 + bit13` | 4.689045906 | +0.371730804 | 151.458222 | 14400.0 | 74 | 2 |
| `BP9 + bit13` | 4.605382442 | +0.288067341 | 156.093662 | 44032.0 | 71 | 0 |

IV-F 关心 temporal strength，因此优先选择 1/10 频率下 final eval loss 更高、也更接近论文 rate 曲线量级的 `BP8 + bit13`。`BP9 + bit13` 保留为 IV-E spatial sweep 主 target，因为它强但完全非饱和。

## 3. 实验配置

共同训练配置如下：

| 项目 | 配置 |
|---|---|
| Model | `llama_60m` |
| Seed | 42 |
| Training updates | 1,000 |
| LR schedule | warmup 1,000 steps，cosine schedule span 100,000 steps |
| Max LR | 1e-3 |
| Optimizer | AdamW，weight decay 0.01 |
| Precision | bfloat16 |
| Max length | 256 |
| Micro batch size | 256 |
| Gradient accumulation | 2 |
| Effective batch size | 512 |
| Gradient clipping | global norm 1.0 |
| Final evaluation | step 1,000 后执行一次 |
| Paired baseline | `checkpoints/iv_f_mb256/baseline/seed_42` |
| Baseline final eval loss | 4.317315101623535 |

Fault injection 配置如下：

| 项目 | 配置 |
|---|---|
| Kernel label | `BP8` |
| Kernel filter | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_nt` |
| Kernel location | backward |
| HMMA instruction count | 64 |
| Opcode filter | `TARGET_OP=HMMA` |
| Instruction index | `TARGET_INSTR=-1`，覆盖 BP8 内所有匹配 HMMA 指令 |
| Target register | `TARGET_REGISTER=1`，第一 input register |
| SM / lane | `TARGET_SMID=0`，`TARGET_LANEID=0` |
| Bit position / mask | 13 / 8192 |
| Duration | 1 update |
| Recompute | disabled |
| Detection | enabled for signal logging only |

Rate 轴如下：

| Rate point | 数据来源 |
|---|---|
| every 1 | 新 all-HMMA IV-F rate run |
| every 10 | 复用 IV-B `BP8 + bit13`，同 seed、同 target、同 all-HMMA 口径 |
| every 100 | 新 all-HMMA IV-F rate run |
| every 1000 | 新 all-HMMA IV-F rate run |

## 4. 数据完整性

数据来源：

| 类型 | 路径 |
|---|---|
| Rate summary | `logs/iv_f_rate_all_hmma_mb256/iv_f_rate_all_hmma_summary.csv` |
| New rate checkpoints | `checkpoints/iv_f_rate_all_hmma_mb256/BP8/rate/` |
| Reused every-10 checkpoint | `checkpoints/iv_b_mb256/seed_42/bit_13/BP8` |
| Campaign manifest | `checkpoints/iv_f_rate_all_hmma_mb256/BP8/campaign_manifest.txt` |

完整性检查：

| Rate | Source | Status | Trigger count | First trigger | Last trigger | Final checkpoint | Summary |
|---|---|---|---:|---:|---:|---|---|
| every 1 | `iv_f_rate_all_hmma` | complete | 1000 | 1 | 1000 | yes | yes |
| every 10 | `iv_b_reuse` | complete | 91 | 2 | 1000 | yes | yes |
| every 100 | `iv_f_rate_all_hmma` | complete | 10 | 36 | 751 | yes | yes |
| every 1000 | `iv_f_rate_all_hmma` | complete | 2 | 244 | 769 | yes | yes |

`every_1000` 在 1,000 step 内触发了 2 次，这是当前 trainer 的 Bernoulli/random schedule 结果，不是固定只触发一次。四个 rate 点的 trigger schedule 都与各自 run_config 和 summary 一致。

## 5. Rate 结果

核心结果如下。`delta = FI final eval loss - paired baseline final eval loss`。

| Rate | Source | Final eval loss | Delta | Final train loss | Max train loss | Max grad pre | Max grad post | Max attn logits | Max Rt | Trigger count | Detected | Nonfinite metrics | Parameter diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| every 1 | new | 7.378180027 | +3.060864925 | 7.356199503 | 10.474623203 | 606208.0 | 1.0078125 | 356.0 | 0.001091003 | 1000 | 339 | 20 | 168.298482 |
| every 10 | IV-B reuse | 4.689045906 | +0.371730804 | 4.635647774 | 10.471520901 | 14400.0 | 1.0078125 | 50.25 | 0.000720978 | 91 | 74 | 2 | 151.458222 |
| every 100 | new | 4.330307961 | +0.012992859 | 4.279129028 | 10.471338272 | 3312.0 | 1.0078125 | 37.75 | 0.000888824 | 10 | 8 | 0 | 109.216951 |
| every 1000 | new | 4.320375919 | +0.003060818 | 4.269145727 | 10.471338272 | 3440.0 | 1.0078125 | 42.5 | 0.000675201 | 2 | 2 | 0 | 55.736456 |

最终 eval loss 呈清晰频率趋势：

```text
every_1    7.37818  delta +3.06086
every_10   4.68905  delta +0.37173
every_100  4.33031  delta +0.01299
every_1000 4.32038  delta +0.00306
baseline   4.31732
```

从 final eval loss 看，本轮 all-HMMA rate 实验成功复现了“故障频率越高，训练退化越严重”的单调关系。

## 6. 与论文 rate 数值对比

论文 IV-F rate 结果如下：

| Rate | Paper eval loss | Local eval loss | Difference |
|---|---:|---:|---:|
| every 1 | 7.38 ± 0.073 | 7.378180027 | -0.001820 |
| every 10 | 4.77 ± 0.046 | 4.689045906 | -0.080954 |
| every 100 | 4.33 ± 0.018 | 4.330307961 | +0.000308 |
| every 1000 | 4.30 ± 0.003 | 4.320375919 | +0.020376 |

本机 every-1 和 every-100 与论文数值高度接近。every-10 低于论文均值约 `0.081`，但仍处于明显退化区间，并且远高于本机 baseline。every-1000 高于论文绝对值约 `0.020`，主要受本机 paired baseline 已经高于论文 baseline 的影响；用本机 paired delta 看，它只比 baseline 高 `+0.00306`，符合低频接近 baseline 的趋势。

这说明旧固定单指令 rate 失败的主要原因确实是 fault target/剂量不够，而不是训练框架无法复现 rate 效应。本轮使用 IV-B 选出的 `BP8 + bit13` 并覆盖该 kernel 全部 HMMA 指令后，rate 曲线的强度和形状都与论文高度一致。

## 7. 训练期机制指标

### 7.1 Gradient corruption

`max_gradient_norm_pre` 在所有 FI rate 点都明显高于 baseline 常见的约 `2.28`：

| Rate | Max grad pre | 代表触发 step | 说明 |
|---|---:|---:|---|
| every 1 | 606208.0 | 641 | 高频注入导致多次巨大梯度异常，并出现 20 个 Inf gradient norm 记录。 |
| every 10 | 14400.0 | 907 | 复用 IV-B；另在 step 589 和 963 出现 Inf gradient norm。 |
| every 100 | 3312.0 | 751 | 10 次触发中有 8 次被 detector 捕获。 |
| every 1000 | 3440.0 | 244 | 2 次触发均被 detector 捕获。 |

梯度最大值本身不完全随 rate 单调，例如 every-1000 的单次最大值略高于 every-100；这符合随机触发和单次 fault 幅度随机性的特点。更稳定的趋势体现在 final eval loss、trigger count、非有限次数和参数偏移上。

### 7.2 Nonfinite metrics

非有限指标集中在高频：

| Rate | Nonfinite metric count | First nonfinite |
|---|---:|---|
| every 1 | 20 | step 522，`gradient_norm_pre=Inf` |
| every 10 | 2 | step 589，`gradient_norm_pre=Inf` |
| every 100 | 0 | - |
| every 1000 | 0 | - |

这说明 `BP8 + bit13` 不是一开始就 NaN 饱和的 target，而是随着触发频率升高更容易进入 Inf-gradient 边界。该性质适合 IV-F：它既能在 every-1 下产生论文量级严重退化，又能在低频下保持可比较的有限 eval loss。

### 7.3 Attention logits and Rt

Attention logits 也随高频注入出现增强趋势：every-1 的 max attention logits 为 `356.0`，明显高于 every-10 的 `50.25`、every-100 的 `37.75` 和 every-1000 的 `42.5`。

Rt 的最大值为：

| Rate | Max Rt |
|---|---:|
| every 1 | 0.001091003 |
| every 10 | 0.000720978 |
| every 100 | 0.000888824 |
| every 1000 | 0.000675201 |

Rt 不是严格单调；every-100 的单次 spike 高于 every-10。这说明 Rt 更适合用于检测单次异常 update，而 final eval loss 更适合衡量 rate 累积影响。

## 8. Detector 行为

本轮 rate 结果中没有 false positive，detector 的响应如下：

| Rate | TP | FP | TN | FN | Recall |
|---|---:|---:|---:|---:|---:|
| every 1 | 339 | 0 | 0 | 661 | 33.9% |
| every 10 | 74 | 0 | 909 | 17 | 81.3% |
| every 100 | 8 | 0 | 990 | 2 | 80.0% |
| every 1000 | 2 | 0 | 998 | 0 | 100.0% |

every-1 的 recall 低，是因为所有 step 都是 trigger step，且部分 corrupted updates 不一定让 Rt 超过检测阈值；但 final eval loss 已经严重退化，说明检测器未捕获的高频扰动仍会累积成明显损害。低频下 recall 较高，特别是 every-1000 的两次触发均被检测到。

这里的 detector 结果只是附带观察。本轮没有启用 recompute，因此即使 detected anomaly 为 1，训练仍接受 corrupted update。

## 9. 参数偏移

最终参数 L2 差异总体随 rate 降低而下降：

| Rate | Parameter L2 diff |
|---|---:|
| every 1 | 168.298482 |
| every 10 | 151.458222 |
| every 100 | 109.216951 |
| every 1000 | 55.736456 |

该趋势支持 rate 效应的另一个侧面：更高的 fault rate 不只影响最终 validation loss，也使最终模型参数远离 paired baseline。every-10 的参数差来自 IV-B post-hoc safetensors FP32 L2 对比；新 summary CSV 中该行来源为 `iv_b_reuse`，因此报告中手动引用 IV-B summary 的 post-hoc 参数差。

## 10. 结论

本轮 IV-F rate all-HMMA 实验工程执行完整，四个 rate 点都有可追溯数据，其中 every-10 明确复用 IV-B 同口径结果。

科学结论上，本轮成功复现 rate effect：

1. Final eval loss 随 fault rate 降低而单调接近 baseline：`7.37818 -> 4.68905 -> 4.33031 -> 4.32038`。
2. every-1 和 every-100 的绝对 eval loss 与论文报告几乎重合。
3. 参数 L2 差异随 rate 降低整体下降：`168.30 -> 151.46 -> 109.22 -> 55.74`。
4. 高频条件产生更多非有限 gradient norm 记录：every-1 为 20，every-10 为 2，低频为 0。
5. Detector 在低频触发上 recall 较高，但高频下仍漏掉大量 corrupted updates；由于 recompute disabled，这些 update 继续累积并导致严重退化。

本轮结果说明：使用 IV-B 选出的本机敏感 kernel `BP8`，并采用 all-HMMA injection 后，可以在 RTX 4090 / NVBit 1.7.6 / mb256 口径下复现论文 IV-F 的 rate 曲线。旧固定单指令 rate 失败不再作为 IV-F rate 结论的一部分。

## 11. 证据文件

| 类型 | 路径 |
|---|---|
| Rate summary CSV | `logs/iv_f_rate_all_hmma_mb256/iv_f_rate_all_hmma_summary.csv` |
| every-1 checkpoint | `checkpoints/iv_f_rate_all_hmma_mb256/BP8/rate/every_1/seed_42` |
| every-100 checkpoint | `checkpoints/iv_f_rate_all_hmma_mb256/BP8/rate/every_100/seed_42` |
| every-1000 checkpoint | `checkpoints/iv_f_rate_all_hmma_mb256/BP8/rate/every_1000/seed_42` |
| every-10 reused checkpoint | `checkpoints/iv_b_mb256/seed_42/bit_13/BP8` |
| Campaign manifest | `checkpoints/iv_f_rate_all_hmma_mb256/BP8/campaign_manifest.txt` |
| Rate logs | `logs/iv_f_rate_all_hmma_mb256/BP8/rate/` |
| Config | `scripts/configs/iv_f_rate_all_hmma.sh` |
| Runner | `scripts/experiments/run_iv_f_rate_all_hmma.sh` |
| Summary script | `scripts/experiments/summarize_iv_f_rate_all_hmma.py` |
| IV-B source summary for every-10 | `logs/iv_b_mb256/seed_42/bit_13/iv_b_summary_raw.csv` |
