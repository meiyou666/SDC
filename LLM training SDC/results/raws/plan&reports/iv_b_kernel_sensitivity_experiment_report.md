# IV-B Kernel Sensitivity 复现实验报告

## 1. 实验定位

本报告记录 Section IV-B kernel sensitivity 的正式复现实验结果。IV-B 要回答的问题是：在相同训练配置、相同 seed、相同 bit、相同触发 schedule 下，把 fault injection 限定到不同 GEMM/HMMA kernel 时，训练退化、NaN 传播、梯度异常、attention-logit spike 和参数偏移是否存在明显差异。

本轮 IV-B 采用本机 profiling 得到的 kernel manifest，而不是强行套用论文中的 kernel 数量或标签。当前 manifest 包含 4 个 forward kernel（`FP1`-`FP4`）和 9 个 backward kernel（`BP1`-`BP9`），共 13 个 kernel。每个 bit 对 13 个 kernel 各跑一次，因此总计 39 组 fault-injection 训练。

## 2. 实验配置

训练配置继承 IV-A mb256 主线：

| 项目 | 配置 |
|---|---|
| Model | `llama_60m`，约 58.07M parameters |
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
| Final evaluation | training span 后评估一次 |
| Paired baseline final eval loss | 4.317315101623535 |

Fault injection 配置如下：

| 项目 | 配置 |
|---|---|
| NVBit version | 1.7.6 |
| Opcode filter | `TARGET_OP=HMMA` |
| Kernel filter | 来自 `logs/iv_a_mb256/kernel_manifest.tsv` 的 `target_func_contains` |
| Instruction index | `TARGET_INSTR=-1`，覆盖该 kernel 内所有匹配 HMMA 指令 |
| Target register | `TARGET_REGISTER=1`，第一 input register |
| SM / lane | `TARGET_SMID=0`，`TARGET_LANEID=0` |
| Trigger rate | 平均每 10 个 update 触发一次 |
| Duration | 1 update |
| Recompute | disabled |
| Detection | enabled for signal logging only |

本轮 bit 集合为 `10, 13, 14`：

| Bit | Bitmask | 选择理由 |
|---:|---:|---|
| 10 | 1024 | 本机 IV-A 中出现局部梯度尖峰和较大参数偏移的较弱 exponent bit。 |
| 13 | 8192 | 论文中多处使用的敏感 exponent bit；本机 IV-B 中也产生最清楚的非 NaN backward kernel sensitivity。 |
| 14 | 16384 | 本机 IV-A 已确认可导致 NaN 的强阳性 exponent bit，用于验证灾难性传播路径。 |

## 3. 数据完整性

输入汇总文件：

| Bit | Summary |
|---:|---|
| 10 | `logs/iv_b_mb256/seed_42/bit_10/iv_b_summary_raw.csv` |
| 13 | `logs/iv_b_mb256/seed_42/bit_13/iv_b_summary_raw.csv` |
| 14 | `logs/iv_b_mb256/seed_42/bit_14/iv_b_summary_raw.csv` |

完整性检查结果：

| 项目 | 结果 |
|---|---:|
| Total runs | 39 |
| Complete runs | 39 |
| Missing / failed runs | 0 |
| Kernels per bit | 13 |
| Forward kernels | 4 |
| Backward kernels | 9 |
| Trigger count per run | 91 |
| First trigger step | 2 |
| Last trigger step | 1000 |
| Trigger schedule ok | 39 / 39 |

注意：`logs/iv_b_mb256/seed_42/bit_14/iv_b_summary_current.csv` 是中间状态文件，曾显示 BP5-BP9 missing；正式分析应使用 `iv_b_summary_raw.csv`，其中 bit14 的 13 个 kernel 全部 complete。

## 4. 总体结果

按 bit 和 forward/backward 分组统计如下。`delta = FI final eval loss - paired baseline final eval loss`。NaN run 不参与 finite delta 均值。

| Bit | Location | Runs | Finite | NaN final eval | Avg finite delta | Max finite delta | Min finite delta | Avg finite param diff | Max finite param diff | Detected sum | Nonfinite metric sum |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | forward | 4 | 4 | 0 | 0.006844878 | 0.024280071 | -0.000554085 | 73.266103 | 92.072876 | 0 | 0 |
| 10 | backward | 9 | 9 | 0 | 0.009901577 | 0.064957619 | -0.001084328 | 73.570221 | 114.726624 | 89 | 0 |
| 13 | forward | 4 | 4 | 0 | 0.000280261 | 0.001207829 | -0.000641823 | 58.670107 | 65.030963 | 0 | 0 |
| 13 | backward | 9 | 8 | 1 | 0.109418690 | 0.371730804 | -0.000268459 | 95.421431 | 156.093662 | 699 | 9519 |
| 14 | forward | 4 | 0 | 4 | - | - | - | - | - | 3996 | 99898 |
| 14 | backward | 9 | 2 | 7 | 0.061376572 | 0.061376572 | 0.061376572 | 94.632904 | 94.632904 | 7175 | 174874 |

总体结论很明确：

1. `bit13` 的 backward kernel sensitivity 最有分析价值：同一 bit 下，不同 backward kernel 从几乎无影响到 NaN、从 `+0.0006` 到 `+0.3717` 的 eval loss delta 都出现了。
2. `bit14` 是强阳性/饱和 bit：forward 全 NaN，backward 大多数 NaN，适合证明注入链路和灾难性传播，但不适合做细粒度 kernel 排序。
3. `bit10` 是弱到中等强度 bit：没有 NaN；少数 kernel 产生明显 loss 或 gradient spike，适合作为非饱和弱故障对照。
4. Forward faults 主要表现为 attention/logit 或短时 loss spike；Backward faults 更容易转化为 gradient corruption、detector response、final eval loss 退化和参数偏移。

## 5. 有限退化排序

下表列出 final eval loss 有限且 delta 为正的主要组合，按 eval delta 从大到小排序。

| Rank | Bit | Kernel | Location | Final eval loss | Delta | Param diff | Max grad pre | Max attn logits | Detected | Nonfinite metrics |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | BP8 | backward | 4.689045906 | 0.371730804 | 151.458222 | 14400.0 | 50.25 | 74 | 2 |
| 2 | 13 | BP9 | backward | 4.605382442 | 0.288067341 | 156.093662 | 44032.0 | 97.0 | 71 | 0 |
| 3 | 13 | BP6 | backward | 4.469129562 | 0.151814461 | 149.018848 | 223.0 | 46.0 | 43 | 0 |
| 4 | 10 | BP4 | backward | 4.382272720 | 0.064957619 | 114.726624 | 177209344.0 | 53.75 | 89 | 0 |
| 5 | 13 | BP5 | backward | 4.379658699 | 0.062343597 | 95.883597 | 2.350316e+16 | 49.0 | 87 | 0 |
| 6 | 14 | BP1 | backward | 4.378691673 | 0.061376572 | 94.632904 | 2.296875 | 44.0 | 91 | 91 |
| 7 | 14 | BP2 | backward | 4.378691673 | 0.061376572 | 94.632904 | 2.296875 | 44.0 | 91 | 91 |
| 8 | 10 | FP4 | forward | 4.341595173 | 0.024280071 | 83.429768 | 2.28125 | 45.25 | 0 | 0 |
| 9 | 10 | BP8 | backward | 4.328261852 | 0.010946751 | 77.664002 | 2.28125 | 44.0 | 0 | 0 |
| 10 | 10 | BP9 | backward | 4.326557159 | 0.009242058 | 80.434339 | 2.984375 | 51.5 | 0 | 0 |
| 11 | 10 | FP1 | forward | 4.321304798 | 0.003989697 | 92.072876 | 2.28125 | 540.0 | 0 | 0 |

从排序看，最强的非 NaN 退化集中在 `bit13` 的 backward kernels：`BP8`、`BP9`、`BP6`。其中 `BP8` 的 final eval loss 最高，`BP9` 的 parameter difference 最大且完全没有非有限指标，二者分别适合 IV-F 和 IV-E 的后续 target 选择：

- `BP8 + bit13`：最高 finite eval loss，适合 IV-F rate/duration 中追求足够强的 temporal signal。
- `BP9 + bit13`：强但 `nonfinite_metric_count=0`，适合 IV-E lane/SM sweep 的连续可比分析。

## 6. NaN 与非有限传播

NaN final eval loss 的组合如下：

| Bit | Kernel | Location | Detected count | Nonfinite metric count | 解释 |
|---:|---|---|---:|---:|---|
| 13 | BP7 | backward | 424 | 9517 | bit13 下唯一 final eval NaN 的 backward kernel，说明 BP7 对 bit13 明显过敏。 |
| 14 | FP1 | forward | 999 | 24976 | bit14 forward 强饱和，第一次注入后持续 NaN/Inf。 |
| 14 | FP2 | forward | 999 | 24976 | 同上。 |
| 14 | FP3 | forward | 999 | 24973 | 同上。 |
| 14 | FP4 | forward | 999 | 24973 | 同上。 |
| 14 | BP3 | backward | 999 | 24956 | bit14 backward 多数 kernel 直接饱和。 |
| 14 | BP4 | backward | 999 | 24956 | 同上。 |
| 14 | BP5 | backward | 999 | 24956 | 同上。 |
| 14 | BP6 | backward | 999 | 24956 | 同上。 |
| 14 | BP7 | backward | 999 | 24956 | 同上。 |
| 14 | BP8 | backward | 999 | 24956 | 同上。 |
| 14 | BP9 | backward | 999 | 24956 | 同上。 |

bit14 的结果不应作为细粒度 kernel sensitivity 排序证据。它在多数 kernel 上进入同一类 NaN 饱和路径，detector 在非触发 step 也持续报 anomaly，因此高 detected count 更多表示训练状态已被污染，而不是每个 kernel 的 detector 边界更敏感。

BP1/BP2 是 bit14 backward 中的特殊非 NaN 情况：final eval loss 均为 4.378691673，delta 均为 `+0.061376572`，detected count 均为 91，nonfinite metric count 均为 91。此前短验证报告已经确认 BP1/BP2 的 kernel filter 和 HMMA idx 集合不同，但触发 step 都使 aggregate gradient norm 进入 `Inf`，再经 clipping 后产生相同的数值轨迹。因此，BP1/BP2 在 bit14/all-HMMA 口径下更适合作为 clipping interaction 的强阳性证据，而不是 kernel 排名证据。

## 7. Bit 10 分析：弱到中等强度、非饱和

bit10 的 13 个 run 全部 final eval finite，没有非有限指标。总体上 bit10 不会造成普遍退化，但能在特定 kernel 上产生可观扰动。

Forward 侧：

| Kernel | Final eval loss | Delta | Param diff | Max training loss | Max attn logits | Detected |
|---|---:|---:|---:|---:|---:|---:|
| FP1 | 4.321304798 | 0.003989697 | 92.072876 | 10.471338 | 540.0 | 0 |
| FP2 | 4.316761017 | -0.000554085 | 67.224704 | 10.471521 | 3920.0 | 0 |
| FP3 | 4.316978931 | -0.000336170 | 50.337063 | 10.471521 | 39.75 | 0 |
| FP4 | 4.341595173 | 0.024280071 | 83.429768 | 10.619143 | 45.25 | 0 |

`FP4 + bit10` 是 forward 侧最清楚的 loss-spike 样例：最大 training loss 为 `10.619143`，出现在 step 2，也就是第一次触发 step；最终 eval delta 为 `+0.02428`。但 gradient norm、rt 和 detector 都没有明显响应，说明这种 forward fault 更像局部输出扰动，不一定通过梯度链路放大为持续异常。

Backward 侧：

| Kernel | Final eval loss | Delta | Param diff | Max grad pre | Detected |
|---|---:|---:|---:|---:|---:|
| BP4 | 4.382272720 | 0.064957619 | 114.726624 | 177209344.0 | 89 |
| BP8 | 4.328261852 | 0.010946751 | 77.664002 | 2.28125 | 0 |
| BP9 | 4.326557159 | 0.009242058 | 80.434339 | 2.984375 | 0 |
| BP1 | 4.320097446 | 0.002782345 | 74.168665 | 2.28125 | 0 |

`BP4 + bit10` 是 bit10 的主要 backward 强信号：第一次触发 step 2 即出现 `gradient_norm_pre=177209344.0`，detector 从 step 12 到 1000 共响应 89 次，最终 eval delta 为 `+0.06496`。这说明较低 exponent bit 在特定 backward kernel 上也能造成明显梯度污染，但这种现象不是 bit10 的普遍行为。

## 8. Bit 13 分析：最有价值的 kernel sensitivity 主结果

bit13 的 forward 结果整体稳定，backward 结果分化强，是本轮 IV-B 的核心发现。

Forward 侧：

| Kernel | Final eval loss | Delta | Param diff | Max attn logits | Detected |
|---|---:|---:|---:|---:|---:|
| FP1 | 4.318522930 | 0.001207829 | 55.850217 | 44.0 | 0 |
| FP2 | 4.317298412 | -0.000016689 | 51.503240 | 2.559486e+20 | 0 |
| FP3 | 4.317886829 | 0.000571728 | 62.296010 | 56.75 | 0 |
| FP4 | 4.316673279 | -0.000641823 | 65.030963 | 53.0 | 0 |

`FP2 + bit13` 出现极端 attention-logit spike：`max_attention_logits=2.559486e+20`，最大值出现在 step 943。但 final eval delta 基本为 0，detector 没有响应，gradient norm 也保持正常。这支持 IV-C 的机制解释：forward attention logits 可以被 SDC 瞬时打到极端值，但不一定转化为长期训练退化。

Backward 侧：

| Kernel | Final eval loss | Delta | Param diff | Max grad pre | Max attn logits | Detected | Nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP1 | 4.317046642 | -0.000268459 | 49.247874 | 2.28125 | 42.5 | 0 | 0 |
| BP2 | 4.317900658 | 0.000585556 | 51.528348 | 2.28125 | 53.0 | 0 | 0 |
| BP3 | 4.318227291 | 0.000912189 | 55.290170 | 2.28125 | 47.0 | 0 | 0 |
| BP4 | 4.317479134 | 0.000164032 | 54.850727 | 3.15625 | 48.5 | 0 | 0 |
| BP5 | 4.379658699 | 0.062343597 | 95.883597 | 2.350316e+16 | 49.0 | 87 | 0 |
| BP6 | 4.469129562 | 0.151814461 | 149.018848 | 223.0 | 46.0 | 43 | 0 |
| BP7 | NaN | NaN | NaN | 1.820791e+15 | 51.0 | 424 | 9517 |
| BP8 | 4.689045906 | 0.371730804 | 151.458222 | 14400.0 | 50.25 | 74 | 2 |
| BP9 | 4.605382442 | 0.288067341 | 156.093662 | 44032.0 | 97.0 | 71 | 0 |

这里的 kernel sensitivity 很强：`BP1`-`BP4` 基本接近 baseline；`BP5`-`BP6` 开始明显退化；`BP7` 进入 NaN；`BP8` 和 `BP9` 是最强 finite 退化组合。该结果也是后续 target 选择的主要依据：

- IV-E 选 `BP9 + bit13`：delta 足够大（`+0.2881`），Parameter Difference 最大（`156.09`），且完全没有非有限指标，适合 lane/SM sweep。
- IV-F 选 `BP8 + bit13`：final eval loss 最高（`4.6890`），更接近论文 IV-F 1/10 频率下的 loss 量级，适合 temporal strength 主线。
- `BP6 + bit13` 可作为 fallback：强度低于 BP8/BP9，但仍有 `+0.1518` delta 且 nonfinite 为 0。

## 9. Bit 14 分析：强阳性但大面积饱和

bit14 是本机确认的强阳性 exponent bit。IV-B 中它表现为大面积 NaN 饱和：

| Location | Runs | NaN final eval | Finite final eval | 主要现象 |
|---|---:|---:|---:|---|
| Forward | 4 | 4 | 0 | 第一次触发 step 后持续 NaN/Inf，detected count 999。 |
| Backward | 9 | 7 | 2 | BP3-BP9 全 NaN；BP1/BP2 为 Inf-gradient but finite eval。 |

bit14 forward 的代表 `FP1`：step 2 即出现 loss null、perplexity NaN、gradient norm NaN、attention logits Inf、rt NaN；detector 从 step 2 到 step 1000 持续为 1，共 999 次。它说明 fault 已污染训练状态，不再是单个触发 step 的局部扰动。

bit14 backward 的 `BP1/BP2`：每个触发 step 的 `gradient_norm_pre` 为 Inf，`gradient_norm_post` 在触发 step 被 clipping 压到 0；最终 eval delta 为 `+0.06138`。这说明强 backward fault 与 global norm clipping 之间存在明显交互，适合作为 IV-D no-clipping 对照的强阳性样例。

因此 bit14 的报告定位应是：

1. 证明注入链路可以触发灾难性 SDC。
2. 支撑 NaN propagation 与 detector 持续响应现象。
3. 支撑 clipping interaction 的后续实验选择。
4. 不用于 IV-B 的细粒度 kernel sensitivity 排名。

## 10. Forward vs Backward 对比

Forward 和 backward 的差异可以从三个指标看出。

第一，final eval loss：

- bit10 forward 最大 finite delta 为 `FP4 +0.02428`，backward 最大 finite delta 为 `BP4 +0.06496`。
- bit13 forward 最大 finite delta 仅 `FP1 +0.00121`，backward 最大 finite delta 为 `BP8 +0.37173`，且 BP7 NaN。
- bit14 forward 全 NaN，说明强 exponent bit 可以在 forward 侧直接污染状态；但在非饱和 bit13 下，forward 明显比 backward 更容易恢复。

第二，detector response：

- bit10/13 forward detected sum 都为 0。
- bit13 backward detected sum 为 699。
- bit10 backward detected sum 为 89，主要来自 `BP4`。
- bit14 的高 detected count 主要反映 NaN/Inf 污染后的持续异常，不代表正常阈值下的细粒度检测质量。

第三，机制指标：

- Forward fault 能产生 attention-logit spike，例如 `FP2 + bit13` 达到 `2.559486e+20`，但 final eval loss 几乎不变。
- Backward fault 更容易产生 gradient corruption，例如 `BP9 + bit13` 的 `gradient_norm_pre=44032.0`、`BP8 + bit13` 出现 Inf 记录、`BP4 + bit10` 达到 `177209344.0`。

这与论文的机制解释方向一致：forward corruption 常表现为 transient activation/logit 扰动；backward corruption 直接进入梯度和 optimizer update，更容易改变训练轨迹和最终模型参数。

## 11. 与论文 IV-B 的对应和差异

本轮复现实验与论文 IV-B 有定性一致处：

1. Kernel sensitivity 存在：同一 bit 下不同 kernel 的影响差异很大，尤其是 `bit13` backward。
2. Backward faults 更严重：最强 finite eval loss 退化、最大参数偏移、detector response 都集中在 backward。
3. Forward faults 可产生 extreme attention logits：`FP2 + bit13` 的 attention logits 达到 `2.56e20`。
4. 高 exponent bit 可导致 NaN：bit14 在多数 kernel 上直接饱和。

主要差异和注意事项：

1. 本机 manifest 是 FP1-FP4、BP1-BP9，而不是论文环境中的 kernel 集合；kernel label 只在本机 profiling 口径下有效。
2. 本轮 IV-B 使用 `TARGET_INSTR=-1` 覆盖目标 kernel 内所有 HMMA 指令，符合论文 IV-B 的 all-HMMA 描述，但强度高于 IV-A 中固定单条 instruction 的口径。
3. bit14/all-HMMA 在本机过强，大量 run 进入同一类 NaN 饱和路径，因此不能用 bit14 对 kernel 做精细排序。
4. 本轮是 single-seed 复现，final eval loss 的小幅正负变化不能解释为稳健统计结论；但 `+0.15` 以上的退化、NaN、Inf-gradient 和大参数偏移已经远超 baseline 小波动。

## 12. 结论

本轮 IV-B 39 组 kernel sensitivity 实验全部完成，trigger schedule 完全一致。结论如下：

1. 本机环境下存在明确 kernel sensitivity，最清楚地出现在 `bit13` backward kernels。
2. `BP8 + bit13` 是最强 finite final eval loss 退化组合：final eval loss `4.6890`，delta `+0.3717`。
3. `BP9 + bit13` 是最干净的强非饱和组合：delta `+0.2881`，Parameter Difference `156.09`，`nonfinite_metric_count=0`。
4. `BP7 + bit13` 是 bit13 下唯一 final eval NaN 的 backward kernel。
5. `FP2 + bit13` 产生极端 attention logits（`2.56e20`），但 final eval loss 基本不变，适合支撑 IV-C 的 forward transient 机制讨论。
6. `BP4 + bit10` 是弱 bit 下的强 backward 梯度污染样例，max pre-clipping gradient norm 达到 `177209344.0`。
7. `bit14` 是强阳性但大面积饱和，适合证明 NaN propagation 和 clipping interaction，不适合作为 kernel 排序依据。

后续实验选择基于本报告结果：IV-E 使用 `BP9 + bit13`，因为它强但没有非有限污染；IV-F rate/duration 使用 `BP8 + bit13`，因为它在 1/10 频率下 final eval loss 更接近论文量级且是本机最强 finite 退化组合；IV-C/IV-D 主要复用 IV-B 数据，并补充 no-clipping 对照。

## 13. 证据文件

| 类型 | 路径 |
|---|---|
| Kernel manifest | `logs/iv_a_mb256/kernel_manifest.tsv` |
| Bit10 summary | `logs/iv_b_mb256/seed_42/bit_10/iv_b_summary_raw.csv` |
| Bit13 summary | `logs/iv_b_mb256/seed_42/bit_13/iv_b_summary_raw.csv` |
| Bit14 summary | `logs/iv_b_mb256/seed_42/bit_14/iv_b_summary_raw.csv` |
| Bit10 checkpoints | `checkpoints/iv_b_mb256/seed_42/bit_10/` |
| Bit13 checkpoints | `checkpoints/iv_b_mb256/seed_42/bit_13/` |
| Bit14 checkpoints | `checkpoints/iv_b_mb256/seed_42/bit_14/` |
| BP1/BP2 verbose validation | `plan&reports/iv_b_bp1_bp2_verbose_validation_report.md` |
| Paired baseline | `checkpoints/iv_f_mb256/baseline/seed_42/` |
| IV-B runner | `scripts/experiments/run_iv_b_kernel_sensitivity.sh` |
| IV-B summary script | `scripts/experiments/summarize_iv_b.py` |
