# VII-B 60M Fault Injection and Recompute 复现实验报告

## 1. 实验定位

本报告记录论文 VII-B scale effects 的本地降级复现结果。论文 VII-B 覆盖 60M、350M、1.3B 多个模型规模；本轮受单卡显存、时间和 kernel 映射成本限制，只复现 Table I 中的 **60M 单列**，且只使用 seed 42。

实验目标是验证：在相同训练配置和相同 fault schedule 下，单纯 fault injection 是否会造成训练质量退化，以及开启 detector + recompute 后是否能把 eval loss 拉回 fault-free baseline 附近。

## 2. 实验配置

共同训练配置如下：

| 项目 | 配置 |
|---|---|
| Model | `llama_60m` |
| Seed | 42 |
| Training updates | 10,000 |
| Evaluation | every 1,000 updates |
| Max sequence length | 256 |
| Micro batch size | 256 |
| Effective batch size | 512 |
| Optimizer | AdamW |
| Max LR | 1e-3 |
| Precision | bfloat16 |
| Gradient clipping | global norm 1.0 |
| Dataset | local preprocessed C4 subset |
| Data repeat | 3 passes for 10,000-step runs |
| Detection alpha | 0.05 |

三种实验条件如下：

| 条件 | 说明 |
|---|---|
| baseline | 无 fault injection，保留 detection-on 代码路径 |
| FI | fault injection + detection logging，不执行 recompute |
| FI + recompute | 与 FI 使用相同 fault schedule，检测到 anomaly 后重算该 step |

Fault injection 配置如下：

| 项目 | 配置 |
|---|---|
| Injection location | backward |
| Kernel labels | BP8 / BP9 |
| Target function IDs | 113 / 60 |
| Opcode filter | `TARGET_OP=HMMA` |
| Instruction index | `TARGET_INSTR=-1`，覆盖匹配 kernel 内全部 HMMA |
| Target register | `TARGET_REGISTER=1` |
| SM / lane | `TARGET_SMID=0` / `TARGET_LANEID=0` |
| Bit position / mask | bit 13 / 8192 |
| Trigger rate | 1/100 |
| Duration | random 1-5 update steps |
| FI + recompute alpha | 0.05 |

## 3. 数据来源与完整性

数据路径如下：

| 类型 | 路径 |
|---|---|
| Baseline 首段 | `checkpoints/vii_b_mb256_3906/baseline/seed_42` |
| Baseline 续跑段 | `checkpoints/vii_b_mb256/baseline/seed_42` |
| FI | `checkpoints/vii_b_mb256/fi/seed_42` |
| FI + recompute | `checkpoints/vii_b_mb256/fi_recompute/seed_42` |
| Campaign manifest | `checkpoints/vii_b_mb256/campaign_manifest.txt` |
| Existing summary CSV | `logs/vii_b_mb256/vii_b_summary.csv` |
| Existing stats CSV | `logs/vii_b_mb256/vii_b_summary_stats.csv` |

Baseline 的 10,000 步数据由两个目录拼接：`vii_b_mb256_3906` 覆盖 step 1-3906，`vii_b_mb256` 覆盖 step 3907-10000。现有 `vii_b_summary.csv` 是在续跑段基础上汇总的，因此 baseline 在 1000/2000/3000 的 eval loss 为空；本报告使用首段与续跑段合并后的 baseline eval curve。FI 和 FI + recompute 均为完整 10,000 步 run。

完整性检查：

| 条件 | Completed updates | Final checkpoint | Trigger count | Nonfinite metrics |
|---|---:|---|---:|---:|
| baseline | 10,000 | yes | 0 | 0 |
| FI | 10,000 | yes | 262 | 0 |
| FI + recompute | 10,000 | yes | 262 | 0 |

注意：baseline 续跑段的 summary 中出现 `fp=2679`，这是续跑时 detector `window_mean` 未恢复导致阈值异常的历史伪影。该问题影响 baseline 的 detection false-positive 统计，不影响训练更新和 eval loss；因此本报告不把 baseline 的 `detected_count` 作为科学结论使用。

## 4. 与论文口径的偏差

论文 VII-B 的 fault 口径是 bit 12 + 随机 kernel 对。本地按该口径完成过一组 60M FI run，但未产生可测量退化：final eval loss 与 baseline 的差值约为 -0.001，trigger step 没有被 detector 捕获，属于单 seed 噪声范围内的 null result。

因此本轮 VII-B 主结果改用 IV-B 本地筛选出的强 finite target：**bit 13 + BP8/BP9 + all-HMMA**。其中 bit 13 是本机最强的非饱和 exponent bit；BP8/BP9 是 bit 13 下最强的两个 finite backward kernel。该调整提高了本地实验对 FI 退化和 recompute 恢复效果的可观测性，但也意味着结果不能直接等同于论文原始 Table I 的 bit12/random-kernel 数值。

另一个偏差是数据采样。由于本地 C4 子集约 3,906 update steps 后耗尽，10,000-step run 使用 `train_data_repeat=3` 复用数据。论文使用 C4 全量流式训练，本报告的 absolute loss 与泛化曲线应按本地复现口径解释。

## 5. Final 结果

| 条件 | Final eval loss | Delta vs baseline | Trigger count | Detected count | Detection rate on trigger | Precision | Max grad pre | Max Rt | Parameter diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3.539680958 | 0.000000000 | 0 | n/a | n/a | n/a | 2.28125 | 0.000644684 | n/a |
| FI | 3.547073126 | +0.007392168 | 262 | 76 | 29.01% | 100.00% | 17152.0 | 0.001152039 | 364.556885 |
| FI + recompute | 3.538131714 | -0.001549244 | 262 | 261 | 99.62% | 100.00% | 288.0 | 0.000659943 | 337.711728 |

FI 条件最终 eval loss 比 baseline 高 0.0074，但中途多次出现更明显退化；FI + recompute 的最终 eval loss 回到 baseline 附近，最终差值为 -0.0015，处于单 seed 噪声范围内。

## 6. Eval Loss 曲线

| Step | Baseline eval loss | FI eval loss | FI delta | FI + recompute eval loss | Recompute delta |
|---:|---:|---:|---:|---:|---:|
| 1000 | 4.317315102 | 4.342903614 | +0.025588512 | 4.324731827 | +0.007416725 |
| 2000 | 3.901448965 | 3.907695532 | +0.006246567 | 3.903640032 | +0.002191067 |
| 3000 | 3.768843412 | 3.785640717 | +0.016797304 | 3.768623114 | -0.000220299 |
| 4000 | 3.694839239 | 3.785001755 | +0.090162516 | 3.694574356 | -0.000264883 |
| 5000 | 3.645809174 | 3.661441565 | +0.015632391 | 3.645953894 | +0.000144720 |
| 6000 | 3.613745451 | 3.704819441 | +0.091073990 | 3.611028671 | -0.002716780 |
| 7000 | 3.588826418 | 3.615526676 | +0.026700258 | 3.587453842 | -0.001372576 |
| 8000 | 3.569879532 | 3.638617277 | +0.068737745 | 3.569297552 | -0.000581980 |
| 9000 | 3.552542210 | 3.601814985 | +0.049272776 | 3.550538301 | -0.002003908 |
| 10000 | 3.539680958 | 3.547073126 | +0.007392168 | 3.538131714 | -0.001549244 |

FI 的退化不是单调累积，而是呈 intermittent spike：step 4000 和 6000 的 eval delta 均约 +0.09，step 8000 仍有 +0.069。到 final step 时 loss 部分恢复，因此仅看 final delta 会低估中途训练质量波动。FI + recompute 在 3000 step 之后基本贴合 baseline，最大绝对 delta 约 0.0027。

## 7. Detection 与 Recompute 统计

| 条件 | TP | FP | FN | TN | Detection rate | Correct recompute | Incorrect recompute |
|---|---:|---:|---:|---:|---:|---:|---:|
| FI | 76 | 0 | 186 | 9738 | 29.01% | 0 | 0 |
| FI + recompute | 261 | 0 | 1 | 9738 | 99.62% | 208 | 53 |

FI 与 FI + recompute 使用相同 trigger count，但 detector 行为差异很大。FI 中 detector 只捕获 76/262 个 trigger step，说明许多 fault 对当前 Rt/gradient signal 不够强或未转化为被阈值捕获的异常；但其中被捕获的 step 没有 false positive。FI + recompute 中 detector 捕获 261/262 个 trigger step，并执行 recompute，使 max gradient norm pre 从 FI 的 17152.0 降到 288.0，final loss 也回到 baseline 附近。

`incorrect_recompute=53` 表示重算后仍被判定为不完全恢复或相关检查未归入 correct recompute；从 eval curve 和 bounded gradient 看，这些 step 没有造成可见训练失败。

## 8. 结论

本地 VII-B 60M 结果支持以下结论：

1. 在 bit 13 + BP8/BP9 + all-HMMA 的本地强 target 下，FI 会造成可观测的训练退化，尤其体现在中途 eval loss spike 和极端 gradient norm 上。
2. 开启 recompute 后，eval loss 曲线基本恢复到 fault-free baseline，final eval loss 与 baseline 的差值为 -0.0015，处于单 seed 噪声范围内。
3. Recompute 条件下 detector 对 trigger step 的覆盖率达到 99.62%，并显著抑制极端 gradient norm，说明该机制对本轮注入模式有效。
4. 由于只做 60M、单 seed、数据 repeat=3，并且 fault target 从论文 bit12/random-kernel 调整为本地 bit13/BP8/BP9，本报告应作为本地复现和机制验证，而不是论文 VII-B 全规模数值的完全复刻。

## 9. 后续建议

如果继续补强 VII-B，优先级最高的是再跑 1-2 个 seed 或补一个 350M 小样本，以区分 final loss 的单 seed 噪声和规模效应。若目标是和论文 Table I 数值更接近，则需要回到 bit12/random-kernel 口径，并接受本机该口径可能退化很弱的问题。

