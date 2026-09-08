# IV-F Duration Effects All-HMMA 复现实验报告

## 1. 实验定位

本报告记录 IV-F temporal effects 中的 **fault duration** 新版重跑结果。旧 duration 报告使用固定单条 `TARGET_INSTR=312`，只覆盖 duration `1/3/5`，退化信号过弱；本报告只分析新版 **BP8 + bit13 + all-HMMA first-register** 数据，不覆盖旧报告，也不与旧单指令口径混合统计。

本轮目标是验证：在相同 seed、相同 backward kernel、相同 bit、相同注入起点下，故障持续时间从 `1` 增加到 `3/5/7/9` 个连续 update step 时，final eval loss、参数偏移、梯度尖峰和 detector response 是否随 duration 增强；同时导出 duration=3 在 step 500 附近的 loss/Rt 时间序列，用于支撑论文 Fig.3 风格图。

## 2. Target 选择依据

新版 duration 与新版 IV-F rate 使用相同 target：`BP8 + bit13 + TARGET_INSTR=-1`。该 target 来自 IV-B kernel sensitivity 的本机结果：`BP8 + bit13` 是本机最强的 finite backward 退化组合，并且新版 rate 实验已经证明它能复现 fault rate 越高、训练退化越强的趋势。

本轮 target 配置如下：

| 项目 | 配置 |
|---|---|
| Kernel label | `BP8` |
| Kernel filter | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_nt` |
| Kernel location | backward |
| HMMA instruction count | 64 |
| Opcode filter | `TARGET_OP=HMMA` |
| Instruction index | `TARGET_INSTR=-1`，覆盖 BP8 内全部匹配 HMMA 指令 |
| Target register | `TARGET_REGISTER=1`，第一 input register |
| SM / lane | `TARGET_SMID=0`，`TARGET_LANEID=0` |
| Bit position / mask | bit 13 / 8192 |
| Trigger start | update step 500 |
| Recompute | disabled |
| Detection | enabled for signal logging |

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
| Evaluation | final evaluation only |
| Paired baseline | `checkpoints/iv_f_mb256/baseline/seed_42` |
| Baseline final eval loss | 4.317315101623535 |

Duration 轴如下：

| Duration | Trigger window |
|---:|---|
| 1 | step 500 |
| 3 | step 500-502 |
| 5 | step 500-504 |
| 7 | step 500-506 |
| 9 | step 500-508 |

## 4. 数据完整性

数据来源：

| 类型 | 路径 |
|---|---|
| Duration summary | `logs/iv_f_duration_all_hmma_mb256/iv_f_duration_all_hmma_summary.csv` |
| Duration aggregate | `logs/iv_f_duration_all_hmma_mb256/iv_f_duration_all_hmma_aggregate.csv` |
| Fig.3 time series | `logs/iv_f_duration_all_hmma_mb256/iv_f_duration_all_hmma_fig3_timeseries.csv` |
| Duration checkpoints | `checkpoints/iv_f_duration_all_hmma_mb256/BP8/duration/` |
| Campaign manifest | `checkpoints/iv_f_duration_all_hmma_mb256/BP8/campaign_manifest.txt` |
| Paired baseline | `checkpoints/iv_f_mb256/baseline/seed_42` |

完整性检查：

| Duration | Status | Trigger count | First trigger | Last trigger | Expected window | Trigger schedule ok | Final checkpoint | Nonfinite metrics |
|---:|---|---:|---:|---:|---|---|---|---:|
| 1 | complete | 1 | 500 | 500 | 500-500 | true | yes | 0 |
| 3 | complete | 3 | 500 | 502 | 500-502 | true | yes | 0 |
| 5 | complete | 5 | 500 | 504 | 500-504 | true | yes | 0 |
| 7 | complete | 7 | 500 | 506 | 500-506 | true | yes | 0 |
| 9 | complete | 9 | 500 | 508 | 500-508 | true | yes | 0 |

五组 run 均正常完成，trigger window 与配置一致，且没有 NaN/Inf 指标。`optimizer.pt` 按脚本设置删除，不影响本报告使用的 final checkpoint、metrics、summary 和参数差分析。

## 5. Duration 主结果

| Duration | Final eval loss | Delta vs baseline | Final train loss | Max grad pre | Max Rt | Max Rt jump | Max attention logits | Parameter diff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.320427418 | +0.003112316 | 4.268017530 | 1296.0 | 0.000644684 | 0.000095367 | 43.00 | 37.501175 |
| 3 | 4.324925900 | +0.007610798 | 4.275601625 | 1736.0 | 0.000831604 | 0.000198364 | 43.00 | 52.182808 |
| 5 | 4.329085827 | +0.011770725 | 4.281071901 | 2480.0 | 0.000892639 | 0.000217438 | 54.25 | 56.842388 |
| 7 | 4.342257023 | +0.024941921 | 4.293146372 | 3616.0 | 0.000843048 | 0.000106812 | 43.00 | 67.586954 |
| 9 | 4.345874310 | +0.028559208 | 4.297841311 | 11136.0 | 0.000801086 | 0.000274658 | 64.00 | 69.335712 |

Final eval loss 随 duration 单调上升：`4.3204 -> 4.3249 -> 4.3291 -> 4.3423 -> 4.3459`。相对 paired baseline 的 delta 也单调增大：`+0.0031 -> +0.0076 -> +0.0118 -> +0.0249 -> +0.0286`。这比旧固定单指令 duration 报告中的弱趋势明确得多，并且方向与论文 IV-F duration 结论一致。

参数 L2 差异整体随 duration 增大：`37.50 -> 52.18 -> 56.84 -> 67.59 -> 69.34`，说明更长的故障窗口不仅影响 final eval loss，也使最终模型权重离 paired baseline 更远。

梯度尖峰同样随 duration 增强，`max_gradient_norm_pre` 从 duration=1 的 `1296.0` 增至 duration=9 的 `11136.0`。由于启用了 global norm clipping，`max_gradient_norm_post` 均约为 `1.0078125`，说明 clipping 抑制了更新范数，但不能完全消除错误累积对训练轨迹和 eval loss 的影响。

Rt 指标能清楚捕获注入窗口附近的异常，但 `max_rt` 本身不严格随 duration 单调变化。更稳定的 duration 效应体现在 final eval loss、参数差和梯度尖峰上。

## 6. Detection 统计

| Duration | TP | FP | FN | TN | Detected count | Detection rate on trigger |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 0 | 999 | 1 | 100.00% |
| 3 | 3 | 0 | 0 | 997 | 3 | 100.00% |
| 5 | 2 | 0 | 3 | 995 | 2 | 40.00% |
| 7 | 4 | 0 | 3 | 993 | 4 | 57.14% |
| 9 | 7 | 0 | 2 | 991 | 7 | 77.78% |

所有 duration 的 false positive 都为 0。duration=1 和 duration=3 的 trigger step 全部被 detector 捕获；duration=5/7/9 中有部分 trigger step 未被判定为 anomaly。这说明当前 Rt 阈值更像是捕获“强异常 update”的 detector，而不是逐个标记所有注入 step 的 oracle。即便如此，随着 duration 增长，最终训练退化和参数偏移仍然增强。

## 7. Fig.3 Sanity 时间序列

Fig.3 time series 导出的是 `BP8 + duration=3 + seed 42`，窗口为 step 400-700。注入发生在 step 500-502。

窗口统计如下：

| 区间 | Count | Avg loss | Max loss | Avg Rt | Max Rt | Max Rt jump | Max grad pre |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre, relative step < 0 | 100 | 5.307193100 | 5.478873491 | 0.000538254 | 0.000568390 | 0.000038147 | 1.648438 |
| injection, step 500-502 | 3 | 5.332983971 | 5.519022226 | 0.000789642 | 0.000831604 | 0.000198364 | 1736.0 |
| post, relative step > 0 | 202 | 4.977264636 | 5.941059589 | 0.000576359 | 0.000831604 | 0.000087738 | 1736.0 |

注入窗口附近的关键 step：

| Step | Relative | Active | Loss | Rt | Rt jump | Grad pre | Detected | Max attn logits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 499 | -1 | 0 | 5.233234406 | 0.000526428 | 0.000038147 | 1.351562 | 0 | 23.00 |
| 500 | 0 | 1 | 5.183700562 | 0.000724792 | 0.000198364 | 736.0 | 1 | 24.00 |
| 501 | 1 | 1 | 5.296229124 | 0.000812531 | 0.000087738 | 1144.0 | 1 | 30.375 |
| 502 | 2 | 1 | 5.519022226 | 0.000831604 | 0.000019073 | 1736.0 | 1 | 25.50 |
| 503 | 3 | 0 | 5.941059589 | 0.000755310 | 0.000076294 | 6.1875 | 0 | 20.00 |
| 504 | 4 | 0 | 5.575751543 | 0.000686646 | 0.000068665 | 2.53125 | 0 | 22.625 |
| 505 | 5 | 0 | 5.422569513 | 0.000633240 | 0.000053406 | 1.460938 | 0 | 32.25 |

这个小实验可以支撑 Fig.3 风格图：step 500 开始 Rt 立即从注入前约 `5.3e-4` 跃升到 `7.25e-4`，step 502 达到 `8.32e-4`；pre-clipping gradient norm 从注入前的约 `1` 级别跃升到 `736/1144/1736`；training loss 的最高点出现在注入结束后的 step 503，为 `5.9411`，符合“Rt spike 先出现，随后 loss bump”的现象。

## 8. 与论文趋势对照

论文 IV-F duration 轴给出的 eval loss 大致为 `4.30/4.31/4.32/4.33/4.35`，随持续时间增加而上升。本轮结果为 `4.3204/4.3249/4.3291/4.3423/4.3459`，同样呈单调上升，并且数值量级接近论文报告范围。

需要注意的是，本轮不是论文原始 target 的完全复刻，而是基于本机 IV-B/IV-F rate 结果选择了 `BP8 + bit13 + all-HMMA`。因此本报告的强结论是“在本机敏感 target 下复现了 duration effect”，而不是“完全复刻论文原始 kernel/bit 分布下的绝对数值”。

## 9. 结论

本轮新版 IV-F duration all-HMMA 实验工程执行完整，五个 duration 点均有完整 checkpoint、summary 和可追溯 trigger window。

科学结论上，本轮成功复现 duration effect：

1. Final eval loss 和 paired delta 随 duration 从 1 到 9 单调增大。
2. 最终参数 L2 差异整体随 duration 增大，说明更长故障窗口造成更大的训练轨迹偏移。
3. Pre-clipping gradient spike 随 duration 明显增强，duration=9 达到 `11136.0`。
4. Duration=3 的时间序列能支撑 Fig.3 风格图：Rt 在注入窗口内 spike，training loss 在注入结束后出现 bump。
5. Detector 没有 false positive，但对较长 duration 并非逐 step 全捕获；这不影响 duration 趋势结论，但说明检测率不能简单等同于注入步数覆盖率。

旧固定单指令 duration 报告保留为历史口径记录；新版 IV-F duration 主结论应以本报告为准。

