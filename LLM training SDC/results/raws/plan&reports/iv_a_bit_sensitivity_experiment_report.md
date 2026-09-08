# IV-A Bit-Position Sensitivity 复现实验报告

## 1. 实验定位

本报告记录 Section IV-A bit-position sensitivity 的正式复现实验结果。前置工程验证、mb256 baseline、chunked loss 修复和 kernel profiling 已单独记录在 [IV-A 前置工程验证报告](iv_a_mb256_sanity_experiment_report.md) 中。

本轮实验要回答的问题是：在固定训练配置、固定 seed、固定注入位置和固定 HMMA target 下，翻转 packed BF16 input register 的不同 bit position 是否会造成不同程度的训练退化、NaN 传播、梯度异常或参数偏移。

## 2. 实验配置

模型与训练配置沿用 IV-A mb256 主线：

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

Fault injection 配置如下：

| 项目 | 配置 |
|---|---|
| NVBit version | 1.7.6 |
| Opcode filter | `TARGET_OP=HMMA` |
| Kernel filter | `TARGET_FUNC_CONTAINS=ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` |
| Instruction index | `TARGET_INSTR=312` |
| Location | backward |
| Target register | `TARGET_REGISTER=1`，即第一 input register |
| SM / lane | `TARGET_SMID=0`，`TARGET_LANEID=0` |
| Trigger rate | 平均每 10 个 update 触发一次 |
| Duration | 1 update |
| Recompute | disabled |
| Detection | enabled for signal logging only |

Bit sweep 覆盖 `0..31`。其中 bit `0..15` 对应 packed register 的 `low_bf16`，bit `16..31` 对应 `high_bf16`。每个 half 的字段划分为：mantissa `0..6`，exponent `7..14`，sign `15`。

## 3. 数据完整性

本轮正式汇总文件为：`logs/iv_a_mb256/seed_42/iv_a_summary_full.csv`。

汇总命令合并了四个 checkpoint root：

```bash
python scripts/experiments/summarize_iv_a.py \
  --checkpoint-root checkpoints/iv_a_mb256/seed_42/bit_13 \
  --checkpoint-root checkpoints/iv_a_mb256/seed_42/bit_14 \
  --checkpoint-root checkpoints/iv_a_mb256/seed_42/bits_00_15 \
  --checkpoint-root checkpoints/iv_a_mb256/seed_42/bits_16_31 \
  --baseline-dir checkpoints/iv_f_mb256/baseline/seed_42 \
  --compute-parameter-difference \
  --output logs/iv_a_mb256/seed_42/iv_a_summary_full.csv \
  --expected-step 1000 \
  --expected-trigger-count 91 \
  --expected-first-trigger 2 \
  --expected-last-trigger 1000
```

完整性结果：

| 项目 | 结果 |
|---|---:|
| Bit runs | 32 / 32 complete |
| 每个 run 的 trigger count | 91 |
| Trigger schedule | 全部 `first=2`，`last=1000`，`ok=True` |
| Paired baseline final eval loss | 4.317315101623535 |
| NaN final eval loss bits | 14, 30 |
| 非 NaN bits | 30 |

## 4. 汇总结果

完整逐 bit 结果如下。`eval_delta = FI final eval loss - baseline final eval loss`。`param_diff` 是最终模型相对 paired baseline 的 post-hoc safetensors FP32 L2 参数差。

| bit | field | value | final eval loss | eval delta | param diff | max grad pre | detected | nonfinite metrics |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | mantissa | low_bf16 | 4.318465709686279 | 0.0011506080627441406 | 52.05637427675372 | 2.28125 | 0 | 0 |
| 1 | mantissa | low_bf16 | 4.3152570724487305 | -0.0020580291748046875 | 66.53117301573353 | 2.484375 | 0 | 0 |
| 2 | mantissa | low_bf16 | 4.31856107711792 | 0.0012459754943847656 | 51.354586474168194 | 2.28125 | 0 | 0 |
| 3 | mantissa | low_bf16 | 4.317941188812256 | 0.0006260871887207031 | 54.50583698318574 | 2.28125 | 0 | 0 |
| 4 | mantissa | low_bf16 | 4.317825794219971 | 0.0005106925964355469 | 53.2699971584322 | 2.28125 | 0 | 0 |
| 5 | mantissa | low_bf16 | 4.318479061126709 | 0.0011639595031738281 | 51.43568702051892 | 2.28125 | 0 | 0 |
| 6 | mantissa | low_bf16 | 4.315539836883545 | -0.0017752647399902344 | 63.66996288791265 | 2.28125 | 0 | 0 |
| 7 | exponent | low_bf16 | 4.315265655517578 | -0.0020494461059570312 | 57.40465235340702 | 2.359375 | 0 | 0 |
| 8 | exponent | low_bf16 | 4.318075656890869 | 0.0007605552673339844 | 51.293354829531104 | 2.28125 | 0 | 0 |
| 9 | exponent | low_bf16 | 4.317178249359131 | -0.00013685226440429688 | 51.60070539588004 | 2.28125 | 0 | 0 |
| 10 | exponent | low_bf16 | 4.314549446105957 | -0.002765655517578125 | 76.37882650136217 | 15.0625 | 0 | 0 |
| 11 | exponent | low_bf16 | 4.315248489379883 | -0.0020666122436523438 | 57.266395802934916 | 2.453125 | 0 | 0 |
| 12 | exponent | low_bf16 | 4.316005229949951 | -0.0013098716735839844 | 57.23180173812955 | 2.71875 | 0 | 0 |
| 13 | exponent | low_bf16 | 4.315753936767578 | -0.0015611648559570312 | 56.28740259180076 | 2.28125 | 0 | 0 |
| 14 | exponent | low_bf16 | NaN | NaN | NaN | 1.625 | 999 | 24956 |
| 15 | sign | low_bf16 | 4.315006732940674 | -0.002308368682861328 | 65.96913499052587 | 2.453125 | 0 | 0 |
| 16 | mantissa | high_bf16 | 4.3181376457214355 | 0.0008225440979003906 | 56.11031752307677 | 2.28125 | 0 | 0 |
| 17 | mantissa | high_bf16 | 4.3167643547058105 | -0.0005507469177246094 | 59.862634533829386 | 2.28125 | 0 | 0 |
| 18 | mantissa | high_bf16 | 4.317370891571045 | 0.000055789947509765625 | 55.776677042344815 | 2.28125 | 0 | 0 |
| 19 | mantissa | high_bf16 | 4.317312717437744 | -0.000002384185791015625 | 56.07824520575765 | 2.28125 | 0 | 0 |
| 20 | mantissa | high_bf16 | 4.316948413848877 | -0.0003666877746582031 | 54.01249147189547 | 2.28125 | 0 | 0 |
| 21 | mantissa | high_bf16 | 4.316396713256836 | -0.0009183883666992188 | 55.0852429029099 | 2.28125 | 0 | 0 |
| 22 | mantissa | high_bf16 | 4.317004203796387 | -0.0003108978271484375 | 52.55856783587821 | 2.28125 | 0 | 0 |
| 23 | exponent | high_bf16 | 4.318169116973877 | 0.0008540153503417969 | 51.750977226890406 | 2.28125 | 0 | 0 |
| 24 | exponent | high_bf16 | 4.315695762634277 | -0.0016193389892578125 | 67.15136131694351 | 2.578125 | 0 | 0 |
| 25 | exponent | high_bf16 | 4.315426826477051 | -0.001888275146484375 | 62.04641080686845 | 2.28125 | 0 | 0 |
| 26 | exponent | high_bf16 | 4.315846920013428 | -0.0014681816101074219 | 57.723131163016454 | 2.28125 | 0 | 0 |
| 27 | exponent | high_bf16 | 4.317054271697998 | -0.0002608299255371094 | 54.1128307070328 | 2.28125 | 0 | 0 |
| 28 | exponent | high_bf16 | 4.317363739013672 | 0.00004863739013671875 | 51.482737439930965 | 2.28125 | 0 | 0 |
| 29 | exponent | high_bf16 | 4.316102981567383 | -0.0012121200561523438 | 53.05837695404113 | 2.28125 | 0 | 0 |
| 30 | exponent | high_bf16 | NaN | NaN | NaN | 1.625 | 999 | 24956 |
| 31 | sign | high_bf16 | 4.3179850578308105 | 0.0006699562072753906 | 51.96969569144906 | 2.28125 | 0 | 0 |

## 5. 分组统计

按字段统计，去除 final eval loss 为 NaN 的 run 后：

| Field | Runs | NaN runs | Finite delta range | Finite avg delta | Finite avg param diff |
|---|---:|---:|---:|---:|---:|
| Mantissa | 14 | 0 | [-0.002058, 0.001246] | -0.000029 | 55.879128 |
| Exponent | 16 | 2 | [-0.002766, 0.000854] | -0.001048 | 57.484926 |
| Sign | 2 | 0 | [-0.002308, 0.000670] | -0.000819 | 58.969415 |

按 packed value 统计，去除 final eval loss 为 NaN 的 run 后：

| Packed value | Runs | NaN runs | Finite delta range | Finite avg delta | Finite avg param diff |
|---|---:|---:|---:|---:|---:|
| low_bf16 | 16 | 1 | [-0.002766, 0.001246] | -0.000705 | 57.750393 |
| high_bf16 | 16 | 1 | [-0.001888, 0.000854] | -0.000410 | 55.918647 |

关键极值：

| 指标 | Bit | 数值 | 解释 |
|---|---:|---:|---|
| 最大有限 eval delta | 2 | +0.0012459755 | 轻微高于 baseline，但幅度很小 |
| 最大有限绝对 eval delta | 10 | -0.0027656555 | 方向是 loss 更低，不构成退化 |
| 最大有限参数差 | 10 | 76.3788265 | 说明发生了显著轨迹偏移 |
| 最大有限 pre-clipping gradient norm | 10 | 15.0625 at step 13 | 局部梯度尖峰，但最终 eval loss 未恶化 |
| NaN final eval loss | 14, 30 | NaN | 对应 low/high BF16 的同一 local bit 14 |

## 6. 现象分析

### 6.1 强阳性集中在 exponent local bit 14

本轮只有 bit 14 和 bit 30 导致最终 evaluation loss 为 NaN。二者分别对应 `low_bf16` 和 `high_bf16` 的 local bit 14，均属于 exponent 区域。两次 run 的第一次非有限指标都出现在 step 2，而 step 2 也是 seed 42 下的第一次触发注入 step。

bit 14 和 bit 30 的 summary 均为：`TP=91, FP=908, TN=1, FN=0`。这里的高 FP 不应解释为正常 detector 阈值误报，而是 NaN 从 step 2 起持续污染训练状态后，非触发 step 也持续被判为 anomaly。

这部分复现了论文 IV-A 的核心定性结论之一：exponent bit 比 mantissa/sign bit 更容易产生灾难性影响，并且 packed BF16 register 的两个 half 呈现对称性。

### 6.2 bit 10 产生局部梯度尖峰和最大参数偏移，但没有 eval loss 退化

bit 10 的 final eval loss 为 4.314549446105957，低于 paired baseline 4.317315101623535，因此没有表现为 evaluation loss 退化。但它在 step 13 出现 `gradient_norm_pre=15.0625`，远高于其它有限 run 的常见范围，并且最终参数差达到 76.3788265，是所有有限 run 中最大。

这说明 bit 10 确实改变了训练轨迹，但在 1,000 steps、单 seed、固定 target 的设置下，这种轨迹偏移没有转化为更差的 final eval loss。

### 6.3 其它 finite bit 的 final eval loss 变化都很小

除 bit 14 和 bit 30 外，其余 30 个 run 的 final eval loss delta 范围是 `[-0.0027656555, +0.0012459755]`。这个范围小于论文中 60M baseline 跨 seed 标注的 `±0.006`，也没有出现稳定的正向 loss 退化。

Mantissa bits 没有 NaN，最大正向 eval delta 为 bit 2 的 `+0.0012459755`。Sign bits 也没有 NaN，bit 15 的 delta 为 `-0.0023083687`，bit 31 的 delta 为 `+0.0006699562`。这些结果支持“mantissa/sign 没有明显 eval loss 退化”的结论。

### 6.4 与论文 IV-A 的差异

论文 IV-A 报告：bit 10、11、26、27 出现超过 baseline variation 的 evaluation loss increase；bit 12、13、14、28、29、30 产生 NaN。当前本机复现实验只观察到 bit 14 和 bit 30 产生 NaN；bit 10、11、12、13、26、27、28、29 均未造成 final eval loss 退化。

主要差异来源应记录为实验口径差异，而不是简单判定论文结论失败：

1. 当前 IV-A 使用固定 backward kernel 和固定 `TARGET_INSTR=312`，而论文描述为每个 kernel function 随机选择 HMMA 指令，并在每 10 个 training iterations 中随机注入。
2. 当前只使用单 seed 42，论文先从多个 seed 中选择 best performing seed，并报告 baseline variation。
3. 当前硬件、CUDA driver、PyTorch/cuBLAS、NVBit 版本、数据子集和 sequence length 均与论文环境不完全一致。
4. 当前本地 profiling 得到的 GEMM/HMMA kernel 集合也不同于论文的 kernel 数量。

因此，本轮 IV-A 的严格结论应表述为：在本机固定 target 复现口径下，观察到 exponent local bit 14 的强 NaN 敏感性和 bit 10 的明显梯度/参数轨迹异常；没有观察到论文中更广泛的 exponent-bit final eval loss increase 或 NaN 集合。

## 7. 结论

本轮 IV-A 正式 bit sweep 已完整跑完，32 个 bit 均有完整 checkpoint、metrics、summary 和 final evaluation 记录。结果支持以下判断：

1. 工程链路稳定：所有 run 均 complete，触发次数和触发 schedule 与 seed 42 预期一致。
2. 强 SDC 现象可复现：bit 14 和 bit 30 在第一次注入 step 后产生持续 NaN 传播，最终 evaluation loss 为 NaN。
3. BF16 packed symmetry 存在：bit 14 与 bit 30 是两个 packed BF16 value 的同一 local exponent bit，表现高度一致。
4. Mantissa/sign bit 未表现出明显 final eval loss 退化。
5. bit 10 产生显著局部梯度尖峰和最大最终参数偏移，但没有造成 final eval loss 上升。
6. 当前结果只部分复现论文 IV-A：复现了 exponent-bit 强敏感性和 packed symmetry 的强阳性样例，但没有复现论文报告的完整敏感 bit 集合。

后续 IV-B 应以当前 IV-A 的两个发现作为选 bit 依据：bit 14 作为强 NaN 阳性位，bit 10 作为非 NaN 但有梯度/参数异常的敏感位，bit 13 作为论文强调但本机固定 target 下未退化的对照 exponent 位。

## 8. 证据文件

| 类型 | 路径 |
|---|---|
| Full IV-A summary | `logs/iv_a_mb256/seed_42/iv_a_summary_full.csv` |
| Baseline metrics | `checkpoints/iv_f_mb256/baseline/seed_42/metrics.jsonl` |
| Baseline checkpoint | `checkpoints/iv_f_mb256/baseline/seed_42/model_1000/` |
| IV-A bit 0-15 checkpoint root | `checkpoints/iv_a_mb256/seed_42/bits_00_15/` |
| IV-A bit 16-31 checkpoint root | `checkpoints/iv_a_mb256/seed_42/bits_16_31/` |
| IV-A bit 13 checkpoint root | `checkpoints/iv_a_mb256/seed_42/bit_13/` |
| IV-A bit 14 checkpoint root | `checkpoints/iv_a_mb256/seed_42/bit_14/` |
| IV-A logs | `logs/iv_a_mb256/seed_42/` |
| IV-A config | `scripts/configs/iv_a.sh` |
| IV-A runner | `scripts/experiments/run_iv_a_bit_sensitivity.sh` |
| Bit 14 runner | `scripts/experiments/run_iv_a_bit14_sanity.sh` |
| Summary script | `scripts/experiments/summarize_iv_a.py` |
