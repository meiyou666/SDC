# IV-C / IV-D Forward 与 Backward 故障传播差异 复现实验报告

## 1. 实验定位

本报告对应论文 Section IV-C "Transient Effects in the Forward Pass" 与 Section IV-D "Severe Effects in the Backward Pass"。这两节在论文中不是独立的大规模 campaign，而是基于 IV-B 的 kernel×bit 注入数据做的机制分析，外加 Fig. 2 所需的 gradient clipping 开/关对照。

本地复现沿用同一思路：

1. **主分析复用 IV-B 数据**：60M、seed 42、1,000 步、13 个 kernel（FP1–FP4 / BP1–BP9）× 3 个 bit（10/13/14）= 39 组 FI run。IV-C 关注 forward kernel 的瞬时 loss/attention-logit 尖峰；IV-D 关注 backward kernel 的 gradient norm 尖峰、NaN/Inf 传播与持久 eval loss 退化。
2. **补 1 组 clipping 对照**：`BP9 + bit13` 的 `--grad_clipping 0.0` no-clipping run，与 IV-B 中同位形的 clipped run（`--grad_clipping 1.0`）配对，复现论文 Fig. 2 的机制结论。

## 2. 实验配置

训练与注入口径与 IV-B 完全一致：

| 项目 | 配置 |
|---|---|
| Model / Seed | `llama_60m` / 42（单一） |
| Training updates | 1,000（LR schedule 仍为 warmup 1,000 + cosine 100,000） |
| Batch | micro 256 × accumulation 2 = effective 512，max length 256，bfloat16 |
| Kernel 集合 | 本机 manifest 13 个 GEMM kernel：FP1–FP4（forward）、BP1–BP9（backward） |
| 注入位形 | `TARGET_OP=HMMA`，`TARGET_INSTR=-1`（kernel 内全部 HMMA），`TARGET_REGISTER=1`，`TARGET_SMID=0`，`TARGET_LANEID=0` |
| Bits | 10 / 13 / 14（bitmask 1024 / 8192 / 16384） |
| Trigger | 平均 1/10 update steps，duration 1，backward kernel 用 `--fi_nvbit_location backward`，forward kernel 用 `forward` |
| Paired baseline | `checkpoints/iv_f_mb256/baseline/seed_42`，eval loss 4.3173 |

Clipping 对照在以上口径上只改 `--grad_clipping`：clipped = 1.0（复用 IV-B `bit_13/BP9`），no-clipping = 0.0（新增 1 组）。

## 3. 数据完整性

| 项目 | 结果 |
|---|---:|
| IV-B FI runs（复用） | 39 / 39 complete，触发 schedule 全部校验通过（91 次，first=2，last=1000） |
| No-clipping 对照 | 1 / 1 complete，触发 schedule 与 clipped run 一致 |
| Baseline | 复用 paired baseline（同上） |

汇总与证据文件：

| 类型 | 路径 |
|---|---|
| IV-B 逐组汇总 | `logs/iv_b_mb256/seed_42/bit_{10,13,14}/iv_b_summary_raw.csv` |
| Clipping 对照汇总 | `logs/iv_cd_mb256/clipping/seed_42/bit_13/BP9/clipping_summary.csv` |
| No-clipping 日志 | `logs/iv_cd_mb256/clipping/seed_42/bit_13/BP9/no_clipping.log` |
| 逐组 metrics | `checkpoints/iv_b_mb256/seed_42/bit_*/{FP,BP}*/metrics.jsonl`、`checkpoints/iv_cd_mb256/.../no_clipping/metrics.jsonl` |

## 4. IV-C 结果：forward fault 只产生瞬时效应

### 4.1 无持久退化

Forward kernel（FP1–FP4）在 bit 10/13 下最终 eval loss delta 全部接近 0（最大为 FP4-bit10 的 +0.0243，其余 |Δ| ≤ 0.006），parameter difference 50–83（同量级于 IV-B 中所有 run 的背景漂移）。与论文"前向故障对训练长期影响有限"一致。

### 4.2 Attention logit 尖峰：强 transient 信号，与触发步逐步对齐

逐步 `max_attention_logits`（softmax 前的最大注意力 logit，正常训练带 ≤ ~55）显示尖锐的单步尖峰，且与触发步严格对齐：

| Run | 尖峰值 | 尖峰步落在触发步上 | 单步恢复性 |
|---|---|---:|---|
| FP1 bit10 | 350 – 540 | 91 / 91 | 是（如 step 2：1.2 → 438 → 1.2） |
| FP2 bit10 | 147 – 3,920 | 84 / 84 | 是 |
| FP2 bit13 | 5.4e18 – 2.6e20 | 85 / 91 | 是（如 step 2：1.2 → 5.4e18 → 1.2） |
| FP1 bit13 | 无尖峰 | 0 / 0 | — |

尖峰量级（最高 2.6e20）与论文 IV-B/IV-C 报告的"3e3–3e20、可产生近 one-hot attention"同量级。尖峰只持续一个 step，下一步立即回到正常带，训练轨迹无持久偏移——正是论文所说的"sharp spike in the maximum attention logits is a strong indicator of corruption"。

**Kernel 标签映射注意**：论文的尖峰集中在 FP1（其 attention-score GEMM）；本地最大尖峰出现在 **FP2**（`s1688gemm_..._nn`）run 中，FP1（`..._tn`，按布局推断为 Q@K^T 的 attention-score kernel）在 bit10 下有 350–540 的中等尖峰、bit13 下无尖峰。本机 kernel 编号与论文不逐一对齐（本机仅 4 个 forward kernel，论文 5 个），且任何 forward kernel 的损坏都可能传播进 attention logits。结论按现象（forward fault → transient attention-logit spike）陈述，不强行对齐序号。

### 4.3 Loss 尖峰：本地表现为小幅抬升而非显著 spike

论文在 FP0/FP4、bit 9/10 观察到小幅 training loss spike 并立即恢复。本地 FP4-bit10 在触发步的 loss 仅比 baseline 高 ~0.05–0.1（如 step 115：7.372 vs 7.290），FP 各 run 的 `max_training_loss` = 10.471，即从未超过 step 1 的初始 loss——本地 forward fault 的 loss 信号弱，**attention logit 尖峰是本地更可靠的 forward corruption 信号**。bit 9 未在 IV-B 矩阵中，论文的 bit 9 对照本地缺失。

### 4.4 bit14：forward 也出现立即 NaN 饱和

bit14 下 4 个 forward kernel 全部在训练早期 NaN 饱和（eval loss = NaN，`nonfinite_metric_count` ≈ 25,000，attention logits 等指标冻结在初始值 1.23，gradient norm 冻结在 1.63）。对应论文"注入 MSB 时 forward pass 也出现 NaN"；本地 bit14 即承担 MSB 型强阳性 bit 的角色。

## 5. IV-D 结果：backward fault 造成严重且持久的梯度污染

### 5.1 Backward >> forward

Backward kernel 在相同 bit 下产生（i）clipping 前 gradient norm 尖峰与（ii）持久 eval loss 退化，forward kernel 两者皆无：

| Bit | Backward 梯度尖峰（gnorm pre 最大值） | Backward eval delta 范围 | Forward 同期表现 |
|---|---|---|---|
| 10 | BP4: **1.8e8**；其余 ≈ 2.3–3.1 | +0.000 ~ +0.065 | gnorm ≈ 2.3（正常），Δ ≤ +0.024 |
| 13 | BP5: 2.4e16，BP7: 1.8e15，BP9: 44,032，BP8: 14,400，BP6: 223 | +0.000 ~ **+0.372**（BP7 NaN 饱和） | gnorm ≈ 2.3，Δ ≤ +0.002 |
| 14 | BP1/BP2: 2.3（91 个 nonfinite 后恢复有限退化 +0.061）；BP3–BP9: NaN 饱和 | NaN（7/9 个 kernel） | 同 NaN 饱和 |

Backward fault 直接污染梯度 → 经 optimizer 放大为有害参数更新，与论文 IV-D 的因果链一致；forward fault 的影响经 loss 间接回传，量级可忽略。

### 5.2 Gradient clipping 的缓解证据（clipped，BP9 bit13）

Clipped run 的逐步指标显示所有 clipping 前尖峰都被裁剪到阈值附近，训练持续进行：

| Step | gnorm pre（clipping 前） | gnorm post（clipping 后） |
|---:|---:|---:|
| 194 | 2,240 | 0.996 |
| 229 | 3,504 | 1.008 |
| 250 | 4,224 | 0.996 |
| 261 | 1,752 | 1.000 |
| … | … | … |
| 最大 | 44,032 | 1.016 |

训练 loss 轨迹持续跟随 baseline（高 ~0.2–0.3 的 persistent bump），final eval loss 4.6054（Δ +0.288）——严重事件被吸收，但污染未完全消除（论文同款现象：clipping 缓解最严重事件，loss bump 仍在）。

### 5.3 No-clipping 对照：训练早期发散并停滞（BP9 bit13）

| 指标 | Clipped（IV-B 复用） | No-clipping（新增） |
|---|---:|---:|
| Final eval loss | 4.6054 | **7.2727** |
| Δ vs baseline | +0.288 | **+2.955** |
| Final train loss | 4.544 | 7.241 |
| Max gradient norm pre / post | 44,032 / 1.016 | 468,992 / 468,992（无裁剪） |
| Detected anomaly count | 71 | 46 |
| Nonfinite 逐步指标 | 0 | 2（first @ step 801） |

逐步轨迹显示 no-clipping run 在**前 100 步内即发散**：step 100 时 loss 已 7.56（baseline 7.50），step 200 时 7.32 对 6.49，随后全程停滞在 ~7.3 的高损失平台（梯度范数萎缩至 0.1–0.6，模型陷入坏 basin），期间 step 801 出现一次 gnorm = inf。对应论文 Fig. 2 绿色曲线的定性结论：**去掉 gradient clipping 后，个别被污染梯度即可摧毁训练**。

**与论文机制的细微差别**：论文描述的 stall 机制是"极大梯度 → Adam 二阶矩发散到 inf → 自适应更新归零 → 参数冻结"。本地观察到的主要路径是早期坏更新后陷入高损失低梯度平台（梯度萎缩而非二阶矩爆炸），inf 事件（step 801）出现一次但未主导轨迹。两种表现都归于"无 clipping 时 corrupted gradient 的破坏不可恢复"，论文机制可视为其特例（当 inf 持续主导时）。

### 5.4 bit14 backward：NaN 饱和与 BP1/BP2 等价轨迹

bit14 下 BP3–BP9 全部 NaN 饱和（指标冻结，训练实际死亡）；BP1/BP2 呈现完全相同的有限退化轨迹（Δ 均 +0.0614，91 个 nonfinite 指标，gnorm 2.30）。该等价现象已在 IV-B verbose 短验证中确认为 bit14/all-HMMA 饱和 + total-norm 指标不可区分所致，不能作为 kernel 排序证据。

## 6. 与论文对比小结

| 论文论点 | 本地结果 |
|---|---|
| IV-C：forward fault 只产生瞬时效应，尖峰立即恢复 | ✔ attention logit 单步尖峰、与触发步对齐、eval 无持久影响 |
| IV-C：attention logit 尖峰是强 corruption 信号（论文 FP1，3e3–3e20） | ✔ 同量级（本地最大 2.6e20），但出现在 FP2 run，kernel 序号不强行对齐 |
| IV-C：forward 注入 MSB 也产生 NaN | ✔ bit14 下全部 forward kernel 立即 NaN 饱和 |
| IV-D：backward fault 造成严重梯度污染与持久退化 | ✔ 梯度尖峰 1e2–1e16，eval delta 最高 +0.372 |
| IV-D/Fig.2：clipping 缓解最严重事件，无 clipping 训练停滞 | ✔ clipped 尖峰全部裁至 ~1.0、训练延续（Δ +0.288）；no-clipping 早期发散停滞（Δ +2.955），机制细节与论文略有差异（见 5.3） |
| IV-D：inf 梯度范数使 clipped 范数塌缩为 0 | 部分：no-clipping run 在 step 801 出现 inf（塌缩机制的前提事件），clipped run 无 inf |

## 7. 结论

本地 IV-C/IV-D 复现支持论文的核心机制结论：**forward fault 表现为可自动恢复的瞬时信号（attention logit 尖峰），backward fault 直接污染梯度并造成持久退化；gradient norm clipping 能把最严重事件限制在单步内，去掉 clipping 则早期即发散停滞。** 与论文的差异已逐条记录：尖峰所在 kernel 的本地标签不同（FP2 vs 论文 FP1）、no-clipping 的发散形态（坏 basin 停滞 vs 二阶矩冻结）、forward loss 尖峰信号弱（本地以 attention logit 为主）。

## 8. 证据文件

| 内容 | 路径 |
|---|---|
| IV-B 逐组汇总（39 组） | `logs/iv_b_mb256/seed_42/bit_{10,13,14}/iv_b_summary_raw.csv` |
| Clipping 对照汇总 | `logs/iv_cd_mb256/clipping/seed_42/bit_13/BP9/clipping_summary.csv` |
| No-clipping 训练日志 | `logs/iv_cd_mb256/clipping/seed_42/bit_13/BP9/no_clipping.log` |
| 逐步 metrics（loss/gnorm/attn） | `checkpoints/iv_b_mb256/seed_42/bit_*/{FP,BP}*/metrics.jsonl`、`checkpoints/iv_cd_mb256/clipping/seed_42/bit_13/BP9/no_clipping/metrics.jsonl` |
| Paired baseline | `checkpoints/iv_f_mb256/baseline/seed_42` |
| 对照运行脚本 | `scripts/experiments/run_iv_cd_clipping_comparison.sh`、`scripts/configs/iv_cd.sh`、`scripts/experiments/summarize_iv_cd_clipping.py` |
| 背景：BP1/BP2 等价轨迹验证 | `plan&reports/iv_b_bp1_bp2_verbose_validation_report.md` |
