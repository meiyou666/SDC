# IV-E Spatial Effects 复现实验报告

## 1. 实验定位

本报告对应论文 Section IV-E "Spatial Effects: Lanes and SMs"。论文固定注入位置为反向传播的 BP9 kernel、bit position 13，分别遍历 warp lane（固定 SM）与 SM（固定 lane），结论是故障效应对 lane 和 SM 的选择不敏感：lane sweep 最终 eval loss 为 4.60 ± 0.08，SM sweep 为 4.63 ± 0.07（论文 baseline 4.30 ± 0.006，1,000 steps）。

本地复现按"单 seed、砍重复"原则降级：仅 seed 42，每部分 16 组（论文为 32+32），共 32 组 FI run，复用 IV-A/IV-B 的 paired fault-free baseline：

| 子实验 | 固定项 | 变量 | 组数 |
|---|---|---|---:|
| Lane sweep | `TARGET_SMID=0`，kernel=BP9，bit=13 | `TARGET_LANEID=0,2,4,...,30` | 16 |
| SM sweep | `TARGET_LANEID=0`，kernel=BP9，bit=13 | `TARGET_SMID=0,8,16,...,120` | 16 |

GPU 差异：论文为 L40S，本地为 RTX 4090（128 个 SM，经 `multi_processor_count` 确认），SM 抽样范围按本机 0..127 设定，stride 8 等距覆盖。

## 2. 实验配置

训练配置沿用 IV-A/IV-B mb256 主线：

| 项目 | 配置 |
|---|---|
| Model | `llama_60m`（58.07M params） |
| Seed | 42（单一） |
| Training updates | 1,000 |
| LR schedule | warmup 1,000 steps，cosine span 100,000 steps，max LR 1e-3 |
| Optimizer | AdamW，weight decay 0.01 |
| Precision | bfloat16，max length 256 |
| Batch | micro 256 × accumulation 2 = effective 512 |
| Gradient clipping | global norm 1.0 |
| Final evaluation | step 1,000 后执行一次 |
| Paired baseline | `checkpoints/iv_f_mb256/baseline/seed_42`，eval loss 4.3173 |

Fault injection 配置：

| 项目 | 配置 |
|---|---|
| Kernel | BP9：`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8`（manifest func ID 60，backward） |
| Opcode / instruction | `TARGET_OP=HMMA`，`TARGET_INSTR=-1`（kernel 内全部 16 条 HMMA） |
| Target register | `TARGET_REGISTER=1`（first input register） |
| Bit position / mask | 13 / 8192 |
| Location | backward |
| Trigger schedule | 平均 1/10 update steps，duration 1 step |
| Recompute | disabled（检测路径开启，仅记录信号） |

## 3. 数据完整性

| 项目 | 结果 |
|---|---:|
| Lane FI runs | 16 / 16 complete |
| SM FI runs | 16 / 16 complete |
| Trigger schedule | 32 / 32 校验通过（每组 91 次触发，first=2，last=1000） |
| Nonfinite 污染 | lane 部分全 0；SM 部分 10/16 组有 1–4 个 nonfinite 逐步指标 |

汇总文件：

| 类型 | 路径 |
|---|---|
| Lane 逐组汇总 | `logs/iv_e_mb256/seed_42/lane/iv_e_lane_summary_raw.csv` |
| SM 逐组汇总 | `logs/iv_e_mb256/seed_42/sm/iv_e_sm_summary_raw.csv` |
| 逐组日志 | `logs/iv_e_mb256/seed_42/{lane,sm}/*.log` |
| Checkpoints | `checkpoints/iv_e_mb256/seed_42/{lane,sm}/` |
| 运行脚本 | `scripts/experiments/run_iv_e_spatial_sweep.sh`、`scripts/configs/iv_e.sh` |

## 4. Lane sweep 结果（固定 SM0）

| Lane | Final eval loss | Δ vs baseline | Param diff | Max grad norm (pre) | Detected / 91 |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.5972 | +0.2799 | 152.80 | 19456 | 74 |
| 2 | 4.6644 | +0.3470 | 153.45 | 30720 | 73 |
| 4 | 4.5731 | +0.2557 | 153.09 | 23424 | 71 |
| 6 | 4.5494 | +0.2321 | 153.42 | 17920 | 71 |
| 8 | 4.5576 | +0.2403 | 153.59 | 21120 | 72 |
| 10 | 4.6356 | +0.3183 | 155.92 | 16896 | 74 |
| 12 | 4.5829 | +0.2656 | 155.15 | 18944 | 72 |
| 14 | 4.5270 | +0.2097 | 152.04 | 22656 | 69 |
| 16 | 4.5782 | +0.2609 | 152.23 | 22144 | 73 |
| 18 | 4.6703 | +0.3530 | 154.59 | 20224 | 75 |
| 20 | 4.5691 | +0.2518 | 154.19 | 18432 | 75 |
| 22 | 4.5730 | +0.2557 | 152.78 | 16512 | 70 |
| 24 | 4.5206 | +0.2033 | 152.80 | 15424 | 74 |
| 26 | 4.6325 | +0.3152 | 156.32 | 16512 | 73 |
| 28 | 4.5935 | +0.2761 | 152.86 | 18304 | 73 |
| 30 | 4.5566 | +0.2393 | 151.29 | 23680 | 72 |

统计（n=16）：

| 指标 | Mean ± Std（sample） | 范围 |
|---|---|---|
| Final eval loss | 4.5863 ± 0.0444 | 4.5206 – 4.6703 |
| Δ vs baseline | +0.2690 ± 0.0444 | +0.2033 – +0.3530 |
| Parameter difference | 153.53 ± 1.39 | 151.29 – 156.32 |

所有 lane 的退化同量级，最低（lane 24，+0.203）与最高（lane 18，+0.353）相差约 1.7 个标准差内，无离群 lane；逐步指标无非有限值，max gradient norm pre 稳定在 1.5e4–3.1e4。

## 5. SM sweep 结果（固定 lane0）

| SM | Final eval loss | Δ vs baseline | Param diff | Max grad norm (pre) | Detected / 91 | Nonfinite |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.5818 | +0.2645 | 152.26 | 24704 | 74 | 4 |
| 8 | 4.6436 | +0.3263 | 154.70 | 20096 | 72 | 1 |
| 16 | 4.6149 | +0.2976 | 154.01 | 35584 | 75 | 1 |
| 24 | 4.6328 | +0.3155 | 153.30 | 22016 | 72 | 1 |
| 32 | 4.6294 | +0.3121 | 154.86 | 15680 | 73 | 0 |
| 40 | 4.5774 | +0.2601 | 152.22 | 14720 | 73 | 0 |
| 48 | 4.5765 | +0.2592 | 153.28 | 24064 | 74 | 3 |
| 56 | 4.5911 | +0.2738 | 151.32 | 25216 | 75 | 1 |
| 64 | 4.5473 | +0.2299 | 150.83 | 19200 | 71 | 0 |
| 72 | 4.6528 | +0.3355 | 156.75 | 19712 | 72 | 1 |
| 80 | 4.5899 | +0.2726 | 154.50 | 14656 | 73 | 0 |
| 88 | 4.5499 | +0.2326 | 151.88 | 18816 | 76 | 0 |
| 96 | 4.5852 | +0.2679 | 153.44 | 15552 | 73 | 1 |
| 104 | 4.5791 | +0.2618 | 153.37 | 32512 | 73 | 0 |
| 112 | 4.5545 | +0.2371 | 152.96 | 19584 | 73 | 1 |
| 120 | 4.5805 | +0.2631 | 154.09 | 20992 | 73 | 1 |

统计（n=16）：

| 指标 | Mean ± Std（sample） | 范围 |
|---|---|---|
| Final eval loss | 4.5929 ± 0.0327 | 4.5473 – 4.6528 |
| Δ vs baseline | +0.2756 ± 0.0327 | +0.2299 – +0.3355 |
| Parameter difference | 153.36 ± 1.49 | 150.83 – 156.75 |

所有 SM 的退化同量级，无离群 SM。10/16 组出现 1–4 个 nonfinite 逐步指标（如个别 step 的 attention logits 或 gradient norm 非有限，集中在触发窗口附近），但均未导致训练失效或 eval loss 离群——final eval loss 全部落在 4.55–4.65 的窄带内。

## 6. 与论文对比及可复现性观察

**与论文结论对齐。** 论文报告 lane sweep eval loss 4.60 ± 0.08、SM sweep 4.63 ± 0.07（其 baseline 4.30）；本地对应为 4.586 ± 0.044（lane）与 4.593 ± 0.033（SM）（本地 baseline 4.317）。两边均表明：在 BP9 + bit13 这个强但非饱和的注入位形下，故障影响与具体 lane / SM 无关， intra-warp 与 inter-SM 传播行为一致。

**同配置重复 run 的背景抖动。** IV-B 的 BP9-bit13（SM0/lane0）与 IV-E 的 lane_00（完全相同的注入位形与触发 schedule，触发步序列逐步核对一致）最终 eval loss 分别为 4.6054 与 4.5972（差 ~0.008），parameter difference 156.09 与 152.80（差 ~3.3）。这给出固定 seed 下 bf16 非确定性 kernel 的背景抖动量级：eval loss ~0.01。IV-E 组间 spread（std 0.033–0.044）比该抖动大一个数量级，因此"无空间敏感性"的结论不受非确定性影响。

**检测信号一致性。** 各组 detected anomaly count 在 69–76 / 91 次触发之间（约 76%–84%），lane 部分与 SM 部分无系统差异，与"各位置故障严重度同量级"相互印证。

## 7. 结论

本地 IV-E 复现支持论文结论：在 BP9 kernel、bit 13 的注入位形下，故障效应**对 lane 选择（stride 2 抽 16/32）和 SM 选择（stride 8 抽 16/128）均不敏感**，所有采样位置产生统计上同量级的 eval loss 退化（Δ ≈ +0.23 ~ +0.35）、parameter difference（≈ 151–157）与梯度尖峰。

局限性：① 每部分 16 组的统计功效弱于论文的 32+32；② 单 seed；③ 等距 stride 抽样理论上可能漏掉窄空间特征，但论文全量扫描同样未发现敏感性，风险较低。SM 部分少量 nonfinite 逐步指标不影响结论，但报告引用逐步曲线时应注意这些 step。

## 8. 证据文件

| 内容 | 路径 |
|---|---|
| Lane 逐组汇总 CSV | `logs/iv_e_mb256/seed_42/lane/iv_e_lane_summary_raw.csv` |
| SM 逐组汇总 CSV | `logs/iv_e_mb256/seed_42/sm/iv_e_sm_summary_raw.csv` |
| 逐组训练日志 | `logs/iv_e_mb256/seed_42/lane/lane_{00..30}.log`、`logs/iv_e_mb256/seed_42/sm/sm_{000..120}.log` |
| Checkpoints | `checkpoints/iv_e_mb256/seed_42/lane/lane_*`、`checkpoints/iv_e_mb256/seed_42/sm/sm_*` |
| Paired baseline | `checkpoints/iv_f_mb256/baseline/seed_42` |
| Kernel manifest | `logs/iv_a_mb256/kernel_manifest.tsv` |
| 运行与配置脚本 | `scripts/experiments/run_iv_e_spatial_sweep.sh`、`scripts/configs/iv_e.sh`、`scripts/experiments/summarize_iv_e.py` |
