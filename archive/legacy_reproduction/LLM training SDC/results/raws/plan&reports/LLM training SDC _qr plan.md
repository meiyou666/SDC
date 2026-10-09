# Exploring Silent Data Corruption as a Reliability Challenge in LLM Training复现



## 一、环境配置与准备

### （一）准备操作

1. 在并行智算云上创建单卡GPU 4090实例
2. 创建虚拟环境，激活环境(**每次开启实例都要操作!!!**)：`source ~/sdc_llm_env/bin/activate` `cd ~/projects/llm-sdc-training`

3. 创建工作目录并拉取论文代码，清理仓库自带output、编译产物
4. 根据`requirements.txt`下载所需的Python库
5. 拉取NVBit：文章使用的是1.7.4版本，与云服务器不适配，改用1.7.6版本（本地下载.tar后传输解压）
6. 编译NVBit：`make -C nvbit/1.7.6_nvbit_release/tools/fault_injection clean` `make -C nvbit/1.7.6_nvbit_release/tools/fault_injection ARCH=sm_89 -j"$(nproc)"`
7. 准备数据集：存储限制无法下载原始C4，改为从镜像站拉取前6个分片，流式截取并保存在`./data/c4_subset_2M.json.gz`中
8. 准备tokenizer：选择t5，本地下载后传输云服务器
9. 修改代码：云服务器无法连接外网——改成离线加载模式，其中wandb功能砍掉，保存数据回传后本地实现；nvbit 1.7.4版本改成1.7.6版本，使代码适配新API


### （二）环境配置

- 资源类型：NVIDIA GeForce RTX 4090
- GPU： 1卡 * 24 GB显存
- CPU：10vCPU
- 内存： 60GB
- 开发框架：PyTorch2.7.0-Ubuntu 24.04
- 操作系统： Ubuntu 24.04
- SM compute capability: 8.9
- Host CPU: x86_64
- OS: Linux
- GCC version : 13.3.0
- CUDA version: 12.8.93
- CUDA driver version: 580.105.08 ???
- numpy==2.3.3
  torch==2.9.0
  wandb==0.26.1
  loguru==0.7.3
  tqdm==4.67.1
  transformers==4.56.2
  datasets==4.1.1

注：模型选择llama_60M

## 二、冒烟测试

### （一）、环境稳定性验证

60M，固定seed跑100步（含warmup），连续运行2次（只改变保存路径），关注逐步loss值和 checkpoint 的 compute_parameter_difference 。

使用`scripts/base/train_baseline.sh`；loss值可直接在终端查看，对比参数差是否为0。

注：仓库采用分离式 checkpoint 机制——模型权重保存为 `model.safetensors`，优化器与调度器状态保存为 `optimizer.pt`，训练指标通过 `loguru` 实时输出至终端并持久化到 `metrics.jsonl`，且在训练循环结束时自动补存最终 checkpoint

本轮实验报告：[environment_stability_test_report](environment_stability_test_report.md)

### （二）、NVBit验证

加载NVBit配置，设置参数（高频率、MSB、反向传播），跑50步故障注入短训练，观察能否看到 NVBit instrumentation 加载信息、训练loss（出现NaN或spike）、梯度范数（日志中出现极大的梯度范数或inf）

使用`scripts/fi/train_fi.sh`、`scripts/configs/nvbit_default.sh`、`scripts/configs/common.sh`等，可在终端或日志中找到待观测数据

本轮实验报告：[nvbit_fault_injection_smoke_test_report](nvbit_fault_injection_smoke_test_report.md)

## 三、论文实验

### IV-A 比特位敏感度

当前 IV-A 主线回到论文的 **bit-position sensitivity**：60M，固定单一种子，1000步，micro batch size 256、gradient accumulation 2、有效 batch size 512，单 SM、单 lane、first input register，注入频率 1/10 steps，故障持续 1 step。实验变量是 `TARGET_BITMASK=1<<bit`。bit 13 和 bit 14 仍然是 IV-A 的一部分，但已先作为 sanity run 单独跑完，用来验证 mb256/chunked loss/NVBit 注入链路；随后 16x2 串行 bit sweep 中跳过 bit 13 和 bit 14，避免重复跑同一 bit。

本轮已修复原先 micro batch 256 的显存问题：训练 loss 不再一次性构造完整 `[batch, sequence, vocab]` logits，而是分块计算 causal LM loss。因此 IV-A 与后续 IV-F 重跑均采用论文 60M 的 micro-batch 结构。后续实验不再筛选 best seed，而是固定使用 `scripts/configs/iv_a.sh` 中的单一种子；旧 `batch_size=32 / accumulation=16` 的 baseline 不再作为 mb256 实验的严格 paired baseline，需要先跑一次同 seed、同 mb256 配置的 fault-free paired baseline。

论文原文 “We selected one out of five HMMA instructions in each kernel function and injected faults in one out of every ten training iterations, both chosen at random” 只明确了两件事：HMMA 指令选择和训练 iteration 选择都是随机的；它没有公开这五条 HMMA 指令的确定标准，也没有给出具体 SASS instruction index。当前本地实现仍沿用已经验证过的固定 target：`TARGET_FUNC_CONTAINS=ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn`、`TARGET_INSTR=312`、`fi_nvbit_location=backward`，每组只改变 bitmask。这个口径牺牲了一部分论文二进制级覆盖度，但变量控制更干净，也避免凭空发明“前五条 HMMA”规则。当前脚本尚未实现“同一个触发 step 覆盖所有 kernel 且每个 kernel 各自随机选一条 HMMA”的精确论文逻辑。

kernel profiling 和 `logs/iv_a_mb256/kernel_manifest.tsv` 保留为 kernel-level 诊断证据文件。manifest 中出现多少个 forward GEMM/HMMA kernel 就记为 FP1..FPn，出现多少个 backward GEMM/HMMA kernel 就记为 BP1..BPm；若本机 profiling 得到的数量不同于论文的 5 个 forward、10 个 backward，以本机 manifest 为准，并在报告中解释 GPU/PyTorch/NVBit/微批结构导致 kernel selection 差异。

`run_iv_a_bit_sensitivity.sh bit_13` 与 `run_iv_a_bit14_sanity.sh` 是 IV-A 的先行 sanity run，结果后续应和其余 30 个 bit 合并进 IV-A bit sensitivity 分析。单独的 `run_iv_a_bit13_kernel_sweep.sh` 是额外诊断脚本：对 manifest 中每个 kernel 单独跑一次 bit 13，用于定位本机敏感 kernel，更接近 IV-B 的 kernel sensitivity 分析，不替代 IV-A 的固定-target bit run。正式 IV-A bit sweep 要求 `checkpoints/iv_f_mb256/baseline/seed_${IV_A_SEED}` 已存在，用于直接汇总 paired eval delta 和 post-hoc parameter difference。

运行顺序：

```bash
# 1. 先跑固定单种子的 paper-style micro-batch paired baseline
bash scripts/experiments/run_iv_paired_baseline.sh

# 2. 先单独跑 bit 13 / bit 14，作为 IV-A sanity run
bash scripts/experiments/run_iv_a_bit_sensitivity.sh bit_13
bash scripts/experiments/run_iv_a_bit14_sanity.sh

# 3. 跑其余 IV-A bit-position sensitivity，分两批，跳过已跑的 bit 13 / bit 14
bash scripts/experiments/run_iv_a_bit_sensitivity.sh bits_00_15
bash scripts/experiments/run_iv_a_bit_sensitivity.sh bits_16_31
```

如果要继续用 bit 13 定位 kernel 敏感性，先 profile kernel，再跑诊断 sweep。此时结果更适合放在 IV-B/敏感 kernel 定位中解释：

```bash
bash scripts/experiments/profile_iv_a_kernels.sh
bash scripts/experiments/run_iv_a_bit13_kernel_sweep.sh logs/iv_a_mb256/kernel_manifest.tsv FP1 BP1 BP4 BP9
```

观测：指标为eval loss、是否NaN、Parameter Difference	

本轮 IV-A 主线目标：验证 exponent bits（10–15, 26–31）才有影响、mantissa + sign bit（0–9, 16–25）无明显差异。bit 13 和 bit 14 已先单独跑 sanity run；`bits_00_15` 汇总时用 `--skip-bit 13 --skip-bit 14` 排除重复项；最终 IV-A 汇总应同时传入 `bit_13`、`bit_14`、`bits_00_15`、`bits_16_31` 四个 checkpoint root，把已跑 sanity bits 和其余 30 个 bit 合并分析。

本轮正式实验报告：[IV-A Bit-Position Sensitivity 复现实验报告](iv_a_bit_sensitivity_experiment_report.md)

本轮预实验报告：[IV-A 前置工程验证报告：mb256 baseline、chunked loss 与 kernel profiling](iv_a_mb256_sanity_experiment_report.md)

### IV-B Kernel 敏感度

IV-B 的主问题是 kernel sensitivity：在相同 seed、相同训练配置、相同注入频率下，比较不同 GEMM/HMMA kernel 被注入时是否更容易造成 eval loss 退化、NaN、梯度异常或更大的参数偏移。为把总训练量控制在 30-50 组，同时保留横向 kernel 比较能力，本轮采用 **13 个本机 profiled kernel × 3 个 bit = 39 组 FI 训练**，baseline 复用 `checkpoints/iv_f_mb256/baseline/seed_${IV_A_SEED}`。

训练配置沿用 IV-A mb256 主线：60M、seed 42、1000 update、micro batch size 256、gradient accumulation 2、有效 batch size 512、`--grad_clipping 1.0`、单 SM、单 lane、first input register、注入频率 1/10 steps、duration 1 step。脚本仍使用带检测的 `torchrun_main.py`，但不启用 recompute，因此可观察 detection signal，不会把故障 step 回滚。

Kernel 集合以本机 `logs/iv_a_mb256/kernel_manifest.tsv` 为准，而不是强行补齐论文中的 FP1-FP5、BP1-BP10。当前 manifest 为 FP1-FP4、BP1-BP9，共 13 个 kernel。每一行使用 profiling 时保存的 `target_func_contains` 来限定单个 kernel function，forward kernel 使用 `fi_nvbit_location=forward`，backward kernel 使用 `fi_nvbit_location=backward`。IV-B 运行时固定 `TARGET_INSTR=-1`，即对该 kernel 内所有匹配 `TARGET_OP=HMMA` 的 HMMA 指令插桩注入，而不是沿用 manifest 中随机保存的一条 `target_instr`。这样 IV-B 的变量是 kernel label，并且满足论文 IV-B “we injected faults into each HMMA instruction in these kernels” 的注入覆盖口径。

Bit 集合选 `10, 13, 14`：bit 10 是本机 IV-A 前半段中出现过局部梯度尖峰的 exponent bit；bit 13 是论文中多处使用/强调的敏感 bit，也便于和已有 IV-A bit13 sanity run 口径对应；bit 14 是本机 IV-A sanity run 已确认可导致 NaN 的强阳性 bit。bit 15 不纳入主矩阵，因为它在当前 IV-A 结果中未造成 eval loss 退化，且论文原文也没有把它作为主要现象位强调；若后续需要 sign/control 对照，可另做少量补充 run，不占用 IV-B 主矩阵预算。

运行顺序：

```bash
# 0. 如果 kernel_manifest.tsv 还不存在或环境/批大小改过，先重新 profile
bash scripts/experiments/profile_iv_a_kernels.sh

# 1. 每个 bit 单独一批；每批跑 manifest 中全部 13 个 kernel
bash scripts/experiments/run_iv_b_kernel_sensitivity.sh bit_10
bash scripts/experiments/run_iv_b_kernel_sensitivity.sh bit_13
bash scripts/experiments/run_iv_b_kernel_sensitivity.sh bit_14
```

每批输出：checkpoint 写入 `checkpoints/iv_b_mb256/seed_42/bit_XX/<kernel_label>/`，日志写入 `logs/iv_b_mb256/seed_42/bit_XX/<kernel_label>.log`，批次汇总写入 `logs/iv_b_mb256/seed_42/bit_XX/iv_b_summary_raw.csv`。脚本会跳过完整 run；若发现 partial checkpoint 或 partial log，会保留现场并报错提示，不自动覆盖。

观测：final eval loss、相对 paired baseline 的 eval loss delta、是否 NaN/Inf、Parameter Difference、trigger count、detected anomaly count、pre/post clipping gradient norm、max attention logits、max `rt`。分析时优先把结果分成三层：第一层看哪些 kernel/bit 直接 NaN；第二层看非 NaN run 是否有显著 eval loss delta；第三层看 eval loss 不显著时，参数差、梯度尖峰和 detection signal 是否仍能说明该 kernel 对注入更敏感。

目标：在精简预算下判断本机 profiled kernels 中是否存在类似论文 IV-B 的 kernel 敏感性排序或强阳性 kernel；若 39 组中除 bit14 外仍普遍没有 eval loss 退化，需要在报告中明确说明当前复现实验没有观察到论文级别的 loss 敏感性，只能证明工程注入链路有效以及 NaN 型故障可被触发。

当前 bit14 部分结果中，BP1 与 BP2 出现了除 `_time` 外逐步 metrics 完全一致的现象。由于二者 manifest kernel filter 不同但触发 step 均表现为 `gradient_norm_pre=inf`、`gradient_norm_post=0.0`，暂不能判断这是 bit14/all-HMMA 饱和导致的等效训练轨迹，还是 kernel filter / NVBit 插桩映射需要复核。为避免把该现象误解释为可靠的 kernel 排序证据，增加一个短验证实验：

```bash
bash scripts/experiments/validate_iv_b_bp1_bp2_verbose.sh
```

该验证只跑 BP1/BP2、bit14、`TARGET_INSTR=-1`、`TOOL_VERBOSE=1`、强制 step 2 注入、`exit_after=5`，输出到 `checkpoints/iv_b_mb256/validation/bp1_bp2_bit14_all_hmma_verbose/` 和 `logs/iv_b_mb256/validation/bp1_bp2_bit14_all_hmma_verbose/`。验证目标是检查两个 run 的 `TARGET_FUNC_CONTAINS` 是否分别对应 BP1/BP2，以及 verbose NVBit 日志中实际 `inspecting` 的 kernel name 和 HMMA `idx:` 集合是否不同。脚本会额外生成 `logs/iv_b_mb256/validation/bp1_bp2_bit14_all_hmma_verbose/verbose_validation_summary.tsv`，记录每个 run 的 inspected kernel、HMMA idx 数量和 idx fingerprint。若 verbose 证据不同，则 BP1/BP2 完全同轨迹应在报告中标记为 bit14 饱和/total-norm 指标不可区分；若 verbose 证据仍不可区分，则需要优先修正 kernel disambiguation 或 manifest 生成逻辑后再继续解释 BP1/BP2。

BP1/BP2 短验证报告：[IV-B BP1/BP2 verbose 短验证报告](iv_b_bp1_bp2_verbose_validation_report.md)

本轮正式实验报告：[IV-B Kernel Sensitivity 复现实验报告](iv_b_kernel_sensitivity_experiment_report.md)

### IV-C / IV-D Forward 与 Backward 故障传播差异

论文 IV-C/IV-D 不是新的大规模 bit/kernel sweep，而是基于 IV-B 观察做机制解释：IV-C 关注 forward fault 的 transient training loss spike 和 maximum attention logits spike；IV-D 关注 backward fault 更严重、gradient norm spike、以及 gradient clipping 开启/关闭后的差异。当前复现实验应尽量复用 IV-B 数据，避免重复跑完整矩阵。

IV-B 数据对 IV-C/IV-D 的支撑情况如下：

| 问题 | IV-B 数据是否足够 | 说明 |
|---|---|---|
| Forward fault 是否主要表现为 transient effect | 基本足够，但需等 IV-B bit10/13/14 的 FP1-FP4 都跑完 | IV-B 已包含 forward kernels、all-HMMA、相同 trigger schedule 和 training metrics，可从时间序列中提取 loss spike 与 max attention logits spike。bit14 可能 NaN 饱和，bit10/13 更适合分析 transient spike。 |
| Backward fault 是否更容易造成 gradient corruption 和 eval loss 退化 | 基本足够，但需等 IV-B bit10/13/14 的 BP1-BP9 都跑完 | IV-B 已包含 backward kernels、pre/post clipping gradient norm、detected anomaly、eval loss、parameter difference，可直接比较 forward/backward 的严重程度。 |
| Gradient clipping 是否缓解 backward 故障 | 不足，需要补充 no-clipping 对照 | 当前 IV-B 全部使用 `--grad_clipping 1.0`，只能分析 clipping enabled 下的现象，不能支持论文 IV-D 中 “with vs without clipping” 的对比。 |
| Infinite gradient norm 导致 clipped gradient collapse to zero | IV-B 可提供证据，但 clipping 对照更有力 | bit14 的 BP1/BP2 可作为强阳性背景证据；正式 no-clipping 对照只选 `BP9 bit13`，避免 bit14 饱和使机制分析退化成 NaN/Inf 失效记录。 |

因此，IV-C/IV-D 采用两阶段计划：

1. **复用 IV-B 做主分析**：在 IV-B 三批完成后，按 `location=forward/backward` 分组分析 `loss`、`max_attention_logits`、`gradient_norm_pre/post`、`rt`、final eval loss 和 parameter difference。IV-C 重点看 FP kernels 的瞬时 loss/attention-logit spike 是否恢复；IV-D 重点看 BP kernels 的 gradient spike、NaN、eval loss delta 和 detection signal。
2. **补跑最小 clipping 对照**：只选一个 backward kernel/bit 组合：`BP9 bit13`。`grad_clipping=1.0` 的 clipped 条件直接复用 IV-B 已完成 run，只新增 `grad_clipping=0.0` 的 no-clipping run，用于验证论文 IV-D 的 clipping 机制结论。选择 `BP9 bit13` 的原因是它在 IV-B clipped 条件下已经有强信号但没有非有限污染：final eval loss `4.6054`、delta `+0.2881`、Parameter Difference `156.09`、`max_gradient_norm_pre=44032.0`、detected count `71`、`nonfinite_metric_count=0`。相比 `BP8 bit13`，BP9 的 final eval loss 稍低但更干净；相比 `BP1 bit14`，BP9 避免了 bit14/all-HMMA 饱和导致 no-clipping 直接 NaN 的风险，更适合作为机制对照主样例。

新增配置与脚本：

```bash
# 配置：继承 IV-B/IV-A 训练口径
scripts/configs/iv_cd.sh

# Clipped 复用 IV-B；脚本只补跑 no-clipping 条件。IV-D 只跑 BP9 bit13。
bash scripts/experiments/run_iv_cd_clipping_comparison.sh BP9 13

# 输出汇总脚本，由 run_iv_cd_clipping_comparison.sh 自动调用
python scripts/experiments/summarize_iv_cd_clipping.py \
  --clipped-dir checkpoints/iv_b_mb256/seed_42/bit_13/BP9 \
  --no-clipping-dir checkpoints/iv_cd_mb256/clipping/seed_42/bit_13/BP9/no_clipping \
  --baseline-dir checkpoints/iv_f_mb256/baseline/seed_42 \
  --kernel-label BP9 \
  --bit 13 \
  --output logs/iv_cd_mb256/clipping/seed_42/bit_13/BP9/clipping_summary.csv
```

输出路径：

| 类型 | 路径 |
|---|---|
| Clipped 条件 checkpoint | 复用 `checkpoints/iv_b_mb256/seed_42/bit_XX/<kernel_label>/` |
| No-clipping 条件 checkpoint | `checkpoints/iv_cd_mb256/clipping/seed_42/bit_XX/<kernel_label>/no_clipping/` |
| No-clipping 条件日志 | `logs/iv_cd_mb256/clipping/seed_42/bit_XX/<kernel_label>/no_clipping.log` |
| Clipping 对照汇总 | `logs/iv_cd_mb256/clipping/seed_42/bit_XX/<kernel_label>/clipping_summary.csv` |

IV-C/IV-D 的最终分析口径：

- IV-C 不再新增 forward 训练，直接复用 IV-B 的 FP1-FP4 数据；若 bit14 全部 NaN 饱和，则以 bit10/13 的 forward 时间序列作为 transient spike 主证据。
- IV-D 先复用 IV-B 的 BP1-BP9 数据说明 backward 更严重；再只用 `BP9 bit13` 的 clipped/no-clipping 对照支撑 gradient clipping 机制讨论。
- 如果 `BP9 bit13` no-clipping run 因 optimizer second moment 变为 Inf 导致训练停滞或 final eval loss 异常，应按论文 IV-D 的 “training stalls / optimizer update collapses” 现象解释；如果直接 NaN，则记录为去掉 clipping 后从 clipped finite 退化转为失效。当前不追加 `BP8 bit13` 或 `BP1 bit14` 的 no-clipping run，避免扩大 IV-C/D 预算和引入饱和 target。

No-clipping 补跑参数选择原则：除 `--grad_clipping 0.0` 外，其余参数必须与被复用的 IV-B clipped run 完全一致，包括 seed、kernel label、bitmask、`TARGET_FUNC_CONTAINS`、`TARGET_INSTR=-1`、SM/lane/register、trigger rate、duration、训练步数、batch 结构、数据路径和 final evaluation 口径。本轮 IV-C/D 明确只补 `BP9 bit13`，不再追加 `BP8 bit13` 或 `BP1 bit14`。每个 no-clipping 条件都必须和一个已完成的 IV-B clipped run 配对，否则不启动补跑。

本轮实验报告：[IV-C/IV-D Forward 与 Backward 故障传播差异 复现实验报告](iv_cd_forward_backward_report.md)

### IV-E Spatial Effects: Lanes and SMs

论文 IV-E 研究 fault location within GPU 的空间敏感性，分为 lane sweep 和 SM sweep 两部分。论文固定 injection site 为 `BP9`、bit position 为 `13`，使用与前面实验相同的随机注入 schedule；lane 实验固定 SM、遍历 32 个 lanes；SM 实验固定 lane、每隔 4 个 SM 抽样一次。论文结论是 BP9/bit13 对 lane 和 SM 的选择不敏感。

注意：本机 GPU 是 NVIDIA GeForce RTX 4090，论文原文 GPU 是 NVIDIA L40S。lane 数两边相同（CUDA warp 固定 32 个 lane）；SM 数已确认：本机 `multi_processor_count=128`（RTX 4090），少于 L40S。因此 SM 编号范围使用 `0..127`，不照搬论文 L40S 的抽样范围。

IV-E 数据不能直接由 IV-B 替代。IV-B 固定 `TARGET_SMID=0`、`TARGET_LANEID=0`，只能证明某个固定空间位置下的 kernel/bit 敏感性；IV-E 必须改变 `TARGET_SMID` 或 `TARGET_LANEID`，因此需要单独补跑空间 sweep。IV-E 主 target 定为 `BP9 + bit13`，选择依据不再依赖论文标签，而是本机 IV-B 的可解释性：`BP9 bit13` 在 1/10 频率下 final eval loss 为 `4.6054`，delta 为 `+0.2881`，Parameter Difference 为 `156.09`，detected count 为 `71`，且 `nonfinite_metric_count=0`。它已经有足够强的故障信号，但没有进入非有限饱和路径，适合用于 lane/SM sweep 中比较空间位置差异。`BP8 bit13` 的 eval loss 更高（`4.6890`，delta `+0.3717`），但已有 `nonfinite_metric_count=2`，更接近 Inf/NaN 边界；若用于空间 sweep，部分 lane/SM 可能直接饱和，使 IV-E 变成“是否触发非有限污染”的比较，而不是连续可比的 spatial sensitivity 分析。因此 BP8 留给 IV-F temporal strength 主线，IV-E 保持 BP9。

本轮预算：**每部分 16 组训练**（lane 部分 16 组、SM 部分 16 组，共 32 组），比论文的 32+32 减半，用等距 stride 抽样保持空间覆盖：

| 子实验 | 固定项 | 变量 | 训练组数 | 说明 |
|---|---|---|---:|---|
| Lane sweep | `TARGET_SMID=0`、`kernel=BP9`、`bit=13` | `TARGET_LANEID=0,2,4,...,30` | 16 | stride 2 等距覆盖 warp lane；BP9 是本机 IV-B 中强但非饱和的 backward target。 |
| SM sweep | `TARGET_LANEID=0`、`kernel=BP9`、`bit=13` | `TARGET_SMID=0,8,16,...,120` | 16 | stride 8 等距覆盖已确认的 128 个 SM；避免使用更接近非有限边界的 BP8。 |

训练配置沿用 IV-B/IV-A：60M、seed 42、1000 update、micro batch size 256、gradient accumulation 2、有效 batch size 512、`--grad_clipping 1.0`、`TARGET_REGISTER=1`、`TARGET_OP=HMMA`、`TARGET_INSTR=-1`（覆盖 BP9 内全部 16 条 HMMA）、注入频率 1/10 steps、duration 1 step、不开 recompute。输出单独放到 `checkpoints/iv_e_mb256/lane/seed_42/`、`checkpoints/iv_e_mb256/sm/seed_42/` 和 `logs/iv_e_mb256/...`，不混入 IV-B 目录。

脚本与配置：

| 文件 | 作用 |
|---|---|
| `scripts/configs/iv_e.sh` | IV-E 配置（target、抽样 stride、16 组/部分校验），继承 IV-A/IV-B 训练形态。 |
| `scripts/experiments/run_iv_e_spatial_sweep.sh lane\|sm` | 跑一个部分；含 manifest 锁定、组数校验、逐组训练与断点续跑。 |
| `scripts/experiments/summarize_iv_e.py` | 汇总一个部分：逐组指标 + complete run 的 eval delta / parameter difference 均值、标准差、范围。 |

运行顺序：

```bash
# 0. IV-E 固定使用本机 IV-B 中强但非饱和的 BP9 bit13；BP8 bit13 留给 IV-F temporal 主线
# 1. 先跑 lane sweep：16 组，固定 SM=0，遍历 lane 0,2,...,30
bash scripts/experiments/run_iv_e_spatial_sweep.sh lane
# 2. 再跑 SM sweep：16 组，固定 lane=0，遍历 SM 0,8,...,120
bash scripts/experiments/run_iv_e_spatial_sweep.sh sm
```

观测指标：final eval loss、eval loss delta vs paired baseline、Parameter Difference、max gradient norm pre/post、max attention logits、max `rt`、trigger count、detected anomaly count、nonfinite metric count。分析时分别计算 lane sweep 和 SM sweep 的 final eval loss delta 与 parameter difference 的均值/标准差/范围（汇总脚本自动输出），并检查是否存在离群 lane 或离群 SM。如果所有结果都落在 paired baseline 可接受波动范围内，可支持“本机未观察到 lane/SM 空间敏感性”；如果某些 SM 明显更敏感，需要先排除该 SM 是否因为 kernel launch 没覆盖、热/频率波动、或 partial run 导致，再作为本机差异记录。注意：16 组抽样的统计功效弱于论文的 32+32，报告中应明确这一覆盖度差异。

本轮实验报告：[IV-E Spatial Effects 复现实验报告](iv_e_spatial_effects_experiment_report.md)

### IV-F 时间效应

duration部分

旧 duration 实验已经完成，但它使用固定单条 `TARGET_INSTR=312`，只覆盖 duration `1/3/5`，且退化幅度过弱；该结果保留为历史单指令口径报告，不再作为新版 IV-F duration 主结论。论文 IV-F 原文 duration 轴为：在 step 500 注入一次故障，持续 `1/3/5/7/9` 个连续 update step；论文给出的 eval loss 约为 `4.30/4.31/4.32/4.33/4.35`，并用 Fig.3 展示 duration=3 的单次注入后 `Rt` spike 和 training-loss bump。

新版 duration 重跑采用和 IV-F rate 新实验一致的 **BP8 + bit13 + all-HMMA first-register** 口径：`TARGET_FUNC_CONTAINS` 从 `logs/iv_a_mb256/kernel_manifest.tsv` 的 `BP8` 行读取，`TARGET_INSTR=-1` 覆盖该 kernel 内所有匹配 HMMA 指令，`TARGET_REGISTER=1`，`TARGET_BITMASK=1<<13`，`TARGET_SMID=0`，`TARGET_LANEID=0`，location 为 backward。训练配置沿用 mb256 主线：60M、1000 update、micro batch 256、total batch 512、warmup 1000、final evaluation only、不开 recompute。主实验先固定 seed 42，与 `checkpoints/iv_f_mb256/baseline/seed_42` 做 paired baseline；若需要跨 seed 误差棒，再补 `full` 模式，但必须先补齐 `seed_1337` 和 `seed_351344` 的 mb256 baseline。

本轮新增脚本：`scripts/configs/iv_f_duration_all_hmma.sh`、`scripts/experiments/run_iv_f_duration_all_hmma.sh`、`scripts/experiments/summarize_iv_f_duration_all_hmma.py`。输出独立放在 `checkpoints/iv_f_duration_all_hmma_mb256/` 与 `logs/iv_f_duration_all_hmma_mb256/`，不覆盖旧 `checkpoints/iv_f_mb256/duration/`。

```bash
# 最小 Fig.3 sanity：只跑 BP8、duration=3、seed 42，并导出 loss/Rt 时间序列
bash scripts/experiments/run_iv_f_duration_all_hmma.sh probe BP8

# 主 duration 轴：BP8、duration=1/3/5/7/9、seed 42
bash scripts/experiments/run_iv_f_duration_all_hmma.sh main BP8
```

汇总脚本输出三类文件：`iv_f_duration_all_hmma_summary.csv` 记录每个 run 的 final eval loss、paired delta、trigger window、detected count、gradient/attention/Rt 最大值和 parameter difference；`iv_f_duration_all_hmma_aggregate.csv` 按 duration 聚合均值/标准差；`iv_f_duration_all_hmma_fig3_timeseries.csv` 导出 duration=3、step 500 附近窗口的 `_step`、`loss`、`rt`、`rt_jump`、`gradient_norm_pre/post`、`trigger_nvbit`、`detected_anomaly`、`max_attention_logits`，可直接支撑 Fig.3 风格曲线。观测重点：final eval loss/paired delta 是否随 duration 增长，duration=3 是否出现 `Rt` spike 后 training-loss bump，以及更长 duration 是否延长恢复时间。

rate部分

IV-F rate 部分准备追加一轮 **single-seed all-HMMA first-register rerun**，用于修正旧 IV-F 中固定单条 `TARGET_INSTR=312` 的注入口径。新口径为：固定一个由 IV-B 选出的 backward kernel，`TARGET_FUNC_CONTAINS=<selected kernel>`，`TARGET_INSTR=-1` 覆盖该 kernel 内所有匹配 HMMA 指令，`TARGET_REGISTER=1` 表示 first input register，`TARGET_BITMASK=1<<13`，`TARGET_SMID=0`，`TARGET_LANEID=0`，duration 固定 1 step，seed 固定为 IV-A/IV-B 的 paired baseline seed 42。这里不是放开到所有 kernel，而是在选定 kernel 内注入每条 HMMA 的 first register。

IV-F 主 target 定为 `BP8 + bit13`：在 1/10 频率、`TARGET_INSTR=-1`、SM0/lane0 的已完成 IV-B 数据中，`BP8 bit13` final eval loss 为 `4.6890`，相对 paired baseline delta 为 `+0.3717`，是本机最强的 finite backward 退化组合，也比 `BP9 bit13` 的 `4.6054` / `+0.2881` 更接近论文 IV-F 1/10 频率的 loss 量级。`BP8 bit13` 已有少量 nonfinite metric（`nonfinite_metric_count=2`），因此新增 rate run 只补 `every_1`、`every_100`、`every_1000` 三组，并与 IV-B 中完全同口径的 `every_10` 结果合并分析；若 `every_1` 过早 NaN 饱和，则报告中把它标记为最高频率饱和点，低频仍按 `BP8 bit13` 解释。

运行口径：`serial` 模式一次串行跑完 `every_1`、`every_100`、`every_1000` 三组；`every_10` 直接复用 IV-B `checkpoints/iv_b_mb256/seed_42/bit_13/BP8`，前提是 seed、bit13、kernel label、`TARGET_INSTR=-1`、SM/lane/register、duration、batch 结构和 final evaluation 口径完全一致。分析时四个 rate 点合并为 `every_1`、`every_10`、`every_100`、`every_1000`。

脚本：新增 `scripts/experiments/run_iv_f_rate_all_hmma.sh`、`scripts/configs/iv_f_rate_all_hmma.sh` 和汇总脚本 `scripts/experiments/summarize_iv_f_rate_all_hmma.py`，输出独立放在 `checkpoints/iv_f_rate_all_hmma_mb256/` 与 `logs/iv_f_rate_all_hmma_mb256/`，不覆盖旧 IV-F。建议顺序如下：

```bash
# IV-F 主线使用 BP8；新增训练一次串行跑完 every_1、every_100、every_1000 三组
bash scripts/experiments/run_iv_f_rate_all_hmma.sh serial BP8
```

最终 rate 轴应覆盖 `every_1`、`every_10`、`every_100`、`every_1000`。其中 `every_1`、`every_100`、`every_1000` 由 `serial` 模式一次串行跑完；`every_10` 复用 IV-B 中相同 seed、bit13、kernel、`TARGET_INSTR=-1`、rate=10 的 `BP8` 结果。`run_iv_f_rate_all_hmma.sh` 在结束汇总时会把 `checkpoints/iv_b_mb256/seed_42/bit_13/BP8` 作为 `--reuse-every10-dir` 传给 `summarize_iv_f_rate_all_hmma.py`，summary CSV 中用 `source=iv_b_reuse` 显式标注该行，避免误认为它来自 `iv_f_rate_all_hmma_mb256` 新 campaign root。

本轮 duration all-HMMA 实验报告：[IV-F Duration Effects All-HMMA 复现实验报告](iv_f_duration_all_hmma_experiment_report.md)

本轮 rate all-HMMA 实验报告：[IV-F Rate Effects All-HMMA 复现实验报告](iv_f_rate_all_hmma_experiment_report.md)

### VI/VII 检测与恢复（Detection + Recomputation）

论文 Section VI 是检测算法本身（无独立实验），Section VII 评估检测与恢复效果。本地实现已在 `torchrun_main.py` 中就绪（`rt` 统计量、warm-up 基线、Eq. 13 判定、非有限梯度范数硬判定、recompute），算法与源码的逐条对照见 [detection_algorithm_analysis](detection_algorithm_analysis.md)。VI/VII 只需设计训练 campaign，**不需要修改训练器代码**：

- 随机 kernel 选择：复用 `--fi_nvbit_target_funcs`（每次触发窗口开始时从候选 function ID 中随机选一个，经 `TARGET_FUNC` 环境变量传给 NVBit 工具），候选 ID 直接从 `logs/iv_a_mb256/kernel_manifest.tsv` 的 `function_id_first_seen` 列读取；
- 随机故障持续 1–5 步：复用 `--fi_nvbit_duration_random --fi_nvbit_duration 5`；
- 平均 1/N 步触发一次：复用 `--fi_nvbit_trigger_rate N`；
- recompute 期间自动关闭注入（`trigger_nvbit=False`），与论文"recompute 时不重新激活故障"一致；
- 检测统计（tp/fp/tn/fn、correct/incorrect recompute）已由训练器写入 `summary.json`，逐步指标在 `metrics.jsonl`，汇总脚本离线计算 detection rate 与 recompute precision。

按"单 seed、砍掉重复实验"的原则降级（论文 VII-A 原为 α 对数扫描 × bit 9–12 × 3 seeds；VII-B 原为 60M×12 seeds / 350M×6 / 1.3B×3，均 10,000 步）：

#### VII-A 检测与恢复验证（60M，α=0.05 单点）——已跳过

最终计划跳过 VII-A campaign，原因：(1) 论文 VII-A 的 α 扫描（Fig.4 trade-off）已随降级砍掉，剩下的 α=0.05 单点验证与 VII-B 重复——VII-B 的 fi 与 fi_recompute 是同一注入 schedule 的配对对照，"恢复有效"论断由 VII-B 覆盖；(2) VII-C 的 detection-on 样本复用 VII-B fault-free baseline 的前 2,000 步，不再依赖 VII-A baseline，也不重复启动一组 detection-on 训练。

原 VII-A 设计（存档参考）：60M、seed 42、2,000 步、bit 9/10/11/12、rate 1/10、duration 1、backward、每次触发随机 backward kernel（BP1–BP9）、`TARGET_INSTR=-1`（kernel 内全部 HMMA）、first input register、SM0/lane0、α=0.05；baseline + 4 bit × {recompute, no_recompute} 共 9 组；指标为 detection rate、recompute precision、final eval loss delta、混淆矩阵。脚本 `scripts/experiments/run_vii_a_detection.sh` 保留但未运行。

#### VII-B 跨规模（只做 60M，即论文 Table I 的 60M 列）

论文 Table I 覆盖 60M/350M/1.3B 三档；本地单卡 RTX 4090 24GB 且单 seed，350M/1.3B 的 kernel name 映射与显存/时间成本过高，**本轮只做 60M**，大模型规模验证写入局限性。60M、seed 42、10,000 步、eval 每 1,000 步记录一次、α=0.05，三种条件各 1 组：

| 条件 | 故障配置 |
|---|---|
| baseline | 无注入，带检测路径 |
| FI（不恢复） | BP8/BP9 每次触发随机二选一、频率 1/100、duration 随机 1–5 步、bit 13，检测开启但不 recompute |
| FI + recompute | 同上 + `--fi_nvbit_recompute`，α=0.05 |

bit 与 kernel 的本地调整（决策记录）：论文口径为 bit 12 + 随机 kernel 对；本地按此口径完成的 FI run 未产生可测量退化（final eval loss Δ≈−0.002，单 seed 噪声内，仅 WandB 留存，作为论文口径 null result）。依据 IV-B 本地证据改为 **bit 13 + BP8/BP9**：bit 13 是最强非饱和 exponent bit；BP8/BP9 是 bit 13 下最强的两个 finite backward kernel（Δ 分别 +0.37/+0.29）；BP4 在 bit 13 下近乎免疫（Δ≈+0.0002），不宜作靶点。饱和风险可控：IV-F 已验证 BP8+bit13 在 rate 1/10（含 every-1）+ duration 1 下保持 finite，本轮 rate 1/100 + duration 1–5 总注入步数更少。工程修复：续跑时检测阈值 window_mean 不恢复导致阈值恒 0 的 bug 已修复（torchrun_main.py，随 training_state.json 保存/恢复）；VII-B baseline 续跑段 summary 中的 fp=2679 为修复前伪影，训练与 eval 不受影响。baseline 条件不再传 `--base_model_path` / `--compare_every`，避免以后从头重跑时在 step 10000 做无意义的 baseline 自比 baseline；FI 与 FI+recompute 仍保留 `--compare_every 10000` 用于对 paired baseline 做 parameter difference。指标：eval loss 曲线（每 1,000 步）+ final eval loss、相对 baseline delta、detection/recompute 统计、s/it、parameter difference。

本轮 VII-B 实验报告：[VII-B 60M Fault Injection and Recompute 复现实验报告](vii_b_scale_60m_experiment_report.md)。

数据说明（与论文的口径偏差）：本地 C4 子集仅 ~2M 条序列（1 epoch = 3,906 update steps），不足以支撑 10,000 步。通过给 `OfflinePreprocessedDataset` 新增 `repeat` 参数（`--train_data_repeat`，索引按 epoch 内 batch 数取模，各 pass 顺序一致、兼容断点续跑）复用数据，`VI_VII_B_DATA_REPEAT=3`（~2.56 个 epoch，可支撑 11,718 步 > 10,000 步）。论文使用 C4 全量、实际不重复采样，此偏差需在报告中写明。VII-A（2,000 步 = 1.02M 条）不受数据量限制，`repeat=1`。

#### VII-C runtime overhead

对比 detection on/off 的 s/it（论文报告约 1%）：两个样本都必须是 fault-free，只比较检测代码路径本身。detection-on 复用 `checkpoints/vii_b_mb256_3906/baseline/seed_42` 中保存的 VII-B baseline 首段数据；该目录包含完整 step 1-3906 的 `metrics.jsonl`，其中 step 1-2000 无缺失，且 `compare_every=10000` 在前 2,000 步不会触发 baseline 自比 baseline。detection-off 只需新增 1 组 `torchrun_main_no_detection.py`，配置与 VII-B baseline 前 2,000 步匹配：60M、seed 42、micro batch 256、total batch 512、AdamW、bf16、grad clipping 1.0、`eval_every=1000`、`save_every=10000`、`train_data_repeat=1`。`repeat=1` 是复用样本的实际口径；VII-B 10,000 步后续使用 `repeat=3` 只是为了解决 3,906 步后本地数据耗尽，前 2,000 步不需要 repeat。

VII-C 汇总只比较两边 step `1-2000` 的 `_time` 差分，并过滤前 100 步 warmup；当前 detection-on 复用样本在 step `101-2000` 上的 median s/it 为 `1.642826`。最终 overhead 由 `run_vii_c_overhead.sh` 在补完 no-detection 后写入 `logs/vii_c_mb256/overhead_summary.csv`。

本轮 VII-C 实验报告：[VII-C Runtime Overhead 复现实验报告](vii_c_runtime_overhead_experiment_report.md)。已完成 no-detection 匹配 run；`overhead_summary.csv` 显示 detection-on median s/it `1.642826`、detection-off median s/it `1.674169`，按脚本口径 overhead 为 `-1.87%`，即本轮未观察到论文约 `+1%` 的正向 slowdown，检测路径开销低于单次运行噪声量级。

#### 运行顺序

```bash
# VII-B：baseline 已完成；重跑 fi 与 fi+recompute 各 1 组 10,000 步（bit 13 + BP8/BP9），串行约 6–7 小时
bash scripts/experiments/run_vii_b_scale_60m.sh

# VII-C：复用 VII-B 3906 baseline 前 2,000 步作为 detection-on，只补跑 1 组 no-detection
bash scripts/experiments/run_vii_c_overhead.sh
```

新增文件：`scripts/configs/vi_vii.sh`、`scripts/experiments/run_vii_a_detection.sh`、`scripts/experiments/run_vii_b_scale_60m.sh`、`scripts/experiments/run_vii_c_overhead.sh`、`scripts/experiments/summarize_vii_a.py`、`scripts/experiments/summarize_vii_b.py`。输出独立放在 `checkpoints/vii_a_mb256/`、`checkpoints/vii_b_mb256/`、`checkpoints/vii_c_mb256/` 与 `logs/vii_*_mb256/`，不覆盖 IV 系列结果。

注意：function ID 取自 IV-A profiling（`TARGET_FUNC_CONTAINS=gemm` 口径、seed 42、mb256），VII 各 run 必须保持同一模型/seed/batch 结构，否则 kernel launch 顺序变化会导致 ID 错位；`run_vii_*.sh` 启动时会校验 manifest 存在并打印实际使用的 ID 列表，跑完后应核对日志中 `[NVBit] Enable FI ... nvbit_target_func: X` 的取值都在候选集合内。

## 四、经验总结

1. 实验前研究仓库中各文件作用，仔细阅读README.md，重点关注版本问题；慎依赖AI指令下安装下载！
2. 不要着急跑冒烟测试，先熟悉数据抓取分析流程
3. 





