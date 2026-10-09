# IV-A 前置工程验证报告：mb256 baseline、chunked loss 与 kernel profiling

## 1. 报告定位

本报告只记录 IV-A 正式 bit-position sweep 之前的工程预实验结果，不讨论论文 IV-A 的 exponent/mantissa/sign bit 敏感性结论。

本阶段要回答三个工程问题：

1. 论文 60M 实验所需的 `micro batch size = 256`、`gradient accumulation = 2` 能否在本地 RTX 4090 环境中跑完 1,000 update steps。
2. chunked causal LM loss 修复是否解决原先 mb256 显存问题，并保持训练、保存、最终 evaluation 链路完整。
3. 在 mb256 结构下，本机 forward/backward 能被 NVBit profile 到多少个 GEMM/HMMA kernel，后续 kernel 标签应以什么为准。

附带记录：`bit_13` 和 `bit_14` sanity run 已完成，可用于验证“工程修复后 FI 训练也能完整跑通”，但它们仍只是前置验证，不替代完整 IV-A bit sweep。

## 2. 两个 mb256 checkpoint 目录的区别

当前 `checkpoints` 下有两个带 `mb256` 的目录，它们不是重复实验，分工不同：

| 目录 | 用途 | 主要内容 |
|---|---|---|
| `checkpoints/iv_f_mb256/` | mb256 paired baseline 根目录 | 目前保存 `baseline/seed_42`。这是无故障 1,000-step baseline，由 `scripts/experiments/run_iv_paired_baseline.sh` 生成。虽然目录名带 `iv_f`，但 `scripts/configs/iv_a.sh` 中 `IV_A_BASELINE_DIR` 就指向这里，所以它是后续 IV-A bit sweep 的 paired baseline。 |
| `checkpoints/iv_a_mb256/` | IV-A 前置 profile 与 FI sanity 根目录 | 保存 `profile/forward`、`profile/backward` 两个 12-step profiling run，以及 `seed_42/bit_13/bit_13`、`seed_42/bit_14/bit_14` 两个 1,000-step FI sanity run。 |

具体记录如下：

| 路径 | 记录内容 | 是否保留 optimizer |
|---|---|---|
| `checkpoints/iv_f_mb256/baseline/seed_42/` | baseline 的 `metrics.jsonl`、`summary.json`、`run_config.json`、`model_1000/` | 是，`model_1000/optimizer.pt` 存在 |
| `checkpoints/iv_a_mb256/profile/forward/` | forward profiling 的 `metrics.jsonl`、`summary.json`、`run_config.json`、`model_12/` | 是 |
| `checkpoints/iv_a_mb256/profile/backward/` | backward profiling 的 `metrics.jsonl`、`summary.json`、`run_config.json`、`model_12/` | 是 |
| `checkpoints/iv_a_mb256/seed_42/bit_13/` | bit13 group manifest 与子 run 目录 | group 层只记录 manifest |
| `checkpoints/iv_a_mb256/seed_42/bit_13/bit_13/` | bit13 FI sanity 的 `metrics.jsonl`、`summary.json`、`run_config.json`、`model_1000/` | 否，`IV_A_KEEP_OPTIMIZER_PT=0`，完成后移除了 optimizer |
| `checkpoints/iv_a_mb256/seed_42/bit_14/` | bit14 group manifest 与子 run 目录 | group 层只记录 manifest |
| `checkpoints/iv_a_mb256/seed_42/bit_14/bit_14/` | bit14 FI sanity 的 `metrics.jsonl`、`summary.json`、`run_config.json`、`model_1000/` | 否，`IV_A_KEEP_OPTIMIZER_PT=0`，完成后移除了 optimizer |

因此：`iv_f_mb256` 现在主要是 baseline 证据；`iv_a_mb256` 现在主要是 profiling 与 bit13/bit14 工程验证证据。后续正式 IV-A sweep 的其他 bit 也应继续落在 `checkpoints/iv_a_mb256/seed_42/...` 下，并与 `checkpoints/iv_f_mb256/baseline/seed_42` 配对比较。

## 3. 工程修复：chunked causal LM loss

原先 micro batch 256 的主要问题是训练 loss 会一次性构造完整 `[batch, sequence, vocab]` logits，显存压力过高。当前 `training/modeling_llama.py` 已改为：

1. `LlamaForCausalLM._chunked_lm_loss()` 将 hidden states 展平为 token 维度。
2. 按 `loss_chunk_size=2048` 分块调用 `lm_head`。
3. 每块使用 `F.cross_entropy(..., reduction="sum")` 累积 loss。
4. 最后除以有效 token 数，得到与普通 causal LM loss 同口径的平均 loss。
5. `forward()` 在训练传入 `labels` 时只返回 loss，不保留完整 logits。

这项修复的直接效果是：`batch_size=256`、`gradient_accumulation=2`、`total_batch_size=512` 的 baseline、profiling run 和 FI sanity run 都能跑完，并完成最终 evaluation 与 checkpoint 保存。bit14 虽然产生 NaN，但训练脚本仍正常保存 checkpoint 和 summary，说明工程链路本身没有被非有限数值打断。

## 4. mb256 baseline 结果

baseline 路径：`checkpoints/iv_f_mb256/baseline/seed_42`。

完整性检查：

| 项目 | 结果 |
|---|---|
| `metrics.jsonl` | 存在，3,001 条记录 |
| `summary.json` | 存在 |
| `run_config.json` | 存在 |
| `model_1000/model.safetensors` | 存在 |
| `model_1000/optimizer.pt` | 存在 |
| `model_1000/training_state.json` | `global_step=1000`，`update_step=1000` |
| 日志结束状态 | `Training finished`，`Final eval loss`，`Script finished successfully` |

关键指标：

| 指标 | 数值 |
|---|---:|
| Final eval loss | 4.317315101623535 |
| Final eval perplexity | 74.9870252289692 |
| Final train loss | 4.267379283905029 |
| Max train loss | 10.471338272094728 |
| Max gradient norm before clipping | 2.28125 |
| Max gradient norm after clipping | 1.0078125 |
| Max attention logits | 39.75 |
| Max `rt/rt` | 0.000644683837890625 |
| Triggered FI steps | 0 |
| Detected anomalies | 0 |
| Total time | 1648.98 s，约 27.5 min |
| Tokens seen | 99,994,972 |

这说明无故障 mb256 训练链路已经稳定可用。后续 IV-A 的 paired delta 应以这条 baseline 为基准：

```text
eval_loss_delta = FI final eval loss - 4.317315101623535
```

## 5. Kernel profiling 结果

profiling 证据文件：`logs/iv_a_mb256/kernel_manifest.tsv`。

本机 mb256 profiling 得到的 kernel 数：

| 位置 | Kernel 数 | 本机标签 |
|---|---:|---|
| forward | 4 | FP1-FP4 |
| backward | 9 | BP1-BP9 |

profile run 本身也完整结束：

| Run | 保存路径 | 训练步数 | Total time | 结束状态 |
|---|---|---:|---:|---|
| forward profile | `checkpoints/iv_a_mb256/profile/forward` | 12 | 66.60 s | 成功 |
| backward profile | `checkpoints/iv_a_mb256/profile/backward` | 12 | 128.53 s | 成功 |

这个结果和论文描述的 5 个 forward、10 个 backward kernel 不完全一致。这里不需要强行补齐到论文数量，因为 kernel 数会受 GPU、PyTorch/cuBLAS、NVBit 版本、batch 结构和实际矩阵形状影响。后续本机报告应以 `kernel_manifest.tsv` 为准：forward 只写 FP1-FP4，backward 只写 BP1-BP9。

manifest 中与当前固定 target 名称相同的是 BP4：

| 标签 | location | launch_count | HMMA 指令数 | manifest target_instr | kernel name |
|---|---|---:|---:|---:|---|
| BP4 | backward | 96 | 128 | 354 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` |

注意：manifest 的 `target_instr=354` 是 profiling 脚本为 kernel 诊断选出的代表指令；当前 bit sweep 配置实际使用的是 `TARGET_INSTR=312`。两者不能混写。

## 6. bit 13 / bit 14 sanity run 对工程修复的验证

bit13 路径：`checkpoints/iv_a_mb256/seed_42/bit_13/bit_13`。bit14 路径：`checkpoints/iv_a_mb256/seed_42/bit_14/bit_14`。

两次 run 均已完整结束，可作为工程修复后的 FI 训练链路验证。除 bit position / bitmask 外，二者配置相同：`batch_size=256`、`gradient_accumulation=2`、`total_batch_size=512`、backward FI、`TARGET_FUNC_CONTAINS=ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn`、`TARGET_INSTR=312`、`trigger_rate=10`、`duration=1`。

| 指标 | bit 13 | bit 14 |
|---|---:|---:|
| Bitmask | 8192 | 16384 |
| Field | exponent | exponent |
| Status | complete | complete |
| Trigger count | 91 | 91 |
| Trigger schedule | first=2, last=1000, ok=True | first=2, last=1000, ok=True |
| Final eval loss | 4.315753936767578 | NaN |
| Eval delta vs mb256 baseline | -0.0015611648559570312 | NaN |
| Final parameter L2 difference | 56.28740259180076 | NaN |
| Max finite gradient norm before clipping | 2.28125 | 1.625 |
| Max finite attention logits | 47.25 | 1.2734375 |
| Max finite `rt/rt` | 0.00064849853515625 | 0.0 |
| Nonfinite metric count | 0 | 24,956 |
| Detected count | 0 | 999 |
| Confusion count | TP=0, FP=0, TN=909, FN=91 | TP=91, FP=908, TN=1, FN=0 |

工程解释：

- chunked loss 修复不只让无故障 baseline 跑通，也让带 NVBit backward FI 的 mb256 run 跑通。
- 两个 sanity run 的 `trigger_count=91` 均与 seed 42 下脚本预期一致，说明 Python 侧故障调度在 1,000 steps 内完整执行。
- bit13 造成了非零参数偏离，但最终 eval loss 没有恶化。
- bit14 在第一次触发 step 2 后立即污染梯度、权重和 `rt`：step 2 的 `gradient_norm_pre/post`、`weight_norm`、`rt/rt` 已为 NaN；step 3 起训练 loss、perplexity 和 attention metrics 持续为非有限；最终 eval loss 为 NaN。
- bit14 证明当前 mb256 + NVBit 链路确实能够产生强 SDC 现象。此前 bit13 与旧 IV-F 未出现明显 loss 恶化，更可能是 bit / target 组合强度问题，而不是工程修复把故障“消掉”了。
- bit14 的检测器表现为 TP=91、FN=0，但 FP=908 很高。原因是模型从 step 2 后基本持续非有限，后续无触发 step 也被判为 anomaly；这里应记录为非有限传播后的检测副作用，而不是正常阈值质量评估。

## 7. 当前结论

本阶段的结论限于工程层面：

1. `batch_size=256 / accumulation=2` 的 baseline 已完整跑通，说明 chunked causal LM loss 修复有效。
2. mb256 下的 forward/backward profiling 已完成，本机实际 kernel 标签为 FP1-FP4、BP1-BP9。
3. `bit_13` sanity run 已完整跑通，91 次触发全部记录到 metrics，最终模型与 baseline 存在非零参数差异，但 eval loss 没有恶化。
4. `bit_14` sanity run 已完整跑通，并在 step 2 后立即出现 NaN 传播，最终 eval loss 为 NaN；这说明当前本地注入链路确实能制造强 SDC 失效。
5. 正式 IV-A 串行 sweep 应跳过已完成的 bit13 和 bit14，并在最终汇总时把 `bit_13`、`bit_14`、`bits_00_15`、`bits_16_31` 的 checkpoint roots 合并分析。
6. 这些结果支持“可以进入正式 IV-A bit sweep”的工程判断，但仍不构成完整 IV-A bit-position sensitivity 结论。

## 8. 证据文件

| 类型 | 路径 |
|---|---|
| IV-A 配置 | `scripts/configs/iv_a.sh` |
| Baseline 脚本 | `scripts/experiments/run_iv_paired_baseline.sh` |
| Profiling 脚本 | `scripts/experiments/profile_iv_a_kernels.sh` |
| Bit13 脚本 | `scripts/experiments/run_iv_a_bit_sensitivity.sh` |
| Chunked loss 实现 | `training/modeling_llama.py` |
| mb256 baseline log | `logs/iv_a_mb256/baseline/seed_42.log` |
| mb256 baseline checkpoint | `checkpoints/iv_f_mb256/baseline/seed_42/model_1000/` |
| mb256 baseline metrics | `checkpoints/iv_f_mb256/baseline/seed_42/metrics.jsonl` |
| profile logs | `logs/iv_a_mb256/profile/profile_forward.log`，`logs/iv_a_mb256/profile/profile_backward.log` |
| kernel manifest | `logs/iv_a_mb256/kernel_manifest.tsv` |
| bit13 metrics | `checkpoints/iv_a_mb256/seed_42/bit_13/bit_13/metrics.jsonl` |
| bit13 summary | `checkpoints/iv_a_mb256/seed_42/bit_13/bit_13/summary.json` |
| bit13 raw summary | `logs/iv_a_mb256/seed_42/bit_13/iv_a_summary_raw.csv` |
| bit14 metrics | `checkpoints/iv_a_mb256/seed_42/bit_14/bit_14/metrics.jsonl` |
| bit14 summary | `checkpoints/iv_a_mb256/seed_42/bit_14/bit_14/summary.json` |
| bit14 raw summary | `logs/iv_a_mb256/seed_42/bit_14/iv_a_summary_raw.csv` |
