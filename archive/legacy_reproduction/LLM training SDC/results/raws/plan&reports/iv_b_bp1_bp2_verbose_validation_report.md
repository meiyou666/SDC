# IV-B BP1/BP2 verbose 短验证报告

## 1. 验证目的

本验证针对 IV-B bit14 批次中 BP1 与 BP2 出现逐步 metrics 除 `_time` 外完全一致的问题。需要判断该现象是 kernel filter / manifest 映射错误，还是 bit14 + all-HMMA 注入过强导致两个不同 kernel 进入相同的数值饱和路径。

## 2. 验证配置

验证脚本：`scripts/experiments/validate_iv_b_bp1_bp2_verbose.sh`。

| 项目 | 配置 |
|---|---|
| Kernel labels | BP1, BP2 |
| Bit | 14 |
| Bitmask | 16384 |
| Target register | `TARGET_REGISTER=1` |
| Target instruction | `TARGET_INSTR=-1`，该 kernel 内所有 HMMA 指令 |
| Verbose | `TOOL_VERBOSE=1` |
| FI location | backward |
| FI step | 强制 update step 2 |
| Exit after | 5 update steps |
| Final evaluation | disabled |

输出位置：

| 类型 | 路径 |
|---|---|
| Logs | `logs/iv_b_mb256/validation/bp1_bp2_bit14_all_hmma_verbose/` |
| Checkpoints | `checkpoints/iv_b_mb256/validation/bp1_bp2_bit14_all_hmma_verbose/` |
| Verbose summary | `logs/iv_b_mb256/validation/bp1_bp2_bit14_all_hmma_verbose/verbose_validation_summary.tsv` |

## 3. 插桩证据

`verbose_validation_summary.tsv` 显示 BP1 和 BP2 实际 inspected kernel 不同，HMMA 指令集合也不同。

| Run | Inspected kernel | HMMA idx count | HMMA idx fingerprint | HMMA idx |
|---|---|---:|---|---|
| BP1 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nt_align8` | 16 | `78cf98b5d1c96153` | 443,450,457,463,469,473,475,479,482,485,489,494,497,499,501,503 |
| BP2 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` | 16 | `c98b13b85542da15` | 480,487,494,501,509,516,522,527,531,533,535,539,540,541,543,545 |

日志中也能直接看到：

- BP1 的 `TARGET_FUNC_CONTAINS` 是 `...64x64_32x6_nt_align8...`，并 inspecting 同名 kernel。
- BP2 的 `TARGET_FUNC_CONTAINS` 是 `...64x64_32x6_nn_align8...`，并 inspecting 同名 kernel。
- 两者均为 `TARGET_INSTR=-1`，即所有匹配 HMMA 指令均被插桩。

因此，当前证据不支持“BP1/BP2 实际打到了同一个 kernel”的怀疑；kernel disambiguation 和 manifest 映射在这次验证中是正常的。

## 4. 训练指标结果

两个 5-step 验证 run 均完整结束：

| Run | Status | Confusion count | Total time |
|---|---|---|---:|
| BP1 | complete | TP=1, FP=0, TN=4, FN=0 | 36.01 s |
| BP2 | complete | TP=1, FP=0, TN=4, FN=0 | 24.32 s |

两者的 metrics 除 `_time` 外完全一致：

| Step | trigger_nvbit | detected_anomaly | gradient_norm_pre | gradient_norm_post | loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 1.625 | 0.9921875 | 10.470211029052734 |
| 2 | 1 | 1 | Inf | 0.0 | 10.468757152557373 |
| 3 | 0 | 0 | 1.71875 | 0.9921875 | 10.466224193572998 |
| 4 | 0 | 0 | 1.6875 | 0.984375 | 10.468598365783691 |
| 5 | 0 | 0 | 1.6796875 | 0.9921875 | 10.471520900726318 |

这个结果说明 BP1/BP2 在 step 2 虽然来自不同 kernel 的不同 HMMA 指令集合，但最终暴露给训练循环的 aggregate gradient norm 都变成了 `Inf`。随后 `clip_grad_norm_` 后的记录为 `gradient_norm_post=0.0`，非触发 step 又回到正常有限值。

## 5. 结论

本次短验证支持以下判断：

1. BP1/BP2 的 kernel filter 和 NVBit verbose 插桩证据是区分开的，不是同一个 kernel 被重复命中。
2. BP1/BP2 的 HMMA instruction set 确实不同，idx fingerprint 分别为 `78cf98b5d1c96153` 和 `c98b13b85542da15`。
3. 两者 metrics 除 `_time` 外完全一致，主要原因更可能是 bit14 + all-HMMA 注入过强，使触发 step 的总梯度范数都饱和为 `Inf`，再经 clipping 变成 `gradient_norm_post=0.0`，导致后续训练轨迹等价。
4. 因此，当前 IV-B 主实验中 BP1/BP2 的完全同轨迹不应解释为 kernel mapping 错误；但也不应作为细粒度 kernel sensitivity 排序证据，只能表述为“在 bit14 all-HMMA 口径下二者均进入相同的 Inf-gradient 饱和路径”。

## 6. 后续建议

继续完成 IV-B 的 bit10 和 bit13 批次，用较弱 bit 观察 BP1/BP2 是否分化。如果 bit10/bit13 下 BP1/BP2 仍除 `_time` 外完全一致，再重新检查更细粒度的 target disambiguation；如果开始分化，则可以确认 bit14 的一致性主要来自数值饱和。

正式 IV-B 报告中建议把 bit14 结果分为两类解释：NaN kernel 和 Inf-gradient-but-recoverable kernel。BP1/BP2 应放在后者，并注明 total gradient norm 对不同 kernel 的强故障幅度没有区分能力。
