# NVBit 故障注入冒烟测试报告

## 1. 实验目的

本实验用于验证当前 RTX 4090 单 GPU 训练环境中，NVBit 1.7.6 故障注入工具能否完成以下闭环：

1. 通过 `LD_PRELOAD` 正常加载；
2. 识别 PyTorch/CUTLASS 生成的 BF16 GEMM/HMMA kernel；
3. 响应训练器发送的 `SIGUSR1` / `SIGUSR2` 信号；
4. 在指定 SASS 指令、SM、lane 和寄存器上执行位翻转；
5. 使 GPU 计算结果或训练参数产生可验证的数值变化；
6. 在注入后继续完成训练并保存完整 checkpoint。

测试分为最小矩阵乘法注入和 15 步短训练注入两部分。

## 2. 实验环境

- 操作系统：Ubuntu 24.04（云端训练环境）
- GPU：NVIDIA GeForce RTX 4090，单卡
- Python：3.12.3
- PyTorch：2.9.0
- Transformers：4.56.2
- 训练数据类型：bfloat16
- NVBit：1.7.6
- 运行模式：离线、单 GPU
- 故障注入工具：`nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so`

## 3. 工具修复与注入目标确认

### 3.1 `TARGET_FUNC=-1` 通配语义修复

原工具在 kernel launch 时仅执行以下判断：

```cpp
enable = (fid == active_target_func.load());
```

因此 `TARGET_FUNC=-1` 不会匹配任何动态分配的 function ID。修复后逻辑为：

```cpp
int active_fid = active_target_func.load();
enable = (active_fid < 0 || fid == active_fid);
```

修复后的含义是：负数表示启用所有已经匹配并插桩的函数，非负数仍表示只启用指定 function ID。工具同时增加了 launch 诊断日志，用于记录 function ID、当前目标和插桩启用状态。

### 3.2 避免依赖动态 function ID

多次运行表明 NVBit function ID 会随 kernel 首次出现顺序变化。例如，同类 kernel 在不同运行中可能由 function 28 变为 function 24。因此最终短训练采用：

```text
TARGET_FUNC=-1
```

并通过固定 `TARGET_INSTR=312` 缩小插桩范围，避免使用不稳定的 function ID。

v3 日志确认 function 108 中实际存在目标指令：

```text
[NVBit Fault Injector] inspecting ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn - num instrs: 1056 - count: 108
[NVBit Fault Injector] HMMA.1688.F32.BF16 R0, R176, R192.reuse, R0 ;idx: 312; func: 108
```

## 4. 最小矩阵乘法注入

证据日志：

```text
logs/nvbit_smoke/matmul_injection_all_funcs.log
```

关键结果：

```text
NVBit v1.7.6 Loaded
TARGET_FUNC=-1
TARGET_OP=HMMA
Launch function ID 85, active target -1, instrumentation enabled
baseline_sum: -32109.0234375
fault_sum: 21516269912064.0
exactly_equal: False
changed_elements: 1048576
max_abs_difference: 192199786496.0
finite_baseline: True
finite_fault: True
```

该结果证明 NVBit 已成功加载，HMMA kernel 能被识别和启用，寄存器位翻转会实际改变 GPU 矩阵乘法输出。

## 5. v3 短训练注入配置

v3 使用以下核心训练配置：

| 配置项 | 取值 |
|---|---:|
| 模型 | LLaMA 60M |
| 参数量 | 58.0736M |
| Seed | 351344 |
| Micro batch size | 32 |
| Gradient accumulation | 16 |
| 有效总 batch size | 512 |
| 最大序列长度 | 256 |
| 数据类型 | bfloat16 |
| 优化器 | AdamW |
| 最大学习率 | 1e-3 |
| Warmup steps | 1000 |
| 训练更新步数 | 15 |
| Checkpoint 间隔 | 15 |
| 注入位置 | backward |
| 指定注入步 | 11（按当前调度在记录的 Step 13 执行） |
| 注入持续时间 | 1 个更新迭代 |

NVBit 环境变量为：

```text
TARGET_FUNC=-1
TARGET_INSTR=312
TARGET_OP=HMMA
TARGET_SMID=0
TARGET_LANEID=0
TARGET_REGISTER=1
TARGET_BITMASK=4096
TARGET_FUNC_CONTAINS=gemm
```

输出位置：

```text
logs/nvbit_smoke/fi_allfunc_instr312_v3.log
checkpoints/nvbit_smoke/fi_allfunc_instr312_v3
```

## 6. v3 运行结果

### 6.1 NVBit 插桩与触发

日志共出现 2,336 条：

```text
active target -1, instrumentation enabled
```

插桩禁用记录为 0。实际启用的训练 kernel 包括 function 24、44、52、69、108、109 和 110；其中 function 108 明确包含第 312 条 HMMA 指令。

结构化指标在 Step 13 记录：

```text
trigger_nvbit: 1.0
```

这与训练器在前一步输出的以下日志对应：

```text
[NVBit] Enable FI for the next 1 iterations
```

### 6.2 训练完成与 checkpoint 完整性

v3 正常完成 15 个更新步骤，日志末尾包含：

```text
Training finished
Script finished successfully
Rank 0 finished successfully
```

未发现 Traceback、Segmentation fault、CUDA error、OOM 或训练指标中的 NaN/Inf。

最终 checkpoint 为：

```text
checkpoints/nvbit_smoke/fi_allfunc_instr312_v3/model_15/
├── config.json
├── model.safetensors
├── optimizer.pt
└── training_state.json
```

`training_state.json` 记录：

| 字段 | 取值 |
|---|---:|
| `global_step` | 15 |
| `update_step` | 15 |
| `tokens_seen` | 1,499,851 |
| `batch_idx` | 239 |

### 6.3 与无实际插桩运行的逐步指标比较

使用 v2 作为 15 步对照。v2 的训练配置、数据、seed 和训练步数与 v3 相同，但其目标 function 28 在该次运行中从未 launch：2,336 次 launch 全部显示 `instrumentation disabled`，所以它可作为同长度的未实际注入对照。

比较 v2 与 v3 的前 15 步结构化指标：

| 指标 | 最大绝对差异 |
|---|---:|
| Loss | 0.0 |
| Perplexity | 0.0 |
| Gradient norm before clipping | 0.0 |
| Gradient norm after clipping | 0.0 |
| Weight norm | 0.0 |
| Learning rate | 0.0 |

该结果不表示故障未执行。这里的 loss 在 backward 注入前已经由 forward 得到；梯度范数和权重范数又以 bfloat16/有限精度记录，小幅扰动可能被日志量化掩盖。是否发生参数变化需要比较未舍入的 checkpoint 内容。

### 6.4 最终参数变化

两份 15 步模型权重的 SHA-256 为：

```text
v2（未实际插桩）：903e3e7c58993fba30f31e2899700c7202b4e96d9111f1811f7c66306b79a8ec
v3（真实插桩）  ：81df69a96f3ea37eb8a3b93313dde24230980820b5b60e63a94cf7b9fe05b122
```

两份 safetensors 的 header 完全相同，因此哈希差异不是张量布局或元数据变化。逐张量解析比较结果为：

| 项目 | 结果 |
|---|---:|
| 模型张量总数 | 83 |
| 发生变化的张量数 | 58 |
| 参数元素总数 | 58,073,856 |
| 发生变化的参数元素数 | 68,015 |
| 最大绝对参数差异 | 3.0517578125e-05 |

变化覆盖 embedding、LM head、MLP projection 等多个张量。例如：

| 张量 | 变化元素数 | 最大绝对差异 |
|---|---:|---:|
| `lm_head.weight` | 14,450 | 3.0517578125e-05 |
| `model.embed_tokens.weight` | 10,118 | 3.0517578125e-05 |
| `model.layers.6.mlp.gate_proj.weight` | 1,234 | 1.52587890625e-05 |
| `model.layers.4.mlp.down_proj.weight` | 1,233 | 3.0517578125e-05 |

该结果证明 v3 的 backward HMMA 位翻转已经传播到梯度更新和最终模型参数，而不仅仅是 Python 侧记录了触发标志。

## 7. 通过标准与判定

| 判据 | 结果 |
|---|---|
| NVBit 1.7.6 正常加载 | 通过 |
| 最小矩阵乘法输出被位翻转改变 | 通过 |
| `TARGET_FUNC=-1` 对实际 kernel 启用插桩 | 通过 |
| 目标第 312 条 HMMA 指令存在 | 通过 |
| Python 侧在预定窗口触发注入 | 通过 |
| 短训练正常完成 15 步 | 通过 |
| 最终 checkpoint 完整 | 通过 |
| 最终训练参数相对未注入对照发生变化 | 通过 |
| 无崩溃、CUDA error、NaN/Inf | 通过 |

因此，本次 NVBit 故障注入冒烟测试 **通过**。

## 8. 检测器结果与结论边界

v3 的 `summary.json` 将这次触发记为：

```json
"fn": 1
```

其含义是训练器知道该步启用了 NVBit，但当前异常检测阈值没有对这次轻微扰动告警。它不表示 NVBit 注入失败；checkpoint 已证明参数确实改变。此次测试验证的是故障注入链路，而不是检测器对所有位翻转的召回率。

本次扰动的最大参数差异仅约 `3.05e-05`，常规 loss 和梯度范数日志未显示差异，这也说明单次低强度位翻转可能成为静默数据破坏。后续检测/缓解实验应使用多组 bitmask、注入位置、指令和随机种子，统计检测率、误报率以及重算后的恢复效果，不能根据本次单样本推断检测器总体性能。
