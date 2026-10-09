# 环境稳定性冒烟测试报告

## 1. 实验目的

本实验用于验证当前单 GPU 离线训练环境在固定模型、固定数据、固定超参数和固定随机种子的条件下，能否重复得到一致的训练轨迹与最终模型参数，为后续 NVBit 故障注入和检测/重算实验建立可信的无故障基线。

测试方法为：使用相同配置独立运行两次 LLaMA 60M 基线训练，每次执行 100 个参数更新步骤；分别保存日志、结构化指标和最终 checkpoint，然后比较两次运行的配置、逐步指标、模型权重、优化器状态及最终验证结果。

## 2. 实验环境

- 操作系统：Ubuntu 24.04（云端训练环境）
- GPU：NVIDIA GeForce RTX 4090，单卡，约 24 GiB 显存
- Python：3.12.3
- PyTorch：2.9.0
- Transformers：4.56.2
- Tokenizers：0.22.2
- Datasets：4.1.1
- NumPy：2.3.3
- Safetensors：0.8.0
- Loguru：0.7.3
- 运行模式：单 GPU，`cuda:0`
- 网络模式：离线；tokenizer 与 C4 数据均从本地路径加载
- NVBit：本实验未启用故障注入，且未加载故障注入 `LD_PRELOAD`

依赖版本证据位于：

- `logs/stability/environment.txt`
- `logs/stability/pip-freeze.txt`

## 3. 实验配置

两次正式运行除 `name` 和 `save_dir` 外，其余配置完全一致。

| 配置项 | 取值 |
|---|---:|
| 模型 | `llama_60m` |
| 参数量 | 58.0736M |
| 随机种子 | 351344 |
| 数据集 | 本地 C4 子集 |
| Tokenizer | 本地 T5 tokenizer |
| 最大序列长度 | 256 |
| Micro batch size | 32 |
| Gradient accumulation | 16 |
| 有效总 batch size | 512 |
| 优化器 | AdamW |
| 学习率 | 1e-3 |
| 调度器 | Cosine |
| Warmup steps | 1000 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| 数据类型 | bfloat16 |
| 更新步数 | 100 |
| Checkpoint 间隔 | 100 |
| 最终验证 | 启用 |
| 激活 checkpointing | 关闭 |
| 故障注入 | 关闭 |

两次运行的输出目录分别为：

```text
checkpoints/stability/run_1
checkpoints/stability/run_2
```

终端日志分别为：

```text
logs/stability/run_1.log
logs/stability/run_2.log
```

## 4. 实验前问题及处理

### 4.1 Tokenizer 配置兼容性

最初下载的 `tokenizer_config.json` 含有较新 Transformers 版本使用的列表形式 `extra_special_tokens` 字段，而训练环境的 Transformers 4.56.2 将该字段按字典处理，导致：

```text
AttributeError: 'list' object has no attribute 'keys'
```

清理不兼容的新版元数据后，本地 `T5TokenizerFast` 能够在强制离线模式下正常加载。

### 4.2 GPU 显存不足

原始 micro batch size 256 在构造 `[batch, sequence, vocab]` logits 时触发 CUDA OOM。随后将 micro batch size 降为 32，并将梯度累积提高到 16，使有效总 batch size 继续保持 512。

在正式测试前执行了 1 步显存探测：

```text
checkpoints/stability/memory_probe_bs32/model_1
logs/stability/memory_probe_bs32.log
```

探测运行正常完成并成功保存 checkpoint。

## 5. 通过标准

环境稳定性测试采用以下判据：

1. 两次运行均正常完成 100 个更新步骤，无未处理异常、CUDA OOM、NaN 或 Inf。
2. 两次运行均生成完整的 `model_100` checkpoint。
3. 除运行名称和保存目录外，两次训练配置一致。
4. 100 步训练 loss 逐步完全一致。
5. 剔除运行耗时字段 `_time` 后，所有结构化训练指标逐项一致。
6. 最终验证 loss 完全一致。
7. 最终 `model.safetensors` 逐字节一致。
8. 优化器与学习率调度器的内部状态逐项一致。

## 6. 实验结果

### 6.1 运行完成情况

两次运行均完成 100 个更新步骤，并输出：

```text
Script finished successfully
Rank 0 finished successfully
```

最终状态如下：

| 指标 | Run 1 | Run 2 | 是否一致 |
|---|---:|---:|---|
| `global_step` | 100 | 100 | 是 |
| `update_step` | 100 | 100 | 是 |
| `tokens_seen` | 9,981,647 | 9,981,647 | 是 |
| `tokens_seen_before` | 9,882,456 | 9,882,456 | 是 |
| `batch_idx` | 1599 | 1599 | 是 |
| 检测统计 `tn` | 100 | 100 | 是 |
| NaN/Inf 指标数 | 0 | 0 | 是 |

### 6.2 Checkpoint 完整性

两次运行均生成：

```text
model_100/
├── config.json
├── model.safetensors
├── optimizer.pt
└── training_state.json
```

模型文件大小均为 116,156,896 bytes，优化器文件大小均为 232,362,379 bytes。

### 6.3 训练轨迹一致性

每次运行的 `metrics.jsonl` 均包含 201 条记录。剔除仅反映墙钟时间的 `_time` 字段后：

```text
不一致记录数：0
```

100 个训练步骤均包含 loss，比较结果为：

| 指标 | 结果 |
|---|---:|
| 比较步数 | 100 |
| 最大逐步 loss 差异 | 0.0 |
| Step 1 loss | 10.4921875 |
| Step 100 loss | 7.50390625 |
| Run 1 最终 eval loss | 7.46795129776001 |
| Run 2 最终 eval loss | 7.46795129776001 |
| 最终 eval loss 差异 | 0.0 |

除 loss 外，梯度范数、权重范数、学习率、检测统计及 `rt` 等结构化指标也逐项一致。

### 6.4 模型参数一致性

两份最终模型的 SHA-256 完全相同：

```text
85918ad06adc171b714354ecb9d28a43c5555763a06b7f3fde27fa4c3675a7bd
```

因此两份 `model.safetensors` 不仅数值一致，而且文件内容逐字节一致，可以判定最终模型参数完全相同。

### 6.5 优化器与调度器状态

两个 `optimizer.pt` 的文件 SHA-256 不同，但加载后逐项比较得到：

| 内容 | 是否一致 |
|---|---|
| Optimizer state | 是 |
| Scheduler state | 是 |
| `update_step` | 是 |
| `global_step` | 是 |
| `dtype` | 是 |

文件哈希不同是因为 checkpoint 内嵌的 `name` 和 `save_dir` 不同：

```text
llama_60m_stability_run_1 / checkpoints/stability/run_1
llama_60m_stability_run_2 / checkpoints/stability/run_2
```

这不表示优化器数值状态不同，也不影响稳定性结论。

### 6.6 运行时间

| 运行 | 总时间 |
|---|---:|
| Run 1 | 109.7085 s |
| Run 2 | 107.0576 s |

两次总时间的相对差异约为 2.45%。该差异属于墙钟时间波动，不影响训练数值；所有训练指标和最终权重仍完全一致。

## 7. 结论

本次环境稳定性冒烟测试 **通过**。

在当前 RTX 4090 单 GPU、PyTorch 2.9.0、Transformers 4.56.2、bfloat16、固定 seed 351344 的环境中，两次独立的 100 步 LLaMA 60M 基线训练满足以下条件：

- 均正常完成并保存最终 checkpoint；
- 逐步训练 loss 和所有结构化数值指标完全一致；
- 最终验证 loss 完全一致；
- 最终模型文件逐字节一致；
- 优化器与调度器内部状态一致；
- 未出现 NaN、Inf、OOM 或其他运行时异常。

因此，当前环境在本实验覆盖的 100 步范围内具有可重复、确定的无故障训练行为，可作为后续 NVBit 故障注入实验、故障检测实验以及检测后重算实验的基线环境。

## 8. 结论边界与后续建议

本测试证明的是当前软硬件组合在固定配置、固定数据顺序和 100 步训练范围内的重复性，不等价于证明：

- 不同 GPU、CUDA、驱动或 PyTorch 版本之间逐位一致；
- 长达数千或数万步训练仍始终逐位一致；
- 开启 NVBit、分布式 NCCL 或多 GPU 后仍保持相同行为。

建议下一步执行 NVBit 冒烟测试，并保留本次 `run_1/model_100` 与 `run_2/model_100` 作为无故障对照证据。后续不同实验应继续使用独立的 `save_dir` 和日志文件，避免覆盖本次基线结果。
