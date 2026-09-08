# MODIFICATIONS

本文档记录本复现仓库相对论文原始代码的主要修改。修改目标不是改变论文中的训练算法、故障注入目标或检测/重算机制，而是让代码能够在本地离线环境、CUDA 13.0/NVBit 1.7.6 和有限显存条件下完成实验。

## 修改总览

| 序号 | 修改 | 主要目的 | 对实验口径的影响 |
|---:|---|---|---|
| 1 | 离线数据、离线 tokenizer 与本地指标日志 | 云服务器无法访问外网，不能依赖 Hugging Face Hub 或 wandb | 数据来源和日志落盘方式改变，训练主逻辑不变 |
| 2 | NVBit 从 1.7.4 迁移到 1.7.6 | 适配 CUDA 13.0 环境 | NVBit 版本和路径改变，故障注入工具使用的核心 API 不变 |
| 3 | NVBit 插桩控制路径加速 | 避免未触发故障注入时仍对每个 CUDA kernel launch 调用昂贵控制接口 | 不改变 device-side 位翻转语义，只降低控制层开销 |
| 4 | logits/loss 显存优化 | 在显存受限条件下使用论文要求的 micro batch 规模运行 | 训练 loss 等价计算，但训练时不再保留完整 logits 张量 |
| 5 | 本地 C4 子集重复使用 | 本地数据量不足以支撑 VII-B 10,000 step | VII-B 使用约 2.56 个本地数据 pass，应在报告中说明 |

## 1. 离线运行支持

原始代码直接从 Hugging Face 加载 `allenai/c4` 和 `t5-base` tokenizer，并使用 wandb 记录指标。本地云服务器无法稳定连接外网，因此训练入口改为完全离线运行。

主要改动如下：

- `torchrun_main.py` 和 `torchrun_main_no_detection.py` 新增 `--tokenizer_path`、`--train_data_path`、`--val_data_path` 参数，默认分别指向 `./tokenizer/t5-base`、`./data/c4_subset_2M.json.gz` 和 `./data/c4_validation.jsonl`。
- tokenizer 使用 `AutoTokenizer.from_pretrained(..., local_files_only=True)`，避免训练过程中意外访问 Hugging Face Hub。
- `training/dataloader.py` 新增 `LocalJsonlDataset`，支持读取本地 JSONL 或 gzip 压缩 JSONL 数据，并按 rank/world size 分片。
- 训练和验证数据由本地文件提供，不再调用 `datasets.load_dataset("allenai/c4", ...)` 下载数据。
- `scripts/configs/common.sh` 设置 `TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`，从脚本层面约束离线行为。
- 新增 `training/metrics_logger.py`，用本地 `metrics.jsonl`、`summary.json` 和 `run_config.json` 替代 wandb 远端记录。

这部分修改只改变数据和指标 I/O，不改变 forward/backward、优化器、检测器或故障注入判定逻辑。

## 2. NVBit 1.7.4 到 1.7.6 迁移

论文原始工程使用 NVBit 1.7.4。本地环境使用 CUDA 13.0，因此将 NVBit 目录和脚本路径调整为 `nvbit/1.7.6_nvbit_release/`。

主要改动如下：

- `LD_PRELOAD`、实验脚本和构建说明均从 `1.7.4_nvbit_release` 更新到 `1.7.6_nvbit_release`。
- `nvbit/README_NVBit_1.7.6.md` 记录 1.7.6 的适配说明。
- `Makefile` 保持 fault injection tool 的构建方式，默认架构仍按本实验 GPU 设置，可通过 `ARCH=...` 覆盖。
- fault injection tool 使用的关键 NVBit API 仍是 `nvbit_get_related_functions`、`nvbit_get_instrs`、`nvbit_insert_call`、`nvbit_add_call_arg_*`、`nvbit_enable_instrumented` 等稳定接口。

迁移后，故障注入的基本语义仍是：在匹配的 SASS 指令前后插入 `insert_fault`，对指定寄存器执行 bitmask XOR。

## 3. NVBit 插桩控制路径加速

本地实验发现，fault injection run 即使尚未触发实际注入，也会明显慢化。同一 seed、尚未到 step 500 的前 32 步中：

| 运行 | 平均耗时 |
|---|---:|
| Baseline | 1.53 s/step |
| FI，尚未触发 | 22.37 s/step |
| 慢化 | 14.6x |

这 32 步的 992 个指标值与 baseline 完全一致，说明慢化不是模型数值异常造成的，而是 NVBit 控制层开销。核对原始 1.7.4 工具后确认，该问题来自原始 callback 控制流，不是离线数据/tokenizer 修改或 NVBit 1.7.6 迁移时引入的新错误：

```cpp
bool global_on = g_on.load(std::memory_order_relaxed);
bool enable = false;
if (global_on) {
    instrument_function_if_needed(ctx, func);
    ...
}

nvbit_enable_instrumented(ctx, func, enable);
```

当 Python 侧尚未发送 `SIGUSR1` 时，`g_on == false`，`enable` 保持默认值 `false`，但 `nvbit_enable_instrumented(ctx, func, false)` 仍会在每次 CUDA kernel launch callback 中执行。LLM 训练每个 update 包含大量 CUDA kernel，因此该路径会产生显著慢化。14.6x 是本地环境中的实测倍率，具体数值可能受 CUDA、NVBit、PyTorch、GPU 和驱动状态影响。

当前复现版本对 NVBit callback 做了三项优化：

- 当 `g_on == false` 且没有已启用 function 时快速返回，不再对每个 kernel 调用 `nvbit_enable_instrumented(..., false)`。
- 新增 `enabled_functions` 和 `enabled_function_count`，只在 instrumentation 状态变化时调用 NVBit API。
- 在调用 `nvbit_get_related_functions()` 前检查 `already_instrumented`，避免重复扫描已经分析过的 function。
- 增加 `instrumentation_mutex`，避免多个 CUDA launch callback 线程并发修改 function 缓存和 enable 状态。

该优化没有修改 device-side 的 BEFORE/AFTER 两次 `insert_fault` 插桩逻辑，因此故障瞬态语义保持不变。

## 4. logits/loss 显存优化

原始 causal LM forward 在带 `labels` 训练时会构造完整 logits，形状约为：

```text
batch_size x sequence_length x vocab_size
```

在 60M 模型、`max_length=256`、较大 micro batch 和 bfloat16 条件下，完整 logits 会占用大量显存，限制本地复现实验按论文要求的 micro batch 配置运行。

本复现版本在 `training/modeling_llama.py` 中新增 `_chunked_lm_loss`：

- 对 hidden states 做 shift 后，按 `loss_chunk_size` 分块；
- 每个 chunk 单独通过 `lm_head` 计算局部 logits；
- 对每个 chunk 计算 `F.cross_entropy(..., reduction="sum")`；
- 最后用总 loss 除以有效 token 数，得到与完整 logits 口径一致的 mean loss。

当 forward 收到 `labels` 时，训练只返回 `loss`，`logits` 保持为 `None`；当没有 `labels` 时，仍按原方式计算并返回完整 logits，保留推理/评估中需要 logits 的接口行为。

该修改的目的只是降低训练时的峰值显存，不改变 loss 的数学定义。

## 5. 本地数据重复使用

本地 C4 子集 `c4_subset_2M.json.gz` 约 2M 条序列。在 batch 512 的设置下，1 个 epoch 约为 3,906 个 update steps。VII-B 需要运行 10,000 steps，因此默认单 pass 数据会在 step 3906 左右正常耗尽。

日志中的表现是：训练脚本正常输出 `Script finished successfully` 并保存 `model_3906` final checkpoint；随后 campaign 脚本检查不到预期的 `model_10000`，按保护逻辑报错。这不是训练崩溃，而是本地数据量不足。

为支持 VII-B 完整运行，本复现版本做了如下修改：

- `OfflinePreprocessedDataset` 新增 `repeat` 参数，`__len__` 返回单 pass batch 数乘以 repeat。
- `__getitem__` 对 batch index 按单 pass batch 数取模，使每个 pass 的 batch 顺序完全一致。
- `torchrun_main.py` 和 `torchrun_main_no_detection.py` 新增 `--train_data_repeat` 参数，默认值为 1，不影响已有短实验。
- `scripts/configs/vi_vii.sh` 中 VII-B 设置 `VI_VII_B_DATA_REPEAT=3`。1 个本地 epoch 约 3,906 steps，repeat 3 可提供约 11,718 steps，足以覆盖 10,000-step run。
- `run_vii_b_scale_60m.sh` 对 baseline、FI 和 FI+recompute 三个条件都传入相同的 `--train_data_repeat 3`，避免条件间数据口径漂移。

这部分是明确的实验口径偏差：论文使用更完整的 C4 数据流，本地 VII-B 使用约 2.56 个本地数据 pass。报告中应说明该差异，尤其是在讨论泛化 loss 或长程训练趋势时。

## 6. 总结

总体而言，本复现版本主要做的是工程适配：让原始训练与故障注入代码能够在离线数据、本地 tokenizer、CUDA 13.0/NVBit 1.7.6 和有限显存环境中运行。除本地数据来源、日志方式、NVBit 控制路径性能优化、loss 显存优化以及 VII-B 数据重复使用外，故障注入语义、异常检测统计量和 recompute 流程保持与复现实验设计一致。
