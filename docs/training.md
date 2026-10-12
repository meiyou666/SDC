# 公共训练任务

采用自回归语言模型的全参数继续预训练。模型根据前面的 token 预测下一个 token，计算交叉熵，执行反向传播和 AdamW 参数更新。前向计算、输入梯度和权重梯度均来自实际模型训练。

## 任务与配置

| 配置 | 模型与文本 | 用途 |
|---|---|---|
| `configs/training/qwen3_06b.yaml` | Qwen3-0.6B；WikiText-103 的训练集与独立验证集 | 公共训练任务；NPU，BF16 混合精度 |
| `configs/training/smollm2_cpu_smoke.yaml` | SmolLM2-135M；WikiText-2 的少量真实文本 | 当前容器中的短程训练检查；CPU，FP32 |

两套配置使用同一训练入口。模型和语料固定到 Hugging Face commit。模型采用预训练权重，全参数更新；不使用 LoRA。BF16/FP16 配置使用混合精度计算，主权重以 FP32 加载。默认使用 eager attention，关闭 TF32。

当前开发镜像安装 CPU PyTorch。NPU 训练配置已准备好；服务器到位后，由维护人统一加入匹配的 CANN、PyTorch、`torch_npu` 和设备映射。

## 在容器内运行

更新仓库后，先在 VS Code 执行 **Dev Containers: Rebuild Container**。以下命令在容器的仓库根目录执行。

先用小模型跑通完整流程：

```bash
bash scripts/train.sh prepare --config configs/training/smollm2_cpu_smoke.yaml
bash scripts/train.sh train --config configs/training/smollm2_cpu_smoke.yaml --offline
```

`prepare` 下载语料、分词器和模型，并保存已分词的数据。下载中断后重新执行同一命令即可复用已有文件。此后 `train --offline` 从本地读取，不在训练过程中下载。

公共任务可先准备数据与模型：

```bash
bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
```

昇腾环境配置完成后，单卡与多卡分别运行：

```bash
bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline

uv run --locked torchrun --standalone --nproc_per_node=8 \
  --module training train --config configs/training/qwen3_06b.yaml --offline
```

中断后恢复同一次训练：

```bash
bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline --resume latest
```

也可将 `latest` 换成具体的 `checkpoint-步数` 路径。续训保留原配置、数据及进程数，恢复模型、优化器、学习率调度器、随机状态和训练位置。新实验使用新的 `output_dir`。

## 公共训练设置

- 训练集与验证集分别分词。每条文本后添加 EOS，再连续拼接为固定长度序列；只丢弃各数据集最后不足一个序列的 token。EOS 参与损失计算。
- `max_steps` 是优化器更新次数；每次更新包含 `gradient_accumulation_steps` 个微批次。有效批量为：每卡批量 × 累积次数 × 进程数。
- 使用固定随机种子和训练采样顺序。`full_determinism` 控制 PyTorch 的确定性执行。
- 训练记录 loss、梯度范数、学习率；验证记录 loss 和 perplexity。日志中的吞吐与训练耗时来自整个训练循环，GEMM 设备耗时由后续算子评测单独记录。
- 模型、批量、序列长度、精度、attention 实现与梯度检查点设置由公共配置管理。接入保护算法时保持这些设置一致。

配置也支持自己的文本数据：将 `data.source` 设为 `jsonl`，填写 `train_file` 和 `validation_file`。每行为 `{"text": "实际训练文本"}`。修改数据或分词设置后，使用新的 `prepared_dir`。

## 文件位置

```text
src/training/                 训练、数据准备与配置读取
configs/training/             公共训练配置
data/training/                固定的 token 数据与 manifest.json
results/training/<任务名>/
  config.resolved.json         实际采用的完整配置
  run.json                    模型精度、软件版本、数据标识与进程数
  metrics.jsonl               按 step 记录的训练和验证指标
  result.json                 最终训练与验证结果
  checkpoint-<step>/          模型、优化器、调度器、随机状态
```

Hugging Face 下载缓存位于容器的持久化 `~/.cache/huggingface`。模型、数据和训练输出不提交到 Git。检查点按 `save_total_limit` 保留，默认不再额外复制一份最终模型。

运行相关测试：

```bash
uv run --locked pytest -q tests/training
```

资料：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) · [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
