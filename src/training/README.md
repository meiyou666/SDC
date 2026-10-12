# 训练模块

Qwen3-0.6B 在 WikiText-103 上的全参数继续预训练，单卡运行。训练精度为 BF16 混合精度、FP32 主权重；FP8、FP4 量化在评测侧进行。

## 文件职责

| 文件 | 职责 |
|---|---|
| `config.py` | 配置结构、校验、`data_signature` 与 `run_signature` |
| `data.py` | 下载语料、分词、拼接定长序列、写入带哈希清单的快照 |
| `metrics.py` | 指标落盘、JSON 非有限值处理、loss 非有限值即失败 |
| `runner.py` | 设备解析、Trainer 参数、续训校验、训练编排 |
| `__main__.py` | `python -m training prepare|train` 入口 |

配置只从 YAML 读取。`device: accelerator` 由容器的 `SDC_RUNTIME` 解析为 `cuda` 或 `npu`。

## 运行

在 NVIDIA 或昇腾训练容器中：

```bash
bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
bash scripts/train.sh train   --config configs/training/qwen3_06b.yaml --offline
bash scripts/train.sh train   --config configs/training/qwen3_06b.yaml --offline --resume latest
```

`prepare` 单进程执行一次。启动容器、后台训练见[训练说明](../../docs/training.md)。

## 输出

写在 `results/training/<任务>/<设备>/`：

| 文件 | 内容 |
|---|---|
| `config.resolved.json` | 设备解析后的实际配置 |
| `run.json` | 软件版本、`uv.lock` 哈希、git commit、设备、数据摘要、训练预算 |
| `metrics.jsonl` | 每条 Trainer 日志一行 JSON |
| `result.json` | 最终训练与验证指标；失败时记录 `status: failed` 和原因 |
| `checkpoint-<step>/` | 模型、优化器、调度器、随机状态 |

非有限值写成 `null`，原值记在 `nonfinite_metrics`。loss 非有限值报错停止，其他指标非有限时继续训练。

## 采集操作数

`train(config, callbacks=[...])` 接受 `TrainerCallback`，用于在训练过程中导出指定层的 GEMM 操作数。

## 无加速卡时的检查

```bash
bash scripts/check.sh                 # 全部测试与 ruff
bash scripts/check.sh tests/training  # 只跑训练模块测试
```

`tests/training/test_training.py` 在 CPU 上用微型 Qwen3 跑前向、反向、参数更新和续训比对。自己的小规模配置放在 `experiments/*.yaml`。
