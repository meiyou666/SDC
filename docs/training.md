# Qwen3 训练教程

本教程使用 Qwen3-0.6B，在 WikiText-103 文本上进行单卡全参数继续预训练。默认采用 BF16 混合精度、FP32 主权重；NVIDIA 另提供 FP8、NVFP4 线性层训练入口。三者共用数据、优化器和训练流程。

先按[环境教程](../environment/README.md)进入 NVIDIA 或昇腾训练容器。除“后台运行”中的 Docker 调度命令外，以下命令均在容器内的仓库根目录执行。使用 CUTLASS 算子时，先完成[GEMM 教程](gemm.md)中的编译与检查。

## 准备数据

```bash
python environment/verify_environment.py --accelerator
bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
```

`prepare` 单进程执行一次，下载固定版本的模型、分词器和 WikiText-103，并将分词后的训练集、验证集保存到 `data/training/qwen3_06b_wikitext103/`。每条文本后添加 EOS，再拼接为定长序列。模型和分词器保存在容器配置的 Hugging Face 缓存中。

下载中断后可重试相同命令。若提示 `prepared_dir` 不完整，改用新的目录后重新准备；已有完整数据会复用。训练读取数据时会检查清单与文件哈希。修改数据或分词设置后，使用新的 `data.prepared_dir`。

个人数据可在实验 YAML 中设置 `data.source: jsonl`、`train_file` 和 `validation_file`，每行使用 `{"text": "训练文本"}`。只准备数据和分词器可给 `prepare` 添加 `--data-only`；离线训练前仍需下载模型权重。

## 启动训练

```bash
bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline
```

`--offline` 从本地缓存加载模型和分词器。`device: accelerator` 根据容器选择 CUDA 或 NPU。序列长度、批量、更新步数和保存间隔见[训练配置](../configs/training/qwen3_06b.yaml)。训练按 `max_steps` 指定的参数更新步数结束；数据遍历完后继续下一轮。

需要调整训练参数时，复制配置到 `experiments/`，再把命令中的 `--config` 改为该文件。新实验设置独立的 `training.output_dir`；公共训练入口通过 YAML 设置输出目录，没有 `--output-dir` 参数。不要同时向同一输出目录启动多个训练进程。

## FP8 与 FP4 训练

在更新后的 NVIDIA 镜像中运行。首次构建会编译固定的 Transformer Engine 官方源码，后续构建复用缓存。先执行上面的数据准备，再检查低精度训练连接：

```bash
python -m pytest -q tests/low_precision/
```

GPU 用例分别检查 FP8、FP4 的真实前向和反向 GEMM、梯度检查点、参数更新及保存恢复。没有 GPU 时会跳过这些用例。

选择精度并使用独立输出目录：

```bash
python -m low_precision.train --precision fp8 \
  --config configs/training/qwen3_06b.yaml \
  --output-dir results/training/qwen3_fp8 --offline

python -m low_precision.train --precision fp4 \
  --config configs/training/qwen3_06b.yaml \
  --output-dir results/training/qwen3_fp4 --offline
```

这两条命令分别启动训练，单卡依次运行。`--precision` 选择 Transformer Engine 的线性层训练配方：

| 选择 | 前向 GEMM | 反向 GEMM | 实现 |
| --- | --- | --- | --- |
| `fp8` | E4M3 操作数 | E5M2 梯度与 E4M3 操作数 | Transformer Engine FP8 |
| `fp4` | E2M1 操作数 | E2M1 操作数 | Transformer Engine NVFP4 |

`training.precision` 保持 `bf16`，用于外围算子的 autocast。FP32 主权重、FP32 累加和 BF16 激活继续保留；低精度线性层内部执行对应的 FP8 或 FP4 GEMM。FP4 使用 NVFP4 的分块缩放、随机舍入和 Hadamard 变换。全部线性层参与替换，包括 `lm_head`；注意力内部矩阵乘法、softmax 和归一化沿用原模型。

`run.json` 的 `low_precision` 字段记录配方、精度路径和组件版本。这条入口使用 Transformer Engine 的 cuBLASLt 后端；需要修改 CUTLASS 内核时使用 [GEMM 教程](gemm.md)中的独立入口。

续训使用相同精度、配置和输出目录，追加 `--resume latest`。更改精度或配方后使用新目录开始训练。

## 查看结果

默认输出目录按设备区分：

- NVIDIA：`results/training/qwen3_06b_wikitext103/cuda/`
- 昇腾：`results/training/qwen3_06b_wikitext103/npu/`

| 文件 | 内容 |
| --- | --- |
| `config.resolved.json` | 实际使用的配置 |
| `run.json` | 数据与配置标识、依赖锁文件哈希、提交版本、设备、模型变换与续训来源 |
| `metrics.jsonl` | 逐步 loss、梯度范数、学习率和验证指标 |
| `result.json` | 最终状态、训练与验证指标；失败时记录原因和训练步数 |
| `checkpoint-<step>/` | 模型、优化器、调度器、随机状态和训练位置 |

以 NVIDIA 默认目录为例；昇腾或自定义实验替换 `run_dir`：

```bash
run_dir=results/training/qwen3_06b_wikitext103/cuda
tail -f "$run_dir/metrics.jsonl"
```

`Ctrl+C` 结束日志查看。训练结束后查看最终结果：

```bash
cat "$run_dir/result.json"
```

完成时 `status` 为 `completed`，并记录 `validation_loss` 和 `validation_perplexity`。日志的吞吐统计来自训练循环；GEMM 设备耗时需单独测量。loss 出现 NaN 或 Inf 会报错停止；其他非有限指标写为 `null`，原值类型保存在 `nonfinite_metrics`。模型、数据和完整输出不提交 Git。

## 后台运行

长时间训练使用独立后台容器，关闭 VS Code 后仍可继续。以下命令在 **Linux / WSL 宿主机的 Bash、仓库根目录**执行；云端使用 Remote-SSH 的服务器终端。项目训练仍在容器中运行。先按环境教程配置设备，以 NVIDIA 为例：

```bash
docker compose --profile nvidia run --build --rm --user "$(id -u):$(id -g)" nvidia \
  bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
docker compose --profile nvidia run -d --name sdc-qwen3-nvidia \
  --user "$(id -u):$(id -g)" nvidia \
  bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline
docker logs -f sdc-qwen3-nvidia
```

准备与训练使用相同 Compose 服务，共用模型缓存。`Ctrl+C` 只结束日志查看，训练继续。CUTLASS 训练将服务名后的训练命令替换为 [GEMM 教程](gemm.md#接入训练)中的命令。

昇腾先在同一个**宿主机 Bash 终端**设置芯片和设备组，再把上方 profile、服务名改为 `ascend`，容器名改为 `sdc-qwen3-ascend`：

```bash
export SDC_ASCEND_SOC=910b  # A2 用 910b，A3 用 a3，950 用 950
export SDC_ASCEND_GID=$(stat -c '%g' /dev/davinci0)
```

这些参数用于 Compose；云端 Dev Container 中的设备设置不会自动传给后台容器。

查看后台任务状态：

```bash
docker ps -a --filter name=sdc-qwen3-nvidia
```

同名容器重跑前，先确认旧任务已停止，再移除旧容器；训练输出保留在仓库的 `results/` 中：

```bash
docker rm sdc-qwen3-nvidia
```

## 中断后续训

使用原配置、原入口和原输出目录，从最近保存的检查点继续：

```bash
bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline --resume latest
```

也可将 `latest` 替换为具体的 `checkpoint-<step>` 路径。后台续训在对应的 Docker 启动命令末尾添加 `--resume latest`。

续训要求配置、数据、`uv.lock` 哈希及模型变换标识和参数一致。CUTLASS 实验通过原 CUTLASS 入口续训。恢复位置是最近保存的检查点；尚未保存的步骤需要重跑。没有完整检查点时，设置新输出目录重新开始。已达到 `max_steps` 的检查点不能继续，修改训练配置后也应开启新实验。
