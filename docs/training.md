# Qwen3 公共训练任务

统一采用 Qwen3-0.6B，在真实 WikiText-103 文本上进行全参数继续预训练。任务包含 next-token loss、前向计算、反向传播、AdamW 更新和独立验证。模型、数据版本与训练参数固定在 `configs/training/qwen3_06b.yaml`。

## 两套训练环境

| 设置 | NVIDIA GPU | 昇腾 NPU |
|---|---|---|
| Compose 服务 | `nvidia` | `ascend` |
| Ubuntu、Python | 22.04、3.11.17 | 22.04、3.11.17 |
| PyTorch 主版本 | 2.7.1 | 2.7.1 |
| 硬件组件 | CUDA 12.8.1、cu128 构建 | CANN 9.1.0、torch_npu 2.7.1.post8 |
| Transformers、Datasets、Accelerate | 4.57.6、4.4.2、1.12.0 | 4.57.6、4.4.2、1.12.0 |
| 训练精度 | BF16 混合精度，FP32 主权重 | BF16 混合精度，FP32 主权重 |

两套镜像共用基础层、`pyproject.toml`、`uv.lock` 和训练代码。配置中的 `device: accelerator` 由容器选择 CUDA 或 NPU，训练参数不随平台改变。

## 启动容器

以下命令在服务器的仓库根目录执行。NVIDIA 宿主机配置驱动和 NVIDIA Container Toolkit；昇腾宿主机配置对应的驱动、固件及设备权限。

```bash
# NVIDIA GPU
docker compose --profile nvidia run --build --rm --user "$(id -u):$(id -g)" nvidia bash

# 昇腾 NPU
docker compose --profile ascend run --build --rm --user "$(id -u):$(id -g)" ascend bash
```

Docker 默认使用宿主机的 x86-64 或 ARM64 架构。昇腾芯片由 `SDC_ASCEND_SOC` 选择：A2 使用 `910b`，A3 使用 `a3`，950 使用 `950`；三者均使用 CANN 9.1.0。例如：

```bash
export SDC_ASCEND_SOC=a3
export SDC_ASCEND_GID=$(stat -c '%g' /dev/davinci0)
docker compose --profile ascend run --build --rm --user "$(id -u):$(id -g)" ascend bash
```

默认使用第 0 张卡。NVIDIA 多卡在启动前设置 `SDC_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。昇腾设备清单在 `compose.yaml` 的 `devices` 中维护，驱动目录只读挂载；多卡时扩展设备清单，并设置 `SDC_ASCEND_VISIBLE_DEVICES`。

使用 VS Code 时，按[云端开发说明](cloud.md)运行初始化脚本，再选择生成的 **SDC 云端（NVIDIA）** 或 **SDC 云端（Ascend）**。进入训练容器后，继续执行下方命令。

## 准备与训练

以下命令在对应训练容器内执行，两种平台完全相同：

```bash
python environment/verify_environment.py --accelerator
bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline
```

`prepare` 下载固定版本的模型、分词器与语料，并保存分词结果。下载中断后可执行同一命令继续。训练从本地缓存读取。

空的 `prepared_dir` 会重新准备；目录已有残留文件但缺少完整清单时，会提示更换目录或清理残留。首次读取校验 SHA256，后续在清单、文件大小和时间等信息未变时复用校验结果。需要重算全部哈希时，在 `prepare` 或 `train` 命令末尾添加 `--verify-data`。

中断后恢复：

```bash
bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline --resume latest
```

检查点包含模型、优化器、学习率调度器、随机状态及训练位置。新实验使用新的 `output_dir`。多卡容器配置好后，通过 `torchrun` 启动；进程数与开放的设备数一致：

```bash
uv run --locked --extra "$SDC_RUNTIME" torchrun --standalone --nproc_per_node=8 \
  --module training train --config configs/training/qwen3_06b.yaml --offline
```

## 后台训练

长时间训练可在独立后台容器中运行，关闭 VS Code 后继续执行。以下命令在**服务器宿主机的仓库目录**执行。先按“启动容器”设置平台与设备变量，下面以 NVIDIA 为例：

```bash
docker compose --profile nvidia run --build --rm --user "$(id -u):$(id -g)" nvidia \
  bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
docker compose --profile nvidia run -d --name sdc-qwen3-nvidia \
  --user "$(id -u):$(id -g)" nvidia \
  bash scripts/train.sh train --config configs/training/qwen3_06b.yaml --offline
docker logs -f sdc-qwen3-nvidia
```

准备与训练都通过 Compose 执行，以共用模型缓存。`Ctrl+C` 退出日志查看，训练继续。昇腾将 profile 和服务名改为 `ascend`，并使用独立容器名。继续已有训练时，在训练命令末尾添加 `--resume latest`。

## 公共设置与输出

- 模型与语料固定到 Hugging Face commit；全参数训练，使用 eager attention，关闭 TF32。
- 训练集和验证集分别分词，每条文本后添加 EOS。文本连续拼接为固定长度序列，EOS 参与损失计算。
- `max_steps` 表示参数更新次数。当前任务按固定步数训练，数据遍历完后继续下一轮。有效批量 = 每卡批量 × 梯度累积次数 × 进程数；序列数见 `run.json`，实际遍历进度见日志和 `result.json` 的 `epoch`。
- 日志记录 loss、梯度范数、学习率、验证 loss 和 perplexity。吞吐来自整个训练循环；GEMM 设备耗时由算子评测单独测量。
- loss 出现 NaN 或 Inf 时明确报错并停止，`result.json` 记录失败原因和训练步数。非有限指标写为 `null`，原值类型保存在 `nonfinite_metrics`。
- 两个平台共用数据、随机种子、精度和训练设置。结果分别保存到 `results/training/qwen3_06b_wikitext103/cuda` 与 `npu`。

```text
src/training/                 训练与数据准备
configs/training/qwen3_06b.yaml  公共任务配置
data/training/                固定的 token 数据
results/training/<任务>/<设备>/
  config.resolved.json         实际配置
  run.json                    软件版本、数据标识与进程数
  metrics.jsonl               逐步指标
  result.json                 最终训练与验证结果
  checkpoint-<step>/          续训检查点
```

个人文本可通过 `data.source: jsonl`、`train_file` 和 `validation_file` 配置，每行格式为 `{"text": "训练文本"}`。修改分词设置或数据后使用新的 `prepared_dir`。模型、数据与输出不提交 Git。

资料：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
