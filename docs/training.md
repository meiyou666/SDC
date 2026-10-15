# Qwen3 训练教程

本教程使用 Qwen3-0.6B，在 WikiText-103 文本上进行单卡全参数继续预训练。NVIDIA 通过统一入口选择 FP32、BF16、FP8 或 NVFP4 CUTLASS 线性层，主权重保持 FP32。各精度共用数据、优化器和训练流程。

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
bash scripts/train.sh train --config configs/training/qwen3_06b.yaml
```

在 YAML 中设置 `training.offline: true` 后，从本地缓存加载模型和分词器。`device: accelerator` 根据容器选择 CUDA 或 NPU。序列长度、批量、更新步数和保存间隔见[训练配置](../configs/training/qwen3_06b.yaml)。训练按 `max_steps` 指定的参数更新步数结束；数据遍历完后继续下一轮。

需要调整训练参数时，复制配置到 `experiments/`，再把命令中的 `--config` 改为该文件。新实验设置独立的 `training.output_dir`。不要同时向同一输出目录启动多个训练进程。

| YAML 设置 | 用途 |
| --- | --- |
| `training.optimizer`、`training.lr_scheduler_type` | 优化器与学习率调度 |
| `training.save_strategy`、`training.save_steps` | 检查点保存方式与间隔；短程实验可用 `save_strategy: "no"` |
| `training.eval_strategy`、`training.eval_steps` | 训练中验证方式与间隔 |
| `training.tf32` | TF32 设置 |
| `model.gradient_checkpointing`、`model.gradient_checkpointing_use_reentrant` | 梯度检查点与重计算方式 |
| `training.progress_bar` | 显示训练和验证进度条，默认开启 |
| `training.logging_steps` | 每多少次参数更新输出一次 loss、梯度范数和学习率 |

关闭检查点保存的实验无法中断续训。完整字段与默认值见[配置实现](../src/training/config.py)。

## 选择 GEMM 精度

所有训练参数写在同一份 YAML 中。NVIDIA 研究实验使用 `gemm.backend: cutlass`，通过以下组合选择精度：

| `gemm.precision` | `gemm.scaling` | `training.precision` | 线性层前向的 GEMM 输入与输出 |
| --- | --- | --- | --- |
| `fp32` | `none` | `fp32` | FP32 输入与输出 |
| `bf16` | `none` | `bf16` | BF16 输入与输出 |
| `fp8` | `tensorwise` | `bf16` | E4M3 输入，官方 FP32 输出经适配转 BF16 |
| `nvfp4` | `block16` | `bf16` | E2M1 输入、每 16 个值一个 E4M3 尺度，BF16 输出 |

四条路径的前向、输入梯度和权重梯度均调用 CUTLASS，采用 FP32 累加。全部线性层参与替换，包括 `lm_head`；注意力内部矩阵乘法、softmax 和归一化沿用原模型。

例如，复制公共配置到 `experiments/my_training.yaml`，修改以下字段，保留模型、数据和其他训练设置：

```yaml
gemm:
  backend: cutlass
  precision: nvfp4
  scaling: block16
  global_scale: amax
  abft: none

training:
  output_dir: results/training/qwen3_cutlass_nvfp4
  device: accelerator
  precision: bf16
  offline: true
  resume: null
```

```bash
bash scripts/train.sh train --config experiments/my_training.yaml
```

`gemm.global_scale` 选择 `amax` 动态缩放或有限正数乘子，用于 FP8、NVFP4 量化。`gemm.abft: none` 关闭保护；启用 `classic`、配置故障和恢复见 [GEMM 教程](gemm.md#启用-abft)。更换精度、缩放或其他实验参数时，设置新的输出目录。

`gemm.backend: torch` 保留原生参考和 NPU 兼容路径，支持 FP32、BF16 和 FP16；CPU 仅用于 FP32 单元测试。昇腾训练将公共配置的 backend 改为 `torch`。

## 在 VS Code 中调试

先通过 Remote-SSH 连接服务器，再进入项目的 NVIDIA Dev Container。VS Code 终端中的仓库路径应为 `/workspace/SDC`，Python 为 `/opt/venv/bin/python`。在扩展面板确认 **Python Debugger** 已安装到当前容器。

1. 打开 [qwen3_debug.yaml](../configs/training/qwen3_debug.yaml)。它复用公共模型与数据，运行两步 BF16 CUTLASS 训练，每步记录日志，不保存检查点。梯度累积设为 1，关闭梯度检查点，便于依次查看前向、反向与参数更新。
2. 模型和数据已准备好时直接使用；首次准备仍执行前文的 `prepare` 命令。调试配置默认离线加载。
3. 在下表的位置点击行号左侧设置断点。按 **F5**，选择“训练：按 YAML 调试（容器内）”，接受默认配置路径。
4. 暂停后用 **F10** 执行当前行，**F11** 进入函数，**Shift+F11** 返回调用者，**F5** 继续到下一个断点。左侧查看变量与调用堆栈，调试控制台可查看张量的 `shape`、`dtype` 和 `device`。

| 断点位置 | 观察内容 |
| --- | --- |
| `runner.py`：`model = model_transform(...)` | 将模型中的 Linear 替换为 CUTLASS 线性层 |
| `runner.py`：`result = trainer.train(...)` | 开始训练；此时可查看 `trainer.model` 和实际训练配置 |
| `linear.py`：`out = backend.gemm(...)` | 前向的输入 `a`、权重 `w` 与所选精度 |
| `linear.py`：`dx = product(...)`、`dw = product(...)` | 输入梯度和权重梯度的两次 GEMM |
| `metrics.py`：`on_step_end()` 内的第一行 | 已完成一次参数更新，查看 `state.global_step` |

先开训练入口与前向断点，再逐步打开反向断点。同一个 Linear 实现会被多层调用，断点会反复命中；看完一次可暂时取消断点。首次执行还可能编译 CUTLASS 扩展。

FP8 或 NVFP4 调试按前文修改 YAML，并在 `backend.py` 的 `alpha = ...` 行查看 `qa`、`qb` 的码值与尺度。`torch.ops.sdc_cutlass.*` 调入 C++/CUDA；Python 调试器用于跟踪其调用前后的张量。查看 CUDA 内核内部需要 CUDA 调试器。

两步训练结束后仍执行最终验证，此时会再次前向而不反向。重新开始调试前，把 `training.output_dir` 改为新目录，例如 `qwen3_debug_02/{device}`；保留已有输出。调试配置不保存检查点，不使用 `resume: latest`。

调试校验时选择 [qwen3_abft_debug.yaml](../configs/training/qwen3_abft_debug.yaml)，在 `abft/engine.py` 的 `execute()` 与 `abft/methods.py` 的 `prepare()`、`verify()` 设置断点，依次查看参考校验、原生投影和报告。配置中的 `diagnostics: true` 保留完整残差与阈值，详细字段见 [GEMM 教程](gemm.md#启用-abft)。

同一份配置也可从终端运行：

```bash
bash scripts/train.sh train --config configs/training/qwen3_debug.yaml
```

## 查看结果

终端会依次显示数据读取、模型加载、GEMM 配置、训练与最终验证。首次调用 CUTLASS 时，扩展加载器会输出编译或加载信息。

训练进度条显示已完成的参数更新次数、耗时与预计剩余时间。指标按 `training.logging_steps` 输出；调试配置设为 `1`，每步都能看到 loss、梯度范数和学习率。验证也有独立进度条，结束后显示验证 loss、perplexity 和结果目录。需要逐行日志时可设置 `progress_bar: false`，指标仍会输出并保存。

进度条中的一步是一次参数更新，包含 `gradient_accumulation_steps` 个批次。调试器停在断点时，程序与进度条一起暂停，按 F5 后继续。

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
  bash scripts/train.sh train --config configs/training/qwen3_06b.yaml
docker logs -f sdc-qwen3-nvidia
```

准备与训练使用相同 Compose 服务，共用模型缓存。`Ctrl+C` 只结束日志查看，训练继续。自定义实验将 `--config` 改为对应 YAML。

昇腾先在 YAML 中设置 `gemm.backend: torch`，再在同一个**宿主机 Bash 终端**设置芯片和设备组，再把上方 profile、服务名改为 `ascend`，容器名改为 `sdc-qwen3-ascend`：

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

在原 YAML 中设置 `training.resume: latest`，保留原输出目录，从最近保存的检查点继续：

```bash
bash scripts/train.sh train --config configs/training/qwen3_06b.yaml
```

也可将配置中的 `latest` 替换为具体的 `checkpoint-<step>` 路径。后台续训使用同一配置。

开启 `gemm.fault` 注入时不支持续训，因为实际 GEMM 调用计数尚未写入检查点。此类实验设置新输出目录并使用 `resume: null`。

续训要求配置、数据、`uv.lock` 哈希及模型变换标识和参数一致。恢复位置是最近保存的检查点；尚未保存的步骤需要重跑。没有完整检查点时，设置新输出目录重新开始。已达到 `max_steps` 的检查点不能继续，修改训练配置后也应开启新实验。
