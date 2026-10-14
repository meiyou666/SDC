# CUTLASS GEMM 教程

本教程使用官方 CUTLASS GEMM 薄包装接入 Qwen3-0.6B，运行 BF16 基线。FP8、FP4 的 Transformer Engine 训练入口见[训练教程](training.md#fp8-与-fp4-训练)。后续在 CUTLASS 上逐项加入低精度 GEMM 和 ABFT。

先按[环境教程](../environment/README.md)进入 NVIDIA 容器。以下命令均在容器内的仓库根目录执行。当前构建目标为 SM120a，设备检查要求 SM120 GPU（如 RTX 5090）。

## 编译 CUTLASS 扩展

```bash
python -m cutlass_gemm.backend
```

命令使用镜像内固定的 CUTLASS 源码编译、加载扩展，并打印构建标识和配置。构建设置见 [environment/gemm.json](../environment/gemm.json)，软件版本见 [environment/baseline.json](../environment/baseline.json)。若提示 CUTLASS 版本或镜像配置不匹配，按环境教程重建 NVIDIA 镜像。

## 运行检查

```bash
bash scripts/gemm-accept.sh
```

脚本先检查加速器环境、目标 GPU 和扩展，再运行 `tests/cutlass_gemm/`。实际 GPU 测试覆盖 GEMM 数值结果、矩阵转置布局、CUDA stream、非法输入、线性层梯度，以及微型 Qwen 的参数更新。

测试摘要显示全部通过后，继续下一步。已有运行记录集中在 [Agent 技术参考](../.agent/reference.md#已有验证记录)。

## 修改算子并再次检查

按改动对象选择文件：

| 文件 | 修改内容 |
| --- | --- |
| `src/cutlass_gemm/gemm.cu` | 官方 GEMM 调用、布局分派和输入检查 |
| `src/cutlass_gemm/bindings.cpp` | PyTorch 算子绑定 |
| `src/cutlass_gemm/linear.py` | 线性层替换、前向和反向适配 |
| `src/cutlass_gemm/backend.py` | 扩展构建与加载 |
| `src/cutlass_gemm/train.py` | CUTLASS 训练入口 |

当前包装调用官方 `cutlass::gemm::device::Gemm`。BF16 使用官方测试中的 `128×128×64` 分块，适配 RTX 5090 的共享内存容量；warp、mainloop 和 epilogue 使用默认配置。先保留这条无保护基线作为对照，每次只加入一个机制，并在 `tests/cutlass_gemm/` 补充对应行为与失败情形的检查。

修改后重新执行：

```bash
python -m cutlass_gemm.backend
bash scripts/gemm-accept.sh
```

源码变化会产生新的扩展构建标识；命令会按新标识加载扩展。修改 `environment/gemm.json` 后先重建镜像。确认检查完成后，再接入训练。

## 接入训练

先按[训练教程](training.md#准备数据)准备公共数据，再启动独立的 CUTLASS 实验：

```bash
python -m cutlass_gemm.train \
  --config configs/training/qwen3_06b.yaml \
  --output-dir results/training/qwen3_cutlass_bf16 --offline
```

该入口替换 Qwen 中的 `nn.Linear`，包括 attention 投影、MLP 和 `lm_head`。前向、输入梯度和权重梯度调用 CUTLASS；公共配置使用 BF16 操作数、FP32 累加和 FP32 主权重，其余训练流程沿用公共任务。

日志、检查点、后台运行和中断续训见[训练教程](training.md)。查看结果时将 `run_dir` 设为上方 `--output-dir`。续训使用相同命令并添加 `--resume latest`。

CUTLASS 源码或构建设置变化后，训练的模型变换标识也会变化。此时设置新的 `--output-dir` 开始实验。实际运行后记录设备、提交版本、命令和结果，并查看 loss、梯度及检查点恢复情况。
