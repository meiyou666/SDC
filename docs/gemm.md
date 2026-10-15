# CUTLASS GEMM 教程

本教程使用官方 CUTLASS GEMM 接入 Qwen3-0.6B。FP32、BF16 使用官方兼容模板，FP8、NVFP4 使用官方 SM120 配置。四种精度共用线性层适配和训练入口，精度、缩放及 ABFT 设置放在 YAML 中。

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

脚本检查加速器环境、目标 GPU 和扩展，并运行算子测试。修改后检查数值结果、矩阵布局、CUDA stream、非法输入、线性层梯度及训练连接。已有运行记录见 [Agent 技术参考](../.agent/reference.md#已有验证记录)。

## 修改算子并再次检查

| 文件 | 修改内容 |
| --- | --- |
| `src/cutlass_gemm/gemm.cu` | FP32、BF16 官方 GEMM 调用与输入检查 |
| `src/cutlass_gemm/fp8.cu` | 官方 SM120 E4M3 GEMM 调用 |
| `src/cutlass_gemm/nvfp4.cu` | 官方 NVFP4 GEMM 调用与尺度布局 |
| `src/cutlass_gemm/quantization.py` | Torch 缩放计算与官方 CUTLASS 转换连接 |
| `src/cutlass_gemm/bindings.cpp` | PyTorch 算子绑定 |
| `src/cutlass_gemm/linear.py` | 统一线性层替换、前向和反向适配 |
| `src/cutlass_gemm/backend.py` | 扩展构建与加载 |
| `src/training/config.py` | 训练和 GEMM 参数校验 |

FP32、BF16 调用官方 `cutlass::gemm::device::Gemm`。BF16 使用官方测试中的 `128×128×64` 分块。FP8 保留官方单测的 FP32 列主序输出，前向再转为 BF16，反向返回对应输入和参数的 dtype；NVFP4 使用官方 example 79a 的 BF16 输出。各路径使用 FP32 累加，保留官方 mainloop 和 epilogue 配置。源码对应关系见[技术参考](../.agent/reference.md#官方源码对应)。

量化准备由现有 Torch 运算计算 amax、乘除和尺度，再调用 `cute.testing` 中未改动的官方转换内核完成类型转换及 FP4 打包。主机端使用 `cute.compile` 缓存转换函数，避免每次调用重复生成中间表示。NVFP4 尺度位置通过官方 CuTe layout 的 CPU 辅助函数生成，再由 Torch scatter 写入。

每次加入一个机制，并在 `tests/cutlass_gemm/` 补充对应行为检查。修改后重新执行：

```bash
python -m cutlass_gemm.backend
bash scripts/gemm-accept.sh
```

源码变化会产生新的扩展构建标识。修改 `environment/gemm.json` 后先重建镜像。

## 接入训练

先按[训练教程](training.md#准备数据)准备公共数据。复制公共 YAML 到 `experiments/`，在配置中设置 `gemm.backend: cutlass`、精度与缩放方式，并指定独立的 `training.output_dir`。随后运行：

```bash
bash scripts/train.sh train --config experiments/my_training.yaml
```

统一入口替换 Qwen 中的 `nn.Linear`，包括 attention 投影、MLP 和 `lm_head`。前向、输入梯度和权重梯度均调用所选精度的 CUTLASS GEMM，主权重保留 FP32。注意力内部矩阵乘法、softmax 和归一化沿用原模型。

日志、检查点、后台运行和续训见[训练教程](training.md)。续训在 YAML 中设置 `training.resume: latest`，再运行相同命令。CUTLASS 源码、构建设置或实验参数变化后，设置新的 `training.output_dir` 开始实验。
