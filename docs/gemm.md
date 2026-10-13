# CUTLASS Qwen 基础训练

使用官方 CUTLASS GEMM 接入 Qwen3-0.6B 全参数继续预训练，先完成 BF16 基线。模型、数据、优化器和检查点沿用[公共训练任务](training.md)。本阶段不加入 ABFT；FP8、FP4 留到下一阶段接入成熟框架的训练方案。

## 实现

`src/cutlass_gemm/` 将 Qwen 中所有 `nn.Linear` 接入 CUTLASS，包括 attention 投影、MLP 和 `lm_head`。前向、输入梯度（dgrad）和权重梯度（wgrad）调用官方 GEMM；其余模型运算保持原实现。

训练沿用 BF16 autocast 和 FP32 主权重。GEMM 使用 BF16 操作数、FP32 累加。线性层复用原模型参数，参数共享与优化器接入由原训练流程管理。

CUTLASS 提供 GEMM 实现。项目薄包装负责 PyTorch Tensor 参数、CUDA stream、线性层替换和 autograd 衔接。版本保持 CUTLASS 4.2.1、PyTorch 2.9.0 cu128、CUDA 12.8.1，配置见 `environment/gemm.json`。

直接调用官方 [`cutlass::gemm::device::Gemm`](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cutlass/gemm/device/gemm.h)，保留默认 tile、mainloop 和 epilogue。后续添加 ABFT 时，先保留这条无保护路径作对照，每次只加入一个机制。

## 构建和检查

在项目 NVIDIA Docker 容器内执行：

```bash
python -m cutlass_gemm.backend
bash scripts/gemm-accept.sh
```

第一个命令编译扩展；第二个命令用于 GPU 检查。当前尚未在 RTX 5090 上运行本基线训练，硬件验证状态见[验证说明](gemm-validation.md)。

## 训练

在同一 NVIDIA 容器中准备公共数据并启动训练：

```bash
bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
python -m cutlass_gemm.train \
  --config configs/training/qwen3_06b.yaml \
  --output-dir results/training/qwen3_cutlass_bf16 --offline
```

恢复训练时增加 `--resume latest`，使用原来的输出目录。训练日志和检查点沿用公共训练格式；不同实验使用独立输出目录。

实现入口：`backend.py` 负责加载扩展，`gemm.cu` 和 `bindings.cpp` 负责 GEMM 与 Tensor 绑定，`linear.py` 负责前向和反向适配，`train.py` 通过现有 `model_transform` 接口启动训练。
