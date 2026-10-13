# CUTLASS 基础训练验证

当前基线为官方 CUTLASS BF16 GEMM 接入 Qwen3-0.6B。

2026-10-13 已在项目 Docker 中完成：

- SM120a 扩展编译、链接和加载通过，构建标识为 `5d8c186a22f49b11`。
- 全部测试 175 项通过，另有 49 项子测试通过；Ruff 通过。
- Python 训练桥接测试使用 `torch.mm`，覆盖前向、反向、参数共享、参数更新与检查点恢复。

11 项实际 GPU 检查及完整 Qwen3-0.6B 训练尚待 RTX 5090 执行。

获得 RTX 5090 后，在项目 NVIDIA Docker 容器中运行：

```bash
python -m cutlass_gemm.backend
bash scripts/gemm-accept.sh
```

随后按[训练说明](gemm.md#训练)执行 Qwen 训练，检查 loss、梯度、参数更新和检查点恢复。记录实际运行的设备、提交版本、命令及结果。
