# 统一 GEMM 工作台

公共入口为 `gemm.gemm`。NVIDIA 后端使用 CUTLASS 4.2.1 与其附带的 C++ CuTe，目标是 RTX 5090（SM120a）。版本保存在 `environment/gemm.json`，沿用 Torch 2.9.0、CUDA 12.8.1 与根目录 `uv.lock`。

## 构建和检查

宿主机仓库目录：

```bash
docker compose build nvidia
docker compose run --rm nvidia bash
```

进入容器后：

```bash
python -m backends.nvidia.build
bash scripts/gemm-accept.sh
```

没有 GPU 的开发机可用同一镜像编译，并在开发容器运行单元测试：

```bash
docker compose run --build --rm nvidia-build python -m backends.nvidia.build
docker compose run --rm dev bash scripts/check.sh
```

扩展存入共享的 `torch_extensions` 缓存。编译配置、Torch 或原生源码改变后自动重编译。`gemm-accept.sh` 要求实际 SM120 GPU，并执行全部 GEMM 测试。

## 接口与精度

```python
from gemm import gemm, Fault

result = gemm(a, b, profile="mxfp8", algorithm="classic", backend="nvidia")
result = gemm(a, b, profile="mxfp8", fault=Fault("accumulator", 3, 7, 23),
              recovery="native")
```

逻辑计算为 `C[M,N] = A[M,K] @ B[K,N]`，返回 FP32 输出、检测状态、残差、阈值和实际翻转前后的编码。支持非连续输入和尾块，内部将尺寸补齐至 128 的倍数。输出保留逻辑尺寸。

| profile | 乘法输入与尺度 | 主计算 |
| --- | --- | --- |
| `fp32` | FP32 | SIMT FP32，关闭 TF32 |
| `bf16` | BF16 | Tensor Core，FP32 累加 |
| `mxfp8` | E4M3，每 32 个值共享 E8M0 scale | 原生分块缩放 Tensor Core，FP32 累加 |
| `nvfp4` | E2M1，每 16 个值共享 E4M3 scale，另有 FP32 全局 scale | 原生分块缩放 Tensor Core，FP32 累加 |

NVFP4 权重采用 16×16 共享 scale；输出梯度操作数采用随机舍入，权重与激活采用最近偶数舍入；wgrad 在归约轴应用成对 H16 变换。每个阶段从原始操作数重新生成所需布局。转置时重新量化，不能把已有低精度码值直接转置后复用原 scale。

保护对象是 Q+S 确定的逻辑乘法。输入量化近似在离线诊断中单独统计。生产校验直接解读 Q+S，不生成完整反量化矩阵。经典 ABFT 的参考生成、传播与输出校验使用 FP32；`reference` 后端使用独立 FP64 公式供单元测试使用。

## 接入训练

继续使用公共 Qwen 数据准备和 Trainer。以下命令在 NVIDIA 容器中执行：

```bash
bash scripts/train.sh prepare --config configs/training/qwen3_06b.yaml
python -m gemm.train --gemm-config configs/gemm/qwen3_mxfp8.yaml \
  --output-dir results/training/qwen3_mxfp8 --offline
```

NVFP4 改用 `configs/gemm/qwen3_nvfp4.yaml`。配置中的 `profile` 同时控制 forward、dgrad、wgrad。MLP 的 gate、up、down 三个投影接入自定义后端，主权重与优化器状态保持 FP32。线性层输出按训练 dtype 转换，其余模型运算沿用公共训练设置。

FP32、BF16 的配置也位于 `configs/gemm/`。`atol` 和 `rtol` 是显式实验参数，使用独立正常样本校准后写入对应配置。

`ProtectedLinear` 复用原 Parameter。默认检测后同源重算一次并复检，恢复失败时停止训练。`recovery: native` 使用校验残差定位修正；`algorithm: none` 使用同源无保护内核。续训加 `--resume latest`；配置和实现标识必须与检查点一致。

需要收集真实操作数时，在 GEMM 配置加入 `capture_dir: data/gemm/qwen3_mxfp8`。每个目标层各保存一组前向、dgrad、wgrad 的操作数和种子。采集涉及主机复制，采集运行与性能运行分开。

## 故障与评价

每次事件翻转一个元素的一位，位号从最低位 0 开始：

| site | 位置与时机 |
| --- | --- |
| `a`、`b` | 参考校验生成后，修改主计算实际读取的码值；FP4 按 nibble 定位 |
| `accumulator` | 主计算完成后、epilogue 全局缩放和写回之前，对真实 FP32 累加器异或 |

`Fault(..., enabled=False)` 是零掩码对照。累加器对照保留相同的带注入 epilogue。输入故障可能影响整行或整列输出；定位与单元素修复按实际受损输出个数统计。

NVFP4 wgrad 的注入位置指 H16 变换后的码值。事件记录同时保留原始形状和实际编码的归约长度。

```bash
python -m gemm --input data/gemm/qwen3_mxfp8/model.layers.0.mlp.gate_proj.forward.pt \
  --profile mxfp8 --output results/gemm/gate_mxfp8 --timing --diagnostics
```

输出 `events.jsonl` 和 `summary.json`。记录逐 bit 方向、变化范围、非有限输出、检出、定位、原生修复、同源重算和输出变化。误报以正常调用为分母；检出以输出受损事件为分母；定位与原生修复以单输出元素受损事件为分母。误修复指复检通过但输出未恢复；同时保留分子、分母及 Wilson 95% 区间。

计时包括输入转换和量化、主 GEMM、参考校验生成及传播、输出归约和判定。CUDA Graph 内使用设备事件，编译、数据下载、故障构造与 Python 调度不计入。报告完整保护时间、同源无保护时间、分阶段时间及原生库参照。原生库不支持当前格式时，该项记录错误原因，不替换成其他精度。恢复时间与训练 wall time 单独测量。

`--diagnostics` 另外计算 FP64 逻辑参考，将残差分解为主计算误差、输出归约误差、参考校验误差和比较舍入误差。这些离线计算不计入检测开销。

## 分工入口

| 目录 | 工作 |
| --- | --- |
| `src/gemm/` | 公共接口、训练适配、注入协议、统计与计时 |
| `src/algorithms/` | 各论文的编码、阈值与恢复；见[接入说明](../src/algorithms/README.md) |
| `src/backends/nvidia/` | CUTLASS/CuTe、量化、位操作、CUDA 校验 |
| `src/backends/ascend/` | 昇腾后端接口 |
| `tests/gemm/` | 表示、算法、训练、原生内核测试 |

昇腾沿用相同前端、线性层和评价契约，设备实现放入独立后端。当前 `backend="ascend"` 返回未实现提示，不会转到 CUDA 或 CPU。
