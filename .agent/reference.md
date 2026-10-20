# Agent 技术参考

开始任务时先读 [AGENTS.md](../AGENTS.md) 和[项目研究说明](../docs/project.md)，再按下表读取相关实现。成员的操作步骤集中在[环境](../environment/README.md)、[训练](../docs/training.md)和 [GEMM](../docs/gemm.md) 三篇教程。

## 按任务阅读

| 任务 | 配置与实现入口 |
| --- | --- |
| 环境、依赖与容器 | [baseline.json](../environment/baseline.json)、[pyproject.toml](../pyproject.toml)、[uv.lock](../uv.lock)、[Dockerfile](../environment/docker/Dockerfile)、[Compose](../compose.yaml)、[Dev Container](../.devcontainer/) |
| 云端初始化 | [setup-cloud.sh](../environment/docker/setup-cloud.sh)、[configure-cloud.py](../environment/docker/configure-cloud.py) |
| 平台容器虚拟环境 | [setup-platform.sh](../environment/setup-platform.sh)、[activate.sh](../environment/activate.sh)、[configure_platform.py](../environment/configure_platform.py) |
| 训练与数据 | [训练配置](../configs/training/qwen3_06b.yaml)、[src/training](../src/training/)、[训练测试](../tests/training/) |
| CUTLASS 与线性层 | [gemm.json](../environment/gemm.json)、[src/cutlass_gemm](../src/cutlass_gemm/)、[算子测试](../tests/cutlass_gemm/) |
| ABFT 与故障注入 | [调试配置](../configs/training/qwen3_abft_debug.yaml)、[src/abft](../src/abft/)、[使用与扩展方法](../docs/gemm.md#启用-abft) |
| CI 与代码审查 | [工作流](../.github/workflows/)、[环境检查](../scripts/ci/)、[检查测试](../tests/ci/)、[审查规范](../.github/CODE_REVIEW.md) |

## 环境与依赖维护

版本以配置文件为准：系统、Python 和训练平台版本在 `environment/baseline.json`，Python 依赖在 `pyproject.toml` 与 `uv.lock`，CUTLASS 源码和编译目标在 `environment/gemm.json`。不在教程中另行维护版本表。

Dockerfile 的 `development`、`nvidia` 和 `ascend` 均从 `common` 阶段构建，分别安装对应的 uv extra。公共 Python 依赖共用锁文件，硬件 SDK 与 PyTorch 构建按平台选择。`tools` 服务使用 `tooling` 阶段，负责依赖解析与锁文件维护；`nvidia-build` 使用 NVIDIA 镜像，只提供编译入口，不挂载 GPU。

驱动与固件位于宿主机，CUDA 或 CANN 位于训练镜像。昇腾通过 `SDC_ASCEND_SOC` 选择算子包；SDK 镜像及 digest 在 `baseline.json` 中登记。昇腾使用的 PyTorch `+cpu` 构建由 `torch_npu` 提供 NPU 计算。

容器中的项目环境为 `/opt/venv`，工作目录为 `/workspace/SDC`。`SDC_RUNTIME` 指定 uv extra；训练和检查脚本自动使用该值。模型下载缓存位于 `~/.cache/huggingface`。Codex 配置、个人工具、SSH 配置和下载缓存通过卷保存；卷的命名由 Compose 或 Dev Container 配置决定。

Codex 首次配置来自 [codex-default.toml](../environment/docker/codex-default.toml)。[setup-codex.sh](../environment/docker/setup-codex.sh) 读取 `.env`，已有个人配置不被覆盖。登录失败不阻止容器启动。AI 工具独立安装在个人目录，不改动项目的 `/opt/venv`。

**依赖调整由维护人执行。** 在宿主机仓库目录调用 `tools` 容器：

```bash
docker compose run --build --rm tools add --no-sync "包名==版本"
docker compose run --build --rm tools lock --check
```

开发工具使用 `add --dev --no-sync`。直接编辑 `pyproject.toml` 后，执行 `docker compose run --build --rm tools lock`。同时提交声明和锁文件，再重建相应容器。Linux 或 WSL 可在 `tools` 前添加 `--user "$(id -u):$(id -g)"`，使生成文件归当前用户所有。

调整 Python 版本时同步检查 `.python-version`、Dockerfile 和 `baseline.json`。发行包名与导入名不同时，更新 `baseline.json` 的 `import_map`。

租用的平台容器直接执行 `environment/setup-platform.sh nvidia`（或 `ascend`），在仓库 `.venv` 安装同一份锁定依赖。`.sdc/` 保存固定 uv、托管 Python、CUTLASS 源码、安装配置快照和 VS Code 工作区，均不提交 Git。安装脚本复用预装 CUDA/CANN；NVIDIA 检查 `nvcc` 和开发头文件，不依赖 `version.json`。系统允许 Ubuntu 22.04/24.04，实际版本保存在环境记录中。`environment/activate.sh` 为训练与检查脚本选择统一环境；快照、Python、依赖或 SDK 不匹配时停止实验。

完整服务器与本地继续使用共享 Dockerfile、Compose 和 Dev Container，构建与运行使用 host 网络。平台容器通过 Remote-SSH 打开 `.sdc/SDC-platform.code-workspace`，该工作区指定 `.venv` 和 SDK 环境，调试仍使用仓库的训练配置。平台安装与工作区生成测试在项目 Docker 中执行；远端命令交由用户运行。

GitHub Actions 检查 Compose 配置、脚本语法、锁文件和环境策略，再执行测试与 Ruff。工作流缓存 Docker 的 `tooling` 和 `development` 构建层，同一 PR 更新时取消旧任务。GitHub runner 使用 Ubuntu 官方 APT 源，并设置 `SDC_INSTALL_AI_TOOLS=false`；本地构建默认使用清华 APT 源并安装 AI 工具。

维护人也可运行 **Update uv lock** 工作流，下载并提交生成的锁文件。改变依赖源时同步更新 `pyproject.toml`。

## 训练模块与接入接口

| 文件 | 职责 |
| --- | --- |
| [config.py](../src/training/config.py) | 解析配置、检查字段、生成数据与运行签名 |
| [data.py](../src/training/data.py) | 读取语料、分词、拼接序列、保存数据及文件哈希 |
| [metrics.py](../src/training/metrics.py) | 保存指标、处理非有限值 |
| [runner.py](../src/training/runner.py) | 解析设备、加载模型、构建 Trainer、保存结果与恢复训练 |
| [__main__.py](../src/training/__main__.py) | `prepare` 和 `train` 命令入口 |

公共配置的 `device: accelerator` 按 `SDC_RUNTIME` 解析为 `cuda` 或 `npu`。NVIDIA 使用 `gemm.backend: cutlass`；NPU 使用 `torch` 兼容路径。模型以 FP32 加载，混合精度由训练配置控制。数据准备先写临时目录，再重命名为目标目录；训练启动时检查数据签名及文件哈希。

`train()` 根据 `gemm` 配置安装对应线性层，统一命令为 `bash scripts/train.sh train --config YAML`。实验参数、输出目录、离线与续训设置均读取 YAML。

`train()` 同时提供模型变换接口。下面是接口示意，其中 `install` 由具体方法实现：

```python
train(
    config,
    model_transform=install,
    transform_id="protected-linear:v1",
    transform_params={"tile_size": 64},
    callbacks=[...],
)
```

- 模型加载后、Trainer 创建前执行 `install(model, tile_size=64)`，返回用于训练的模型。
- 提供 `model_transform` 时必须提供 `transform_id`；`transform_params` 为可保存为 JSON 的参数字典。
- `callbacks` 接收 `TrainerCallback` 列表，用于读取训练状态和记录事件。
- Python、NumPy 和 Torch 的随机种子在模型加载与变换之前设置。需要随检查点保存的校验向量或在线状态，注册为参数或持久化 buffer。

`run.json` 保存配置签名、数据摘要、锁文件哈希、提交版本、设备、模型变换标识和参数。续训时比对配置、数据、锁文件和模型变换信息，并检查检查点中的模型、优化器、调度器和随机状态。缺少模型变换记录的旧运行使用新目录重新训练。

非有限指标写为 `null`，原值记录在 `nonfinite_metrics` 中。loss 非有限时停止训练并保存失败原因；其他指标非有限时继续运行。输出文件用途见[训练教程](../docs/training.md)，不在此重复维护。

## CUTLASS 调用关系

统一 Trainer 根据 `gemm` 配置将 Qwen 的全部 `nn.Linear` 替换为 `CutlassLinear`，包括 attention 投影、MLP 与 `lm_head`。前向 `X @ W.T`、输入梯度 `dY @ W` 和权重梯度 `dY.T @ X` 均调用所选精度的 CUTLASS GEMM。替换保留原 Parameter 对象、共享权重与模块别名，其他算子沿用原模型。

| 文件 | 职责 |
| --- | --- |
| [backend.py](../src/cutlass_gemm/backend.py) | 读取编译配置、生成构建标识、编译加载扩展、检查设备 |
| [bindings.cpp](../src/cutlass_gemm/bindings.cpp) | 注册 PyTorch 算子及 CUDA 实现 |
| [gemm.cu](../src/cutlass_gemm/gemm.cu) | FP32 与 BF16 官方 GEMM |
| [fp8.cu](../src/cutlass_gemm/fp8.cu) | 官方 SM120 E4M3 GEMM，FP32 列主序输出 |
| [nvfp4.cu](../src/cutlass_gemm/nvfp4.cu) | 官方 example 79a GEMM，BF16 输出及 SF 布局辅助函数 |
| [quantization.py](../src/cutlass_gemm/quantization.py) | Torch 缩放与官方 CUTLASS 类型转换连接 |
| [linear.py](../src/cutlass_gemm/linear.py) | 统一线性层与 autograd 适配 |

四种 GEMM 均使用 FP32 累加。FP32 外围精度为 FP32，其余外围精度为 BF16；主权重保持 FP32，参数梯度返回原参数 dtype。BF16 使用官方 SM80 单测的 `128×128×64` 分块和默认三阶段配置，共享内存为 96 KiB。FP8 保留官方单测的 FP32 列主序输出，前向再转为 BF16，反向返回对应输入和参数的 dtype。NVFP4 使用 example 79a 的 BF16 输出配置。

`gemm.precision/scaling` 对应 `fp32/none`、`bf16/none`、`fp8/tensorwise`、`nvfp4/block16`。`global_scale` 为 `amax` 或有限正数乘子；`abft` 选择 `none`、`classic` 或已注册的方法。具体字段见 [config.py](../src/training/config.py) 与 [ABFT 配置](../src/abft/config.py)。

### ABFT 调用关系

线性层通过 `abft.engine.execute()` 准备一次输入，再交给 `execute_prepared()` 完成保护。关闭保护与注入时直接调用原有 GEMM。`classic` 使用全 1 与递增行权重，两路 FP32 列投影从 epilogue 中乘 alpha 后、输出转换前的值生成。`reference_mode: input_fused` 在主循环内复用已经加载的 A、B 及 NVFP4 block16 scale，生成预测和 magnitude；`separate` 保留按 K 分块解码及独立 FP32 GEMM 的参考路径。

| 文件 | 职责 |
| --- | --- |
| [config.py](../src/abft/config.py) | 方法、阈值、恢复与注入配置 |
| [operands.py](../src/abft/operands.py) | 准备操作数、分块解码、码值注入与原生调用 |
| [methods.py](../src/abft/methods.py) | 方法注册、`prepare/propagate/verify/recover` 接口和 `classic` 示例 |
| [engine.py](../src/abft/engine.py) | 调用编排、一次干净重算与复检、设备报告和显式序列化 |
| [abft_legacy.cu](../src/cutlass_gemm/abft_legacy.cu)、[abft_legacy_kernel.h](../src/cutlass_gemm/abft_legacy_kernel.h) | FP32/BF16 epilogue visitor 与分块投影 |
| [abft_fp8.cu](../src/cutlass_gemm/abft_fp8.cu)、[abft_nvfp4.cu](../src/cutlass_gemm/abft_nvfp4.cu)、[abft_sm120.cuh](../src/cutlass_gemm/abft_sm120.cuh) | FP8/NVFP4 epilogue 注入与两路 reduction |

原生路径保留官方 mainloop，通过 epilogue 收集校验。`accumulator` 注入点位于 MMA 之后；输入注入修改准备好的码值。恢复复用干净码值和尺度，不重复量化。每次受保护 GEMM 调用均执行方法工厂，当前未接入插件历史状态的检查点保存。配置、报告和方法扩展步骤集中在 [GEMM 教程](../docs/gemm.md#启用-abft)。

输入融合的原生入口为 `abft_reference_mm`（FP32/BF16）、`abft_fp8_reference_mm`、`abft_nvfp4_reference_mm`，返回 `(output, observed_partials, hit, predicted_partials, magnitude_partials)`。局部数组均为 `[2,ceil(M/tileM),padded_N]`，保持全局行权重，最终沿 M tile 合并。融合模式只接受计算故障，重算复检继续使用首次参考。`parameters` 提供方法专用 YAML 参数，`validate_config/prepare_fused/bind_reference` 用于扩展方法；默认 `classic` 不接受额外参数。

`official_adapter.py` 与 `legacy_adapter.py` 对固定哈希的官方源码生成独立类型，不修改 `/opt/cutlass`。两份 SM120 consumer 函数在剔除新增校验调用后与官方一致，见 `test_mainloop_preservation.py`。legacy iterator 先处理 K 余数块，再处理完整块，参考路径的 K 掩码必须遵循该顺序。

| 输入融合路径 | 原流水线级数 | 新增共享校验缓冲 | 完整 kernel 共享存储 |
| --- | ---: | ---: | ---: |
| FP32 | 2 | 128 B | 21,568 B |
| BF16 | 3 | 1,024 B | 99,328 B |
| FP8 | 6 | 1,024 B | 101,376 B |
| NVFP4 | 4 | 2,048 B | 90,112 B |

以上是编译期布局与单文件编译检查结果；设备容量还由原生入口查询，数值结果及吞吐需在目标 GPU 上测量。现有测试检查四精度输出、预测、幅值、尾块、异构 scale、恢复与线性层前反向。

### 量化准备

量化准备组合现有 Torch 的 amax、乘除等运算与 `nvidia-cutlass-dsl==4.2.1` 中未改动的官方转换内核。主机端缓存 `cute.compile(cute.testing._convert, ...)`，按设备、类型和布局复用编译结果；所有视图保留动态布局，每次传入当前张量。官方转换负责 E4M3、E2M1 类型转换和 FP4 打包，完整量化由项目适配代码组织。项目不新增 CUDA 或 Triton 量化内核。

FP8 对每个操作数做 tensorwise 缩放。`global_scale: amax` 时，FP8 的全局乘子为 `448/amax`，NVFP4 为 `(6*448)/amax`，全零矩阵取 1；显式正数配置直接指定该乘子。NVFP4 使用官方 `gett.hpp` 的通用参考公式，输入采用 E2M1 与 E4M3 block16 尺度：`sf=E4M3(block_amax*(st*(1/6)))`，再使用该尺度的 FP32 倒数与全局乘子准备 E2M1 输入。example 79b 作为相关对照，其输出尺度使用 UE8M0。GEMM 的 alpha 抵消两输入的全局乘子。SF 的物理位置由官方 CuTe layout 在 CPU 辅助函数中生成，再使用现有 Torch scatter 写入尺度张量。

### 官方源码对应

所有链接固定到 CUTLASS 提交 `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。

| 项目用途 | 官方源码 |
| --- | --- |
| FP32、BF16 GEMM 接口 | [gemm.h](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cutlass/gemm/device/gemm.h) |
| FP8 kernel 配置 | [SM120 f8_f8_f32 单测](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/test/unit/gemm/device/sm120_tensorop_gemm/sm120_gemm_f8_f8_f32_tensor_op.cu) |
| NVFP4 kernel 配置与 BF16 输出 | [example 79a](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu) |
| 通用量化公式与相关对照（79b 输出尺度为 UE8M0） | [gett.hpp](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/tools/util/include/cutlass/util/reference/host/gett.hpp)、[example 79b](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu) |
| E4M3、E2M1 转换与 FP4 打包 | [cute.testing.convert](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/python/CuTeDSL/cutlass/cute/testing.py)、[cute_tensor_like](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/python/CuTeDSL/cutlass/torch.py) |
| SF 物理布局 | [sm100_blockscaled_layout.hpp](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cutlass/detail/sm100_blockscaled_layout.hpp) |
| FP32/BF16 ABFT visitor 接口与 kernel 连接 | [example 35 gemm_softmax](https://github.com/NVIDIA/cutlass/tree/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/examples/35_gemm_softmax)、[epilogue_with_visitor.h](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cutlass/epilogue/threadblock/epilogue_with_visitor.h) |
| FP8/NVFP4 ABFT 两路列投影使用的 `Sm90RowReduction` | [sm90_visitor_store_tma_warpspecialized.hpp](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp)、[callbacks](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp) |

`_extension_key()` 按 C++/CUDA 源码与编译设置缓存扩展；`build_key()` 还包含 Python 适配源码，用于训练的模型变换标识，GEMM 配置另存为变换参数。修改实现或实验参数后，使用新输出目录。GPU 检查入口为 [gemm-accept.sh](../scripts/gemm-accept.sh)，操作步骤见 [GEMM 教程](../docs/gemm.md)。

## 已有验证记录

以下为统一入口改造前的 BF16 基线记录。最初记录日期为 2026-10-13，对应提交 `1c79d21`：

- 在项目 Docker 中完成 SM120a 扩展编译、链接和加载，构建标识为 `5d8c186a22f49b11`。
- 测试通过 175 项，另有 49 项子测试通过；Ruff 通过。
- Python 桥接测试使用 `torch.mm`，覆盖前向、反向、参数共享、参数更新与检查点恢复。
- 当时未执行实际 GPU 检查及完整 Qwen3-0.6B 训练。

2026-10-14 在 RTX 5090 上调整 BF16 分块后，构建标识为 `e0bdef356dd8c139`。CUTLASS 检查共 21 项通过，其中包含 11 项实际 GPU 用例；完整检查共 186 项及 49 项子测试通过，Ruff 通过。

同日完成 Qwen3-0.6B 的 CUTLASS BF16 继续预训练，共 1000 个参数更新步。训练中断后从 `checkpoint-100` 恢复，重跑的 110—140 步日志中，loss、梯度范数和学习率与中断前一致。最终验证 loss 为 `2.6606974601745605`，perplexity 为 `14.306263668996365`。配置、日志和恢复记录保存在 `results/validation/5090/`，完整检查点当时保存在旧服务器的 `results/training/qwen3_cutlass_bf16/`。

统一入口改造后，在 RTX 5090 的项目 Docker 容器中完成四种 CUTLASS 精度的 Qwen3-0.6B / WikiText-103 训练。每种精度运行 20 个参数更新步，batch size 为 1，序列长度为 512，梯度累积步数为 1。

| GEMM 精度 | 验证 loss |
| --- | --- |
| FP32 | `2.934720754623413` |
| BF16 | `2.9449222087860107` |
| FP8 | `2.9549427032470703` |
| NVFP4 | `3.1861612796783447` |

本轮完整检查通过 237 项测试及 55 项子测试，Ruff 通过，其中低精度 GPU 用例通过 7 项。未修改的官方 FP32 basic example、FP8 SM120 单测和 NVFP4 example 79a 分别运行通过。结果汇总保存在 `results/validation/5090/cutlass-official/summary.json`。

这四次短程运行用于验证统一训练链路。此前 BF16 运行了 1000 步，梯度累积步数为 16，与本轮训练配置不同，loss 不直接比较。

当时在服务器保存镜像 `sdc-nvidia:py311-cutlass-official`，默认标签 `sdc-nvidia:py311` 指向同一镜像。从新镜像启动的容器使用默认 `sdc` 用户与入口脚本，通过加速器检查及全部 7 项低精度 GPU 测试。镜像标识、启动命令和日志保存在 `results/validation/5090/cutlass-official/image.json` 与 `fresh-container-gpu.log`。

2026-10-15 用户已释放旧服务器，旧 SSH 连接不可用。上述历史验证日志保留在本地 `results/`；旧服务器镜像与 1000 步训练完整检查点的路径仅作为历史记录。

后续硬件运行记录设备、提交版本、执行命令和结果，保存在实验输出目录。教程只维护操作步骤。

2026-10-15 在本地 `meiyou666/feature-abft-gemm` 分支加入 ABFT epilogue 投影与训练连接。项目 NVIDIA 容器完成全部无保护与融合源文件的 SM120a 编译、链接和扩展加载；扩展标识为 `6162930f1741c598`。`bash scripts/check.sh` 通过 261 项测试及 55 项子测试，Ruff 通过。105 项 GPU 用例跳过，其中 87 项为新增 ABFT 原生与端到端用例。当前容器没有目标 GPU，尚未执行融合内核的 SM120 数值测试或受保护 Qwen 训练。日志保存在 `results/validation/abft-local/`。

同日提交 `b2bb9d0` 分离 `execute_prepared()` 与量化准备，并加入 YAML 驱动的配对 GEMM 测量。保护计时覆盖参考生成至设备判定，输入量化与布局准备不计入；阶段计时默认关闭。训练及接口回归通过 136 项，测量和框架检查通过 59 项。

随后接入 FP8 `input_fused` 原型：`abft_fp8_mainloop.cuh` 仅在固定官方 `mma()` 的 stage wait 后消费共享输入，`official_adapter.py` 从固定哈希的官方 kernel 生成独立类，只扩展 MMA 调用点。原始 CUTLASS 安装目录和无保护实例不改。参考部分使用全局行权重，保留原检查粒度；预测与 magnitude 作为局部和返回。寄存器保存列累加器后，新增共享缓冲为 1024 字节，流水线仍为 6 级，完整动态共享内存为 101376 字节。全扩展 `b4e2a13b8c17b118` 编译、链接与加载成功；完整检查为 301 项通过、112 项 GPU 跳过、55 项子测试通过，Ruff 通过。日志在 `results/validation/abft-input-fused/`；无实机时延或数值结论。

2026-10-15 四精度输入融合已接通，示例测量及 ABFT 训练配置使用 `input_fused`。完整扩展 `aeac1951a13c2293` 编译、链接、加载和三个 CUDA 接口注册检查通过，构建标识为 `e51127644c279838`。完整本地检查为 316 项通过、171 项 GPU 跳过、55 项子测试通过；Ruff 通过。独立检查修正了 legacy 余数块先行时的 K 掩码，以及第二路权重索引的 int32 加法溢出。最终源码对照的 6 项检查通过。

日志和 `summary.json` 保存在 `results/validation/abft-four-input/`。用户确认暂时没有目标 GPU，先推进本地实现。因此尚未执行四精度的 GPU 数值、恢复、前反向或性能测试。后续取得设备后先运行 `bash scripts/gemm-accept.sh`，再执行配对 benchmark，按实际瓶颈进行单项优化；不要将编译结果或 CPU 桥接测试视作实机结果。故障注入当前仍在 epilogue 的 alpha 后，尚未扩展为 MMA 指令内部注入。

随后在本机 RTX 4070（SM89）补充 legacy 原生正确性诊断。独立容器 `sdc-abft-sm89-check` 使用同一项目锁文件，环境与 CUDA 检查通过；仅将相同 FP32/BF16 源文件编译为 SM89，正式 `gemm.json` 和生产设备检查保持 SM120。隔离进程复用原测试，只选 FP32/BF16，并显式使用真实 SM89 能力检查。源码基于 `167b55b`。

实际运行 96 项通过：输入融合 44 项、输出校验 48 项、完整执行与线性层前反向 4 项；FP8/NVFP4 的 43 项被排除。Compute Sanitizer 的 memcheck 对 BF16 K80、FP32 K33 两项通过，0 errors；racecheck 对 BF16 K448 四种布局通过，0 hazards、0 errors、0 warnings。原始日志、逐项结果、源码哈希及诊断脚本在 `results/validation/sm89/`。这些结果补充 FP32/BF16 的真实执行证据，未测时延，也不替代 SM120 或 FP8/NVFP4 验证。
