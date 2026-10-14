# Agent 技术参考

开始任务时先读 [AGENTS.md](../AGENTS.md) 和[项目研究说明](../docs/project.md)，再按下表读取相关实现。成员的操作步骤集中在[环境](../environment/README.md)、[训练](../docs/training.md)和 [GEMM](../docs/gemm.md) 三篇教程。

## 按任务阅读

| 任务 | 配置与实现入口 |
| --- | --- |
| 环境、依赖与容器 | [baseline.json](../environment/baseline.json)、[pyproject.toml](../pyproject.toml)、[uv.lock](../uv.lock)、[Dockerfile](../environment/docker/Dockerfile)、[Compose](../compose.yaml)、[Dev Container](../.devcontainer/) |
| 云端初始化 | [setup-cloud.sh](../environment/docker/setup-cloud.sh)、[configure-cloud.py](../environment/docker/configure-cloud.py) |
| 训练与数据 | [训练配置](../configs/training/qwen3_06b.yaml)、[src/training](../src/training/)、[训练测试](../tests/training/) |
| CUTLASS 与线性层 | [gemm.json](../environment/gemm.json)、[src/cutlass_gemm](../src/cutlass_gemm/)、[算子测试](../tests/cutlass_gemm/) |
| FP8、FP4 训练 | [src/low_precision](../src/low_precision/)、[训练连接测试](../tests/low_precision/)、[训练教程](../docs/training.md#fp8-与-fp4-训练) |
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

公共配置的 `device: accelerator` 按 `SDC_RUNTIME` 解析为 `cuda` 或 `npu`。模型以 FP32 加载，混合精度由训练配置控制。数据准备先写临时目录，再重命名为目标目录；训练启动时检查数据签名及文件哈希。

`train()` 提供模型变换接口。下面是接口示意，其中 `install` 由具体方法实现：

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

入口 `cutlass_gemm.train` 复用公共训练流程，通过 `model_transform=install` 将 Qwen 的全部 `nn.Linear` 替换为 `CutlassLinear`，包括 attention 投影、MLP 与 `lm_head`。其他算子沿用原模型实现。这条 CUTLASS 基础训练使用 BF16 或 FP32；FP8、FP4 组件使用下面的独立入口。

| 文件 | 职责 |
| --- | --- |
| [backend.py](../src/cutlass_gemm/backend.py) | 读取编译配置、生成构建标识、编译加载扩展、检查设备 |
| [bindings.cpp](../src/cutlass_gemm/bindings.cpp) | 注册 PyTorch 算子及 CUDA 实现 |
| [gemm.cu](../src/cutlass_gemm/gemm.cu) | 检查输入、分配输出、调用官方 `cutlass::gemm::device::Gemm` |
| [linear.py](../src/cutlass_gemm/linear.py) | 替换线性层并连接 autograd |
| [train.py](../src/cutlass_gemm/train.py) | 解析训练参数，接入公共 `train()` |

GEMM 使用官方 mainloop 和 epilogue。BF16 路径采用官方 SM80 单测中的 `128×128×64` 分块和默认三阶段配置，共享内存为 96 KiB，满足 SM120 每个 CTA 的 99 KiB 上限。原默认分块需要 144 KiB。FP32 路径保留默认分块并使用 SIMT；两条路径均采用 FP32 累加。具体编译目标读取 `environment/gemm.json`。

线性层前向计算 `X @ W.T`，输入梯度计算 `dY @ W`，权重梯度计算 `dY.T @ X`，三者均调用 CUTLASS。BF16 autocast 下，乘法操作数转为 BF16，参数保留 FP32；参数梯度返回原参数 dtype。替换层保留原 Parameter 对象和参数共享关系。

`build_key()` 包含编译配置、PyTorch 版本与 ABI、模块内的 Python/C++/CUDA 源码。训练将该标识写入 `transform_id`。修改这些内容后，重新启动进程编译，并使用新输出目录训练。

[test_linear.py](../tests/cutlass_gemm/test_linear.py) 使用替代后端 `torch.mm` 检查 Python 桥接行为；[test_cuda.py](../tests/cutlass_gemm/test_cuda.py) 检查实际 CUTLASS 运算。[gemm-accept.sh](../scripts/gemm-accept.sh) 为 GPU 检查入口。检查步骤见 [GEMM 教程](../docs/gemm.md)。

## FP8 与 FP4 训练连接

`low_precision.train` 复用公共 Trainer，使用 NVIDIA extra 中固定的 Transformer Engine 上游 PR #3669 提交 `fb98b935d6054a2fc68137951051695de3acaee3`，版本为 `2.21.0.dev0`。该提交在官方快照的 SM120/121 RHT 适配和 CUDA 12.8 编译修复上，修正一处 RHT 矩阵参数转发遗漏。核心与 PyTorch 扩展从同一源码构建。NVIDIA 使用 cuDNN 9.12，PyTorch 2.9 和 CUDA 12.8 保持不变。BF16 的 CUTLASS 基线保持独立。低精度组件调用 TE 的 cuBLASLt GEMM，后续 ABFT 的可修改内核仍在 CUTLASS 中逐项实现。

- FP8 使用 `Float8CurrentScaling(fp8_format=Format.HYBRID)`：前向 E4M3，反向梯度 E5M2，动态缩放。
- FP4 使用 `NVFP4BlockScaling()` 默认训练机制：E2M1 码值、分块尺度、Hadamard 变换、随机舍入及权重二维量化。没有使用 MXFP8 或高精度模拟 GEMM。
- 主权重为 FP32，外围 autocast 为 BF16。FP4 层将来自 Qwen 的 FP32 输入转为 BF16，再交给官方量化器执行 RHT 和 FP4 量化，保留输入梯度。低精度 GEMM 使用 FP32 累加；FP8 的 split accumulator 设置也写入运行记录。
- `adapter.install()` 先检查全部线性层，再用 TE Linear 替换，保留原 Parameter、共享权重与模块别名。每层 `forward()` 内进入 recipe 上下文，供梯度检查点重计算重新进入相同配方。
- 两个配方不维护历史校准状态，TE `_extra_state` 可为空张量。配方由固定的模型变换重新建立，模型、优化器和 RNG 沿用公共检查点保存与恢复。`run.json.low_precision` 记录配方、组件版本和官方源码提交；这些信息及适配代码共同决定模型变换标识。
- NVFP4 的输入、输出维度及展开 token 数均要求 32 对齐，对应官方 CUDA 转置路径。公共 Qwen 配置使用 512 token。未满足条件时直接报错。

上游依据：[SM120/121 RHT 修复](https://github.com/NVIDIA/TransformerEngine/pull/3300)、[CUDA 12.8 编译修复](https://github.com/NVIDIA/TransformerEngine/pull/3602)、[参数转发修复 PR #3669](https://github.com/NVIDIA/TransformerEngine/pull/3669)、[固定源码](https://github.com/NVIDIA/TransformerEngine/tree/fb98b935d6054a2fc68137951051695de3acaee3)。PR #3669 在接入时尚未合入；项目固定其完整提交，不跟随浮动分支。

`tests/low_precision/test_adapter.py` 使用替代层检查连接、参数共享和重计算；`test_low_precision_cuda.py` 在 SM120 上执行真实 TE 内核，观察三类 GEMM 的低精度操作数，并验证参数更新和量化状态保存恢复。没有 GPU 时会跳过后者，不能据此记录为硬件训练通过。

## 已有验证记录

以下内容迁自原 GEMM 验证说明，记录日期为 2026-10-13，对应提交 `1c79d21`：

- 在项目 Docker 中完成 SM120a 扩展编译、链接和加载，构建标识为 `5d8c186a22f49b11`。
- 测试通过 175 项，另有 49 项子测试通过；Ruff 通过。
- Python 桥接测试使用 `torch.mm`，覆盖前向、反向、参数共享、参数更新与检查点恢复。
- 当时未执行实际 GPU 检查及完整 Qwen3-0.6B 训练。

2026-10-14 在 RTX 5090 上调整 BF16 分块后，构建标识为 `e0bdef356dd8c139`。CUTLASS 检查共 21 项通过，其中包含 11 项实际 GPU 用例；完整检查共 186 项及 49 项子测试通过，Ruff 通过。

同日完成 Qwen3-0.6B 的 CUTLASS BF16 继续预训练，共 1000 个参数更新步。训练中断后从 `checkpoint-100` 恢复，重跑的 110—140 步日志中，loss、梯度范数和学习率与中断前一致。最终验证 loss 为 `2.6606974601745605`，perplexity 为 `14.306263668996365`。配置、日志和恢复记录保存在 `results/validation/5090/`，完整检查点保存在服务器的 `results/training/qwen3_cutlass_bf16/`。

同日使用上述固定 TE 提交完成 FP8、NVFP4 接入验证。上游 SM120 RHT 回归测试 16 项通过；项目完整检查 222 项及 49 项子测试通过，Ruff 通过。其中 5 项低精度 GPU 测试检查 cuDNN、真实低精度前向与反向 GEMM，以及带共享权重的微型 Qwen 在公共 Trainer 中恢复模型、优化器和随机状态。

Qwen3-0.6B 在同一 WikiText-103 数据配置下分别完成 20 个参数更新步，FP8 验证 loss 为 `2.818451404571533`，NVFP4 为 `3.0721590518951416`。这两次为短程训练接入验证，不用于判断长期收敛或比较性能；短程运行不写完整检查点，实际恢复由上述 GPU 测试覆盖。结果汇总和日志保存在 `results/validation/5090/fp-training/summary.json` 及同目录中带 `fb98` 的日志，训练记录保存在 `results/training/qwen3_fp8_te221_integration/` 和 `results/training/qwen3_fp4_te221_integration/`。

5090 服务器的默认镜像 `sdc-nvidia:py311` 已更新为 `sdc-nvidia:py311-te-fb98`，旧镜像保留为 `sdc-nvidia:py311-bf16-20261014`。从新镜像独立创建容器后，设备检查及上述 5 项 GPU 测试再次通过，日志为 `fresh-image-gpu.log`。镜像使用同一份已验证的 TE 编译包，不依赖维护容器的源码和编译缓存。

后续硬件运行记录设备、提交版本、执行命令和结果，保存在实验输出目录。教程只维护操作步骤。
