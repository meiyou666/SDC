# Agent 技术参考

开始任务时先读 [AGENTS.md](../AGENTS.md) 和[项目研究说明](../docs/project.md)，再按下表读取相关实现。成员的操作步骤集中在[环境](../environment/README.md)、[训练](../docs/training.md)和 [GEMM](../docs/gemm.md) 三篇教程。

## 按任务阅读

| 任务 | 配置与实现入口 |
| --- | --- |
| 环境、依赖与容器 | [baseline.json](../environment/baseline.json)、[pyproject.toml](../pyproject.toml)、[uv.lock](../uv.lock)、[Dockerfile](../environment/docker/Dockerfile)、[Compose](../compose.yaml)、[Dev Container](../.devcontainer/) |
| 云端初始化 | [setup-cloud.sh](../environment/docker/setup-cloud.sh)、[configure-cloud.py](../environment/docker/configure-cloud.py) |
| 平台容器虚拟环境 | [setup-platform.sh](../environment/setup-platform.sh)、[activate.sh](../environment/activate.sh)、[configure_platform.py](../environment/configure_platform.py) |
| 训练与数据 | [训练配置](../configs/training/qwen3_06b.yaml)、[src/training](../src/training/)、[训练测试](../tests/training/) |
| CUTLASS 与线性层 | [gemm.json](../environment/gemm.json)、[linear.py](../src/cutlass_gemm/linear.py)、[backend.py](../src/cutlass_gemm/backend.py)、[csrc/gemm](../src/cutlass_gemm/csrc/gemm/)、[算子测试](../tests/cutlass_gemm/) |
| ABFT 与故障注入 | [调试配置](../configs/training/qwen3_abft_debug.yaml)、[src/abft](../src/abft/)、[csrc/abft](../src/cutlass_gemm/csrc/abft/)、[codegen](../src/cutlass_gemm/codegen/)、[使用与扩展方法](../docs/gemm.md#启用-abft) |
| BF16 算法接入 | [开发教程](../docs/reproduction/bf16-algorithm-guide.md)、[方法接口](../src/abft/methods.py)、[GPU policy](../src/cutlass_gemm/csrc/abft/common/policy.cuh)、[独立 GEMM 配置](../configs/gemm/my_gemm.yaml) |
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
- Python、NumPy 和 Torch 的随机种子在模型加载与变换之前设置。模型自身的附加状态注册为参数或持久化 buffer；ABFT 方法历史和 RNG 通过 `MethodSession` 的状态接口保存，见后文。

`run.json` 保存配置签名、数据摘要、锁文件哈希、提交版本、设备、模型变换标识和参数。续训时比对配置、数据、锁文件和模型变换信息，并检查检查点中的模型、优化器、调度器和随机状态。缺少模型变换记录的旧运行使用新目录重新训练。

非有限指标写为 `null`，原值记录在 `nonfinite_metrics` 中。loss 非有限时停止训练并保存失败原因；其他指标非有限时继续运行。输出文件用途见[训练教程](../docs/training.md)，不在此重复维护。

## CUTLASS 调用关系

统一 Trainer 根据 `gemm` 配置将 Qwen 的全部 `nn.Linear` 替换为 `CutlassLinear`，包括 attention 投影、MLP 与 `lm_head`。前向 `X @ W.T`、输入梯度 `dY @ W` 和权重梯度 `dY.T @ X` 均调用所选精度的 CUTLASS GEMM。替换保留原 Parameter 对象、共享权重与模块别名，其他算子沿用原模型。

源码分为 Python 入口、`codegen/` 和 `csrc/`。跟踪一次调用时，先读 `linear.py` 和 `backend.py`，再读绑定及所选精度的无保护 GEMM；启用保护后再读 `src/abft/` 与对应精度的 `csrc/abft/`。`codegen/` 负责构建时的官方源码适配，不是另一个运行入口。

| 文件 | 职责 |
| --- | --- |
| [linear.py](../src/cutlass_gemm/linear.py) | 统一线性层与 autograd 适配 |
| [backend.py](../src/cutlass_gemm/backend.py) | 读取编译配置、生成构建标识、编译加载扩展、检查设备 |
| [quantization.py](../src/cutlass_gemm/quantization.py) | Torch 缩放与官方 CUTLASS 类型转换连接 |
| [csrc/bindings.cpp](../src/cutlass_gemm/csrc/bindings.cpp) | 注册 PyTorch 算子及 CUDA 实现 |
| [csrc/gemm/fp32_bf16.cu](../src/cutlass_gemm/csrc/gemm/fp32_bf16.cu) | FP32 与 BF16 官方 GEMM |
| [csrc/gemm/fp8.cu](../src/cutlass_gemm/csrc/gemm/fp8.cu) | 官方 SM120 E4M3 GEMM，FP32 列主序输出 |
| [csrc/gemm/nvfp4.cu](../src/cutlass_gemm/csrc/gemm/nvfp4.cu) | 官方 example 79a GEMM，BF16 输出及 SF 布局辅助函数 |
| [codegen/sm80.py](../src/cutlass_gemm/codegen/sm80.py) | 从固定官方 SM80 主循环生成 FP32/BF16 输入融合与 BF16 增广类型 |
| [codegen/sm120.py](../src/cutlass_gemm/codegen/sm120.py) | 从固定官方 SM120 kernel 生成 FP8/NVFP4 输入融合类型 |

四种 GEMM 均使用 FP32 累加。FP32 外围精度为 FP32，其余外围精度为 BF16；主权重保持 FP32，参数梯度返回原参数 dtype。BF16 使用官方 SM80 单测的 `128×128×64` 分块和默认三阶段配置，共享内存为 96 KiB。FP8 保留官方单测的 FP32 列主序输出，前向再转为 BF16，反向返回对应输入和参数的 dtype。NVFP4 使用 example 79a 的 BF16 输出配置。

`gemm.precision/scaling` 对应 `fp32/none`、`bf16/none`、`fp8/tensorwise`、`nvfp4/block16`。`global_scale` 为 `amax` 或有限正数乘子；`abft` 选择 `none`、`classic`、`sum` 或已注册的方法。具体字段见 [config.py](../src/training/config.py) 与 [ABFT 配置](../src/abft/config.py)。

### ABFT 调用关系

BF16 `reference_mode: augmented` 由 `csrc/abft/fp32_bf16/bf16_augmented.cu/.h` 实现。保留官方 128 数据行 tile；Classic 附加 2 条校验行形成逻辑 130 行，Sum 附加 1 条形成逻辑 129 行，输出仍为原形状的 BF16 矩阵。片内 `W×A` 使用官方 CuTe BF16 MMA，结果 cast BF16 后保存在额外共享缓冲；附加官方 MMA fragment 计算 `q×B`，原 128 行数据主 MMA、A/B 存储和分块坐标保留。A/B 均支持 packed 行/列主序，明确拒绝其他精度，不作额外整矩阵重读。输入权重、数据和校验行均为 BF16，编码及主/附加 MMA 累加均为 FP32；两侧校验取 precast FP32 累加结果。

新模式需显式 `threshold_scale: output`，magnitude为输出幅值 `Σ|w*C_acc|`，不是旧 `|W||A||B|` 上界；报告记录实际口径。BF16编码舍入进入正常残差，不能沿用FP32阈值宣称误报率不变。全局行号权重转BF16会重复，当前关闭比值定位并返回located=false；检测和重算复检仍可用。保留旧两条参考路径作为对照，独立示例 `configs/gemm/my_gemm.yaml` 显式选择 `augmented` 与 `implementation: native`。

`abft.implementation` 默认为 `reference`，保留五张量入口和 Python 方法接口。Classic、Sum 的四精度 `input_fused` 与 BF16 `augmented` 均可选 `native`；`separate` 仍走 `reference`。`engine.execute_prepared()` 先取得方法实例、验证精度/参考模式和参数，再由 `CompiledPolicy` 选择已编译的 CUDA 类型。native 不调用 Python `verify()`，也不接受测试用 collector 替换 hook。Classic 使用全 1/递增权重的两路完整列关系，Sum 使用一路全 1 完整列关系，均未改为 strided 编码。

`csrc/abft/common/policy.cuh` 是 CUDA policy 的唯一注册位置：`Classic=0`、`kChecksums=2`，`Sum=1`、`kChecksums=1`。host dispatch 选择类型，设备内的 `encode/propagate/observe/merge/finalize` 调用真实模板实现，不在内循环按算法 ID 分支。Python `CompiledPolicy` 校验对应的 ID/通道数并记录数值格式；方法类须显式声明，继承或改写 Python `verify()` 不会自动改动 native GPU 判定。新增 GPU 公式需添加 CUDA policy、注册并同步 Python 元数据；继续使用既有官方 GEMM/MMA 与精度适配，不能只添加方法名。

四精度 `input_fused` 保留既有 FP32 参考投影和 FP32 校验累加。FP8/NVFP4 的输入码值与 scale 仍按原路径解码参与参考，不能称为原生 FP8/FP4 checksum。native 使用原五张量证据和 epilogue 输出非有限标记，再由 `common/finalize.cu` 合并判定，避免重新全局扫描 C；必要的初始化仍计入调用。所有 native 路径当前均不定位，返回 `located=false`、`location=[-1,-1]`；Classic 的 Python reference 定位规则保留原有适用范围。

BF16 增广 `native` 的数据流与资源约定如下：

1. `codegen.sm80.generate_augmented_header()` 保留官方主循环原语句顺序。在共享 A 就绪后调用 `consume()` 计算 `WA`。`native` 通过官方 CuTe `make_tiled_copy_A` 将共享 A 加载到编码 MMA 的寄存器片段：行主序 A 使用 `SM75_U16x8_LDSM_T`，列主序 A 使用 `SM75_U32x4_LDSM_N`；静态检查覆盖 copy/MMA 坐标对应与连续对齐向量。权重通过 `UniversalCopy<uint32_t>` 成对加载。主机仅在 `N%128==0 && K%64==0` 时派发 `FullKN` 实例，去除冗余 N/K 掩码但保留 M 边界；其余形状保留原掩码，包含首个 K 余数块。8 次编码 MMA 的顺序与 BF16 cast 位置保留，`reference` 保留原逐元素加载。
2. 在官方 B transform 后、原主 MMA 前调用 `propagate()`，复用 `warp_transformed_frag_B_`。B 到附加 MMA A 操作数的寄存器映射由官方 CuTe 布局推导，每个 warp 负责的 N32 半片在编译期选择，避免再次读取共享 B 或动态寄存器索引。使用官方 CuTe `recast<uint32_t>` 保留 BF16 寄存器对，q 通过官方 32 位 copy atom 加载。非 `FullKN` 实例仍分别判断每个半字的 K 边界，支持奇数 K。原 `reference` 仍从共享 B 执行附加传播。
3. q 使用双缓冲，按 K tile 交替。保留发布 q 的 CTA barrier，并结合官方 `gmem_wait` 保护缓冲生命周期，去除多余的初始化/consume 尾部同步。编码次序、BF16 cast 位置与主 MMA 顺序保留。源码去除新增 hook 后与官方一致，只能证明原语句未被重排；新增寄存器、共享内存、指令和同步仍可能改变占用率及流水性能。
4. epilogue 按固定 ThreadMap 找到每列的 8 个贡献槽，替代遍历 1024 槽及动态列索引表，保持贡献槽升序累加；每个输出向量的 8 个值共享一次行有效性/故障行判断，只加载该行的两个权重，各列仍保留边界检查。通过官方 `NumericArrayConverter` 批量转换 BF16，再用官方 `cutlass::isfinite` 检查实际输出，不重新扫描 C。每次写满局部校验、padding 与标记，使用 `at::empty` 而无预清零内核。
5. `abft_augmented_compact` 保留原实参并允许选择编译算法，默认 Classic。返回 `output`、FP32 `partials[3,channels,m_tiles,padded_N]` 和 int32 `tile_flags[m_tiles,n_tiles]`。`partials` 第一维依次为 observed、predicted、magnitude；flags bit0 表示注入命中，bit1 表示实际 BF16 输出非有限。
6. 公共 `abft_finalize`（增广）与 `abft_finalize_reference`（输入融合）合并全局列校验、比较阈值并生成设备报告，不重读 C。`N*m_tiles < 8192` 时保留单 CTA256；其余形状使用 CTA128 按列×校验通道分配工作，总网格不超过 256 CTA，再由一个 CUB 内核归约整数状态位。每个通道/列仍由一个线程按 M tile 升序累加，跨 CTA 不使用浮点原子操作。返回 `reference[2,channels,N]`（predicted、magnitude）、可选 residual/threshold、五个 bool 标记、hit、location。阈值乘加明确分开舍入，复检仍读取首次 reference。该浮点顺序可能与 Torch `sum` 不同，不能无条件声称与 `implementation: reference` 的校验逐位一致。
7. `native.py` 仅对未覆写准备逻辑的内置无状态 Classic/Sum 缓存不可变权重，按方法、设备、M 和权重 dtype 区分，最多 32 条/16 MiB；CUDA Event、流等待和 `record_stream` 保证跨流使用顺序。自定义或有状态方法执行自己的 `prepare_*()`。每次调用独立拥有 output、partials、flags 与报告张量，不缓存共享可变 scratch。`report/none` 不通过 `.item()` 读取设备判定；`raise/recompute` 按已有策略同步。
8. 报警重算使用同一份干净输入，不重复量化。首次 finalize 的 reference 留存；第二次 finalize 只用首次 reference 复检，不能以重算产生的新预测替代。失败不替换原输出，成功才采用重算结果。`located` 始终为 false，location 为 `[-1,-1]`。

当前注入与保护证据仅限 `accumulator`：epilogue 中主 MMA 结束、输出 cast 之前的 FP32 值。复用 B 寄存器可能使数据与参考共同使用受损 B，不能声称检测这类共模错误，也不能将现有注入结果推广为 MMA 内任意时刻、任意寄存器的故障覆盖。报告通过 `reference_operand_reuse: main_B_register_fragments` 记录该数据流。

2026-10-21 在8GB配额的RTX5090上完成128数据行+2附加行版本的87项GPU测试，覆盖四布局、尾块、BF16编码舍入、故障注入、恢复和线性层前反向。无保护与增广的数据输出逐位一致；官方主循环去除新增类型名和hook后逐字一致。Compute Sanitizer未能附加目标进程，不能记作内存检查通过。

相同输入按旧参考/增广/增广/旧参考顺序测量，512×1024×1024的原生调用阶段约0.221→0.119ms，2048³约0.838→0.411ms；这些数值包含缓冲初始化，不是纯内核时延。两种路径的参考精度、幅值尺度和定位支持不同，不代表相同保护效果下的加速。原调试容差下，新增广路径的三个固定正常输入均报警；每组重复100次不是100个独立样本，需另做阈值校准。服务器记录位于 `results/gemm/augmented-comparison/`，本地备份在 `results/validation/augmented-128-20261021/`。最新profiler可从CUPTI关联记录进一步区分CUTLASS内核与初始化活动。

线性层通过 `abft.engine.execute()` 准备一次输入，再交给 `execute_prepared()` 完成保护。关闭保护与注入时直接调用原有 GEMM。`classic` 使用全 1 与递增行权重，两路 FP32 列投影从 epilogue 中乘 alpha 后、输出转换前的值生成。`reference_mode: input_fused` 在主循环内复用已经加载的 A、B 及 NVFP4 block16 scale，生成预测和 magnitude；`separate` 保留按 K 分块解码及独立 FP32 GEMM 的参考路径。

| 文件 | 职责 |
| --- | --- |
| [config.py](../src/abft/config.py) | 方法、阈值、恢复与注入配置 |
| [operands.py](../src/abft/operands.py) | 准备操作数、分块解码、码值注入与原生调用 |
| [methods.py](../src/abft/methods.py) | 方法注册、`CompiledPolicy`、支持组合与参数检查、Classic/Sum 及状态接口 |
| [session.py](../src/abft/session.py) | 方法实例隔离、物理调用计数、历史与 RNG 快照恢复 |
| [engine.py](../src/abft/engine.py) | 调用编排、一次干净重算与复检、设备报告和显式序列化 |
| [native.py](../src/abft/native.py) | 四精度输入融合/BF16 增广的设备判定编排、权重缓存、首次参考复检 |
| [fp32_bf16/bf16_augmented.cu](../src/cutlass_gemm/csrc/abft/fp32_bf16/bf16_augmented.cu)、[bf16_augmented.h](../src/cutlass_gemm/csrc/abft/fp32_bf16/bf16_augmented.h) | BF16 附加校验 MMA、B 寄存器复用与静态输出归约 |
| [common/policy.cuh](../src/cutlass_gemm/csrc/abft/common/policy.cuh) | 编译算法类型注册及编码、传播、输出观察、合并与判定公式 |
| [common/finalize.cu](../src/cutlass_gemm/csrc/abft/common/finalize.cu) | 公共输入融合/增广合并、阈值覆盖、非有限值与报告标记 |
| [fp32_bf16/fp32_bf16_observe.cu](../src/cutlass_gemm/csrc/abft/fp32_bf16/fp32_bf16_observe.cu)、[fp32_bf16_observe.h](../src/cutlass_gemm/csrc/abft/fp32_bf16/fp32_bf16_observe.h) | FP32/BF16 epilogue visitor 与分块投影 |
| [fp32_bf16/fp32_bf16_reference.cu](../src/cutlass_gemm/csrc/abft/fp32_bf16/fp32_bf16_reference.cu)、[fp32_bf16_reference.h](../src/cutlass_gemm/csrc/abft/fp32_bf16/fp32_bf16_reference.h) | FP32/BF16 输入融合入口及片内参考计算 |
| [fp8/fp8_observe.cu](../src/cutlass_gemm/csrc/abft/fp8/fp8_observe.cu)、[fp8_reference.cu](../src/cutlass_gemm/csrc/abft/fp8/fp8_reference.cu)、[fp8_mainloop.cuh](../src/cutlass_gemm/csrc/abft/fp8/fp8_mainloop.cuh)、[fp8_reference.cuh](../src/cutlass_gemm/csrc/abft/fp8/fp8_reference.cuh) | FP8 输出投影、输入融合连接、共享 tile 消费与参考公式 |
| [nvfp4/nvfp4_observe.cu](../src/cutlass_gemm/csrc/abft/nvfp4/nvfp4_observe.cu)、[nvfp4_reference.cu](../src/cutlass_gemm/csrc/abft/nvfp4/nvfp4_reference.cu)、[nvfp4_mainloop.cuh](../src/cutlass_gemm/csrc/abft/nvfp4/nvfp4_mainloop.cuh)、[nvfp4_reference.cuh](../src/cutlass_gemm/csrc/abft/nvfp4/nvfp4_reference.cuh) | NVFP4 输出投影、输入融合连接及包含 block16 scale 的参考计算 |
| [common/context.cuh](../src/cutlass_gemm/csrc/abft/common/context.cuh) | 共享的 `TileReferenceContext` 声明；FP8 参考公式保留在 `fp8/fp8_reference.cuh` |
| [common/sm120_epilogue.cuh](../src/cutlass_gemm/csrc/abft/common/sm120_epilogue.cuh) | FP8/NVFP4 共用的 epilogue 注入与两路 reduction |

表中 CUDA 路径相对 `src/cutlass_gemm/csrc/abft/`。各精度目录内，`*_observe` 收集输出校验，`*_reference` 生成输入参考，`*_mainloop` 连接对应的官方主循环；BF16 的增广单独命名，公共判定位于 `common/finalize.cu`。

`benchmark.py` 的控制台按形状汇总，不输出逐次测量日志。`profile_repeats > 0` 时显示 GPU 阶段中位数表（μs）及输出逐位一致性/正常报警计数；CPU 范围、均值和完整 samples 只保留在 JSON，内部计时存储单位仍为 ms。关闭分阶段采集时显示 CUDA Event 调用时延，明确其可能包含 CPU 提交间隙。`progress` 只控制加载和设备信息，`preview_size` 独立控制矩阵预览，示例默认 0；FP32/BF16 只预览实际输入，FP8/NVFP4 的预览明确为转换前输入。汇总、报告读取与打印都在测量区间外，不能将缩减控制台输出当作内核优化收益。

原生路径保留官方 mainloop 的原语句，通过 epilogue 收集校验。`accumulator` 注入点位于 MMA 之后；独立参考路径的输入注入修改准备好的码值。恢复复用干净码值和尺度，不重复量化。独立无状态调用可由工厂创建方法；有状态调用须传入持有的 `method_instance`，或通过 `MethodSession.call` 管理，不能依赖每次新建实例保存历史。配置、报告和方法扩展步骤集中在 [GEMM 教程](../docs/gemm.md#启用-abft)。

输入融合的原生入口为 `abft_reference_mm`（FP32/BF16）、`abft_fp8_reference_mm`、`abft_nvfp4_reference_mm`，返回 `(output, observed_partials, hit, predicted_partials, magnitude_partials)`。局部数组均为 `[channels,ceil(M/tileM),padded_N]`，保持全局行权重，最终沿 M tile 合并；默认算法 ID 仍为 Classic，现有调用可继续省略。融合模式只接受计算故障，重算复检继续使用首次参考。`parameters` 提供方法专用 YAML 参数，`validate_config/prepare_fused/bind_reference` 用于扩展方法；内置 Classic/Sum 均不接受额外参数。

### 方法状态、阈值与检查点

`prepare_fused/prepare_augmented` 返回本次调用的 `state`。native 可读取其中的连续、同设备 FP32 `thresholds[channels,N]` 或 `thresholds[1]`；未提供时，公共 policy 使用 `atol + rtol*magnitude`。方法须在 prepare 阶段对可变历史、权重和随机向量取快照，避免当前计算或重试期间更新它们。engine 保留同一份阈值和首次参考用于一次重算复检，完成后调用一次 `commit(state, decision, context)`。commit 只接收观察到的判定、残差/阈值与执行元数据，不接收故障位置、注入命中或其他注入真值。

方法声明 `needs_diagnostics=True` 后，native 会为 `commit()` 保留残差与阈值；YAML 的 `diagnostics` 仅控制最终报告是否显示它们。有状态或需要这些统计的方法将提交工作计入 `state_update` 阶段；默认 Classic/Sum 不增加该阶段的计时事件。

`MethodSession` 的配置签名覆盖算法配置与版本；有状态实例按层、forward/dgrad/wgrad、设备、精度、M/N/K 和缩放配置隔离，无状态方法复用单一实例。动态 amax 数值保留在每次调用的设备操作数中，不为分组读回主机。每次实际 GEMM 请求（包括激活重计算）增加物理 `call_index`，内部恢复复用本次索引。session 不推断逻辑 replay ID；有状态算法如何处理激活重计算，必须由实验协议明确。当前有状态 session 拒绝并发主机调用和切换 CUDA stream。

独立实验使用 `with session.call(a,b,precision,scaling,global_scale,layer,role) as (method,context)`，并将 `method_instance=method` 与该 context 传给 `execute()`。`session.state_dict()` 冻结配置、物理计数和各方法的历史/RNG；`load_state_dict()` 校验 schema、算法与执行配置后恢复。算法 RNG 保存其状态张量，不把 generator 对象放进检查点。普通无状态 benchmark 不会自动建立校准历史或重置实验状态；有状态配对比较需明确校准集、初始快照、提交时机与重放规则。

训练 `_Protection` 持有 session，`AlgorithmState` 回调在 `checkpoint-<step>/abft_state.pt` 保存层映射和 session 快照，续训前恢复。无状态旧检查点缺少该文件时允许续训，物理调用计数从 0 开始；有状态方法缺少状态文件会拒绝恢复。`gemm.fault` 与 resume 的组合仍禁止，因为故障续训协议尚未验证，不再以“未保存调用计数”为理由。此接口接通不代表有状态论文算法、阈值校准或故障续训实验已经完成。

`codegen/sm120.py` 与 `codegen/sm80.py` 对固定哈希的官方源码生成独立类型，不修改 `/opt/cutlass`。两份 SM120 consumer 函数在剔除新增校验调用后与官方一致，见 `test_mainloop_preservation.py`。SM80 iterator 先处理 K 余数块，再处理完整块，参考路径的 K 掩码必须遵循该顺序。

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

本节源码引用按当前目录定位，历史提交号、构建标识、实验数值和结果目录保留原值。

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

随后接入 FP8 `input_fused` 原型：当前对应的 `csrc/abft/fp8/fp8_mainloop.cuh` 仅在固定官方 `mma()` 的 stage wait 后消费共享输入，`codegen/sm120.py` 从固定哈希的官方 kernel 生成独立类，只扩展 MMA 调用点。原始 CUTLASS 安装目录和无保护实例不改。参考部分使用全局行权重，保留原检查粒度；预测与 magnitude 作为局部和返回。寄存器保存列累加器后，新增共享缓冲为 1024 字节，流水线仍为 6 级，完整动态共享内存为 101376 字节。全扩展 `b4e2a13b8c17b118` 编译、链接与加载成功；完整检查为 301 项通过、112 项 GPU 跳过、55 项子测试通过，Ruff 通过。日志在 `results/validation/abft-input-fused/`；无实机时延或数值结论。

2026-10-15 四精度输入融合已接通，示例测量及 ABFT 训练配置使用 `input_fused`。完整扩展 `aeac1951a13c2293` 编译、链接、加载和三个 CUDA 接口注册检查通过，构建标识为 `e51127644c279838`。完整本地检查为 316 项通过、171 项 GPU 跳过、55 项子测试通过；Ruff 通过。独立检查修正了 legacy 余数块先行时的 K 掩码，以及第二路权重索引的 int32 加法溢出。最终源码对照的 6 项检查通过。

日志和 `summary.json` 保存在 `results/validation/abft-four-input/`。用户确认暂时没有目标 GPU，先推进本地实现。因此尚未执行四精度的 GPU 数值、恢复、前反向或性能测试。后续取得设备后先运行 `bash scripts/gemm-accept.sh`，再执行配对 benchmark，按实际瓶颈进行单项优化；不要将编译结果或 CPU 桥接测试视作实机结果。故障注入当前仍在 epilogue 的 alpha 后，尚未扩展为 MMA 指令内部注入。

随后在本机 RTX 4070（SM89）补充 legacy 原生正确性诊断。独立容器 `sdc-abft-sm89-check` 使用同一项目锁文件，环境与 CUDA 检查通过；仅将相同 FP32/BF16 源文件编译为 SM89，正式 `gemm.json` 和生产设备检查保持 SM120。隔离进程复用原测试，只选 FP32/BF16，并显式使用真实 SM89 能力检查。源码基于 `167b55b`。

实际运行 96 项通过：输入融合 44 项、输出校验 48 项、完整执行与线性层前反向 4 项；FP8/NVFP4 的 43 项被排除。Compute Sanitizer 的 memcheck 对 BF16 K80、FP32 K33 两项通过，0 errors；racecheck 对 BF16 K448 四种布局通过，0 hazards、0 errors、0 warnings。原始日志、逐项结果、源码哈希及诊断脚本在 `results/validation/sm89/`。这些结果补充 FP32/BF16 的真实执行证据，未测时延，也不替代 SM120 或 FP8/NVFP4 验证。

2026-10-21 在 8 GiB 配额的 RTX 5090 项目 `.venv` 中完成增广 `implementation: native` 简化的扩展编译、GPU 验证及配对测量。该次测量的扩展标识为 `0e1408d5bcdfe926`，构建标识为 `c9de0e915d97ba0a`。目标检查通过 222 项：186 项 GPU 用例、33 项 CPU 合约检查、3 项源码检查；2 项多设备用例因没有第二张 GPU 跳过。另行运行的 246 项 CPU/训练回归、5 项源码检查及 Ruff 均通过；这些检查存在重叠，不相加为总覆盖数。上文的 87 项 GPU 测试与旧增广时延属于简化前的 `reference` 实现。

三种形状均使用 seed 2026、BF16 输入、相同权重、`atol=rtol=1e-4`、`diagnostics: true` 和无故障配置。每种形状预热 5 组，按 reference/native/native/reference 顺序执行 50 组，共记录每种实现 100 个调用样本；另外独立采集 20 次 CUPTI 分阶段数据。下表为 GPU 活动时间的中位数，单位 ms；无保护列取 native 分阶段采集中的同源基线。完整 GPU 活动不含 CPU 提交间隙，编译、量化、预热、打印与报告读取均在计时外，权重缓存测量的是预热后的稳态。

| `[M,N,K]` | 无保护内核 | reference 全部 GPU 活动 | native 增广内核 | native finalize | native 全部 GPU 活动 | 较 reference 减少 | native / 无保护 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `[128,128,128]` | 0.005472 | 0.093810 | 0.016864 | 0.001664 | 0.018528 | 80.25% | 3.386 |
| `[512,1024,1024]` | 0.025664 | 0.172223 | 0.068768 | 0.003648 | 0.072432 | 57.94% | 2.822 |
| `[2048,2048,2048]` | 0.099136 | 0.480575 | 0.258478 | 0.021056 | 0.279503 | 41.84% | 2.819 |

各列分别取中位数，子阶段中位数之和不必严格等于总活动中位数。改动降低了该实现的测量开销，但保护路径仍为无保护的约 2.8—3.4 倍，尚未达到论文中约 10% 额外开销的量级。不同算法、保护范围与计时口径的论文数字不能直接作为本表的同条件比较。

300 组配对调用的数据输出逐位一致、报告标记一致；原生局部校验 partials 对照也一致。三个固定正常输入在旧调试容差下均报警，重复调用不构成独立样本误报率。2048³ 的最终 FP32 合并因求和次序不同，reference/native 残差最大差值为 1.5、阈值最大差值为 0.001953125；相对同一份 partials 的 FP64 合并，两者残差最大误差分别为 0.9541015625 与 1.162109375。因此不能声称所有最终 checksum 逐位一致，也不能从这些样本的相同标记推断临界阈值判定完全等价。

ptxas 显示各增广实例使用 255 个寄存器，部分实例存在 spill；这提示仍需测量寄存器压力，但当前未完成成功的 Compute Sanitizer 检查或 Nsight 瓶颈定位，不能据此给出运行时归因结论。配置、原始数据、分阶段结果在 `results/gemm/native-comparison/`，本地验证备份在 `results/validation/native-simplify-20261021/`。

### WA 共享加载优化（2026-10-21）

粗粒度内核对照显示，启用 `WA` 编码带来的增量最大。随后仅替换 `native` 编码的共享 A 取数方式，使用官方 CuTe LDSM copy atom；权重、编码 MMA、BF16 转换、qB 传播及校验规则不变。修改已集成到生产扩展，并通过下述正式回归；候选配对性能与正式版本验证分别记录。

候选验证已通过 21 项 GPU 检查与 33 项 CPU 合约检查。在 3 个形状 × 4 种 A/B 布局的 12 组配对对照中，输出 C、局部校验 partials 和 tile flags 均逐位一致。共享内存仍为 99,328 字节。候选 compact 实例的 ptxas 输出为 252—254 个寄存器、0 spill；旧 RR 实例为 255 个寄存器、24 字节 stack frame。寄存器数及编译输出只说明生成代码的资源变化，未测得硬件 bank conflict 或 warp stall 计数。

候选实验使用同一台 8 GiB 配额 RTX 5090、seed 2026、预热 5 次、每项 60 次采样。`comparison-variants.json` 记录四种布局的纯内核时间，以下列出 RR 中位数，单位 μs。旧版指提交 `7b72f59` 的 `native`，不同于前表的 `implementation: reference`。

| `[M,N,K]` | 旧 native 内核 | LDSM 候选内核 | 内核时延减少 |
| --- | ---: | ---: | ---: |
| `[128,128,128]` | 17.119 | 14.176 | 17.2% |
| `[512,1024,1024]` | 68.543 | 46.272 | 32.5% |
| `[2048,2048,2048]` | 257.295 | 167.743 | 34.8% |

另在同一进程采集完整 GPU 活动，包含 GEMM 与 finalize；每项仍为 60 次采样，保留 `atol=rtol=1e-4`、`diagnostics: true`、`recovery: none`。下表来自独立的 `engine-comparison.json`，不能与上一张表混算组件增量。编译、输入转换、预热、打印和报告读取均在计时之外。

| `[M,N,K]` | 无保护 | 旧 native 全部 GPU 活动 | LDSM 候选全部 GPU 活动 | 候选 / 无保护 |
| --- | ---: | ---: | ---: | ---: |
| `[128,128,128]` | 5.472 | 18.544 | 15.584 | 2.848 |
| `[512,1024,1024]` | 25.584 | 72.352 | 50.128 | 1.959 |
| `[2048,2048,2048]` | 98.559 | 278.366 | 188.832 | 1.916 |

完整 GPU 活动分别减少约 16.0%、30.7%、32.2%。三种形状的报告标记一致；固定正常输入仍在原调试容差下报警，没有放宽阈值。较大矩阵的额外开销仍约 92%—96%，尚未降至约 10%。原始结果位于服务器 `results/gemm/wa-optimization/{comparison-variants.json,engine-comparison.json}`。

正式生产扩展 `sdc_cutlass_badccdedfbeda24f` 编译、链接及加载通过，metadata build 为 `ff7014975f62aec1`。GPU 回归通过 226 项，2 项多设备检查因没有第二张 GPU 跳过，用时 15.14 秒；覆盖四种布局、余数块、CUDA 流、故障、重算恢复及线性层前反向，包含新增的 K8/K72 四项边界用例。CPU/训练组 249 项检查及 Ruff 通过；这些分组可能与候选检查重叠，不相加为总覆盖数。`cuobjdump` 显示正式 compact 四种布局分别使用 254/252/252/254 个寄存器，STACK 和 LOCAL 均为 0，与候选一致。

公开入口的正式版本复测已完成。`gemm_breakdown.py` 三种形状的完整诊断版与生产版在 C、partials、flags 上逐位一致；诊断版时延比生产版低约 2.1%—4.3%，继续保留该扰动提示。`abft.benchmark` 原示例的两种形状通过，全部 GPU 活动中位数为 15.647/50.015 μs。2048³ 另用忽略目录下的临时 YAML 复测，仅改原示例的 shape/output，预热 3 次、测量 20 次、profile 20 次：无保护 98.575 μs、GEMM 167.758 μs、finalize 20.927 μs、全部 GPU 活动 188.702 μs，为无保护的 1.91 倍；各项分别取中位数。三种形状均输出逐位一致，固定正常输入仍报警 20/20。上表仍为候选扩展的配对性能记录；正式复测用于确认公开入口与集成后的行为。日志与结果备份目录为 `results/validation/wa-ldsm-20261021/`，其下保留服务器的 `results/gemm/...` 相对目录。

本轮没有采纳的试验：权重共享 stride 从 128 改为 136 收益很小；增加完整 tile 的 mask 快路反而变慢；以 operand hash 防止编译器删除操作的细分诊断使内核时延扰动约 60%，不用于组件归因，现已恢复原粗粒度诊断。Nsight 在该共享 GPU 平台仍不可可靠采集，本轮停止排查其注入兼容问题，继续使用无打印的配对内核测量。粗粒度开关增量包含编译资源、流水线及同步变化，不能当作独立组件耗时。

### 成对取数、epilogue 与 finalize 优化（2026-10-21）

在 `e3b357d` 的 LDSM 版本上继续优化 `native`。WA 权重和 qB 传播改用官方 CuTe 32 位成对取数，epilogue 用官方 `NumericArrayConverter` 与 BF16 `isfinite`，并把重复的行判断移到每个输出向量外层。满足 N128/K64 对齐时由主机选择 `FullKN` 实例，保留 M 掩码，其余形状包括奇数 K 仍使用原边界判断。没有移动 `consume()` 或原主循环语句，也没有改变原 128 行数据 tile、8 次编码 MMA、BF16 cast 或两路完整列校验。

finalize 在 `N*m_tiles >= 8192` 时按列和校验通道并行，再汇总整数标记；小形状保留单 CTA，避免多一次启动。FP32 M tile 累加顺序、阈值乘加舍入和首次 reference 复检语义不变。修改已集成生产扩展并通过正式回归，下表仍记录候选配对结果。

按官方 `m16n8k16` atom 计数，每个 CTA 每个 K64 tile 的主 GEMM 调用 512 次 warp MMA（4 warp × 4 K 组 × 4 M16 × 8 N8）；WA 编码调用 32 次（4 warp × 8 reduction），qB 传播调用 32 次（4 warp × 4 K 组 × 2 M16）。额外 warp MMA 调用数为主计算的 12.5%，两条校验在 atom 的 N8 维中另有 6 列填充。这是指令调用数量，不能作为实测周期或耗时下限，也不能从约 40% 的完整开销中直接相减来归因其他成本。

同一台 8 GiB 配额 RTX 5090 上，使用 seed 2026、预热 5 次、每项 60 次采样。保持 BF16、`atol=rtol=1e-4`、`diagnostics: true`、`recovery: none`。下表为完整 GPU 活动中位数，单位 μs，包含 GEMM 与 finalize 的全部内核；编译、量化、预热、打印和报告读取不计入。

| `[M,N,K]` | 无保护 | 旧 native（`e3b357d`） | 优化候选 | 候选 / 无保护 |
| --- | ---: | ---: | ---: | ---: |
| `[128,128,128]` | 5.472 | 15.649 | 11.184 | 2.044 |
| `[512,1024,1024]` | 25.600 | 50.064 | 39.840 | 1.556 |
| `[2048,2048,2048]` | 98.591 | 189.917 | 138.222 | 1.402 |

候选的输出 C 与报告逐位一致，固定正常输入仍报警，没有调整容差。2048³ 的额外开销从约 92.6% 降至 40.2%，仍未达到约 10%。

正式生产扩展 `sdc_cutlass_a827091076c0a43c.so` 编译、链接及加载通过，metadata build 为 `f62e761b24d2064c`。正式 GPU/合约组通过 245 项，2 项多设备检查跳过，用时 17.34 秒；覆盖 K3/K65、`FullKN` 的 M 余数块、packed NaN/Inf、`N*m_tiles=8191/8192/8193` 的并行分派边界及 parallel reference 复检。CPU 组 249 项及 Ruff 通过，诊断扩展编译通过。各分组存在重叠，不相加为总覆盖数。

公开入口复测已完成。`my_gemm.yaml` 两种形状的正式完整 GPU 活动中位数为 11.328/39.840 μs；2048³ 使用单独 YAML，预热 3 次、测量 20 次、profile 20 次，得到无保护 98.895 μs、GEMM 134.575 μs、finalize 3.265 μs、全部 GPU 活动 137.872 μs，为无保护的 1.394 倍。各项分别取中位数，不要求子项中位数相加等于总值。三种形状输出逐位一致，固定正常输入均仍报警 20/20。正式复测与上表候选的 60 次配对测量分开记录。

公开诊断的三个原形状，以及追加的 masked 尾块 `[136,144,80]`、`FullKN` M 尾块 `[129,128,64]`，均与生产版的 C、partials、flags 逐位一致。三个原形状的诊断时延扰动分别为 -2.12%、+2.22%、+2.98%，两个尾块分别为 +2.39%、-1.52%，因此继续保留扰动提示。

把 qB 传播移到主 MMA 之后的调度试验没有稳定收益，未纳入生产。q 共享存储 stride 从 64 改为 72 也没有跨布局的稳定收益：2048³ 的 RR 内核仅从 133.839 降至 133.631 μs，CR 则从 133.120 升至 134.128 μs，因此保留原 stride。未采集到硬件 bank conflict 计数，不能由该试验反推冲突情况。诊断 `core` 仍有 q 初始化、sink 写回、输出有限性等工作，不能当作纯主 GEMM，也不能将其相对无保护的全部差值归因于寄存器。不同候选的中位数差包含资源与调度相互影响，不能相加为独立阶段成本。

原始结果位于服务器 `results/gemm/remaining-optimization/`：`comparison.json`、`comparison-control.json`、`comparison-combined.json`、`comparison-post.json`、`comparison-qpad.json`、`finalize-channels-comparison.json`、`finalize-clean-comparison.json` 与 `engine-comparison.json`。完整 GPU 活动表来自最后一个文件；本轮备份目录为 `results/validation/remaining-optimization-20261021/`。
