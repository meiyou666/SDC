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

## 启用 ABFT

从 [qwen3_abft_debug.yaml](../configs/training/qwen3_abft_debug.yaml) 开始。它运行两步 BF16 训练，启用 `classic` 两路加权校验。准备好公共数据后执行：

```bash
bash scripts/train.sh train --config configs/training/qwen3_abft_debug.yaml
```

复制到个人实验配置后，可修改以下字段；精度与缩放组合沿用[训练教程](training.md#选择-gemm-精度)。

```yaml
gemm:
  backend: cutlass
  precision: bf16
  scaling: none
  global_scale: amax
  abft:
    method: classic
    reference_mode: input_fused
    atol: 0.0001
    rtol: 0.0001
    recovery: none
    on_error: report
    reference_block_k: 256
    diagnostics: false
  fault:
    location: none
    row: 0
    col: 0
    bit: 0
    call_index: 1
    layer: null
    role: null
  report_max_records: 4
  report_path: null
```

- `abft: none` 关闭校验，`fault.location: none` 关闭注入；两者关闭时走原来的官方 GEMM。
- `classic` 使用全 1 和递增行权重，生成两路列校验。阈值为 `atol + rtol × magnitude`，其中 `magnitude` 是绝对值操作数的加权乘积。示例阈值用于调试，实验前按精度与操作数校准。`reference_block_k` 控制参考校验的 K 分块，取正的 16 倍数。
- `on_error: report` 返回报警报告；`raise` 在未恢复时抛出 `AbftError`。`recovery: recompute` 在报警后用保留的干净码值与尺度重算一次，并使用相同校验复检；复检通过才替换输出。
- `fault.location` 可选 `a`、`b` 或 `accumulator`。前两者翻转准备好的输入码值，`row/col` 使用逻辑矩阵坐标，`bit` 使用对应格式的位编号。`accumulator` 翻转 epilogue 中乘 alpha 后、输出转换前的 FP32 值，位编号为 0—31。此位置位于 MMA 计算之后。
- `call_index` 从 1 开始，按全模型实际 GEMM 调用计数；`layer` 和 `role` 可进一步限定该次调用。`role` 为 `forward`、`dgrad` 或 `wgrad`。梯度重计算和验证也会增加计数。开启注入时使用 `resume: null`。

报告包含检测、非有限值、候选定位、注入命中、恢复及复检结果。`fault_hit` 表示注入动作执行，`location` 是单输出元素错误假设下的候选坐标。`report_max_records` 保留最近若干报告；`diagnostics: true` 额外保留完整残差和阈值张量。设置 `report_path` 后逐调用写 JSONL。设备张量报告本身不触发主机同步；`raise`、`recompute` 的主机判定以及 `to_record()`、JSONL 写入会同步，测量开销时按实验目的配置。

## 添加校验方法

当前提供 `classic` 示例。新增方法在 [src/abft/methods.py](../src/abft/methods.py) 继承 `AbftMethod`，实现下列接口，并用 `register_method("方法名", 方法工厂)` 注册。自定义模块须在训练开始前导入，随后在 YAML 的 `abft.method` 中选择该名称。

方法专用参数放在 `abft.parameters` 中，由方法的 `validate_config(config)` 检查；参数需能保存为 JSON。当前 `classic` 仅使用已有的阈值字段，不接受额外参数。

| 接口 | 实现内容 |
| --- | --- |
| `validate_config(config)` | 检查该方法的专用参数 |
| `prepare(operands, config, mm=None)` | 注入前从干净操作数生成参考校验，返回含 `weights` 的状态；当前原生接口接受形状为 `(2, M)` 的 FP32 行权重 |
| `prepare_fused(operands, config)` | 准备输入融合所需的权重与方法状态，不独立扫描操作数 |
| `bind_reference(predicted_partials, magnitude_partials, state)` | 接收并组织主 GEMM 内生成的参考数据 |
| `propagate(partials, state)` | 合并原生 epilogue 返回的分块列投影 |
| `verify(observed, output, state, config)` | 返回设备张量字典，包含 `detected`、`nonfinite`、`located`、`location` 等判定 |
| `recover(operands, state)` | 返回重算输出、投影和注入命中标记；默认复用干净的已准备操作数 |

默认 `implementation: reference` 通过上述方法接口执行，每次受保护 GEMM 调用都会执行方法工厂。需要历史状态的方法自行管理实例生命周期；当前训练检查点不保存插件状态。`implementation: native` 是下述 BF16 增广 `classic` 的专用实现，不接受自定义方法或额外 `parameters`。V-ABFT、GVFA、Strided ABFT 和 AEGIS 的机制需分别实现与验证。

输出融合位于 epilogue：两路权重对乘 alpha 后、输出 cast 前的 FP32 值做列投影。FP32/BF16 通过官方 example 35 的 visitor 接口接入，FP8/NVFP4 通过 `Sm90RowReduction` 接入。选择 `input_fused` 时，方法以 `prepare_fused()` 准备权重，再由主循环复用已加载输入生成参考；`bind_reference()` 接收原生预测与幅值。选择 `separate` 时调用原来的 `prepare()` 独立生成参考。分块投影合并和判定由 Python 层组织。修改权重数量或原生投影布局时，同时修改 CUDA 接口。文件及固定源码链接见[技术参考](../.agent/reference.md#官方源码对应)。

## 测量 GEMM 保护开销

在 NVIDIA 容器中运行：

```bash
python -m abft.benchmark --config configs/gemm/abft_benchmark.yaml
```

只跑 BF16 增广矩阵 ABFT 并查看调试输出时，使用 `configs/gemm/my_gemm.yaml`。该配置显式选择 `implementation: native`，不加载模型或训练数据。

```bash
python -m abft.benchmark --config configs/gemm/my_gemm.yaml
```

控制台每组显示精度与矩阵尺寸、一张耗时表，以及输出逐位一致性和报警次数，不逐次打印测量日志。`progress: true` 额外显示加载和设备信息，设为 `false` 不影响每组结果输出。

示例的 `preview_size: 0` 关闭矩阵预览；设为 `4` 可查看输入和首组输出的左上角 4×4，最大为 16。FP32/BF16 只显示实际 GEMM 输入，FP8/NVFP4 显示转换前输入并注明位置。所有预览、比较和打印均在计时区间外。

### BF16 增广矩阵路径

`abft.reference_mode: augmented` 保留完整的官方 `128×128×64` 数据 tile，额外附加 2 行校验，逻辑输出为 `130×128`。从已加载的共享 A 计算 `W×A`，将编码写成 BF16 校验行，再用附加的官方 MMA fragment 计算 `q×B`。原 128 行数据的主 MMA 和分块坐标保留；附加 2 行使用额外 MMA 片段承载，并非一条 130 行 MMA 指令。权重、数据和校验行均为 BF16，矩阵乘法使用官方 FP32 累加。两侧校验在主结果转换为 BF16 之前取数，返回的输出仍是原形状的 BF16 矩阵。

在同一份配置的 `abft` 下选择实现：

```yaml
abft:
  method: classic
  reference_mode: augmented
  threshold_scale: output
  implementation: native  # 改为 reference 可运行原实现；省略时默认为 reference。
```

`native` 用官方 CuTe 的 LDSM 矩阵加载操作，从共享 A 取出 `WA` 编码所需的寄存器片段，减少逐元素加载与寻址。编码的 8 次 MMA 顺序、BF16 转换位置和原 128 行数据 tile 保持不变。随后复用主 MMA 已加载到寄存器中的 B，按固定线程布局归约输出校验，再由一个设备内核完成分块合并、阈值比较和报告标记。

仅缓存不可变的校验权重，每次调用独立分配结果与临时缓冲。`reference` 保留逐元素共享 A 加载、共享 B 的附加乘法和 Python 组织的校验合并，便于对照。两者都保留两路完整列校验；原 mainloop 语句顺序保留不代表流水性能不受新增计算、寄存器和同步影响。使用上述 `implementation: native` 配置即可启用，无须增加启动参数。

当前增广路径支持 BF16，A、B 均可为 packed 行主序或列主序，遵循原官方 GEMM 的对齐要求。编码按每个 128 行数据 tile 独立进行，不复制整份 A，也不重新从显存遍历 A/B。其他精度继续使用原路径，不会静默回退。

此模式需显式设置 `abft.threshold_scale: output`，其阈值为 `atol + rtol × Σ|w*C_acc|`；旧 `input_fused/separate` 使用默认 `threshold_scale: input`，对应 `|W||A||B|`。输出幅值是可校准的尺度，不是旧输入误差上界。校验行写为 BF16 引入的舍入误差属于正常残差；示例保留原调试容差并如实记录报警，正式误报率和检出率实验须在独立正常样本上校准。

增广路径支持 `fault.location: accumulator` 和一次重算复检；复检沿用首次调用保存的参考校验。BF16 行号权重可能重复，原 FP32 的比值定位规则不适用，因此当前报告 `located: false`。当前验证范围是已有的累加结果注入点；复用 B 寄存器时，若 B 本身被破坏，数据与参考可能共同受影响，不能据此声称覆盖操作数故障。完整 FP32 标量参考仍可通过 `reference_mode: input_fused` 与 `threshold_scale: input` 选择作对照。

配置指定精度、矩阵形状 `[M,N,K]`、输入种子、阈值、预热和测量次数。两条路径使用同一份已准备好的码值、scale 与布局，量化和布局转换在计时之前完成。保护路径从参考校验开始计时，到输出与设备端判定完成结束；参考解码和校验传播仍计入保护开销。

输出 JSON 保留无保护与保护时延、配对比例、逐次结果与完整样本。测量顺序交替，报告在计时外读取。正常重复调用使用同一输入，报警比例用于检查当前配置，不替代独立正常样本上的阈值校准。

`profile_repeats` 大于 0 时，额外用 PyTorch profiler/CUPTI 采集阶段耗时，控制台表格只显示 GPU 活动中位数，单位为微秒（μs）。`reference` 分别列出校验准备、融合 GEMM、checksum 合并、检测、候选定位与报告状态；`native` 列出权重准备、融合 GEMM 和一次合并判定（`abft.finalize`）。总计已包含子阶段，不能重复相加。CPU 范围耗时、均值和完整样本保留在 JSON 的 `stage_profile` 中；该次采集独立于主测量样本。

`profile_repeats: 0` 时，控制台改为显示 CUDA Event 完整调用时延，并标明可能包含 CPU 提交间隙。这与 CUPTI 的 GPU 活动累计时间口径不同，不能混用。JSON 中的 CPU 范围还含提交、等待和 profiler 开销，不能与 GPU 时间相加。

`reference` 的原生 GEMM 调用还包含校验缓冲初始化；`native` 由内核写满本次缓冲，不另发预清零内核。识别到 CUPTI 的 CUTLASS 内核记录时，输出会进一步区分 `cutlass_kernel_ms` 和 `other_gpu_activity_ms`，避免将初始化等 GPU 活动混称为单个内核耗时。

编译、输入准备与量化均先完成；主测量和分阶段采集分别预热。测量期间不打印矩阵或结果；采集结束后统一输出。融合内核内部的主乘法和 checksum 作为整体测量，未拆成独立内核。平台无法提供 CUDA 活动时间时会明确报错，不以零时间代替。

自行编写实验时，先调用 `PreparedGemm.prepare()`，再将同一对象交给 `raw_native()` 和 `abft.execute_prepared()`。`execute()` 继续作为训练入口，负责准备一次输入并调用后者。`abft.timing` 默认关闭，只在需要阶段计时时开启。

四种精度均已接入 `abft.reference_mode: input_fused`，示例配置使用此模式；`separate` 保留独立参考计算。比较时分别保存两份配置，采用相同输入、权重与阈值。API 未指定 `reference_mode` 时仍保留 `separate` 行为。

输入融合在官方主循环的 A、B tile 就绪后计算参考校验，复用共享内存数据；NVFP4 同时读取该 tile 的 block16 scale。内核返回预测、幅值和结果校验的局部和，随后沿 M tile 合并判定。全局两路行权重及原 MMA 顺序保持一致；FP32、BF16、FP8、NVFP4 分别保留原来的 2、3、6、4 级流水线。此配置使用计算故障注入；`fault.location: accumulator` 对应已有的 epilogue FP32 注入点。

`tests/cutlass_gemm/` 中的检查覆盖多 K 阶段、余数块、持久化调度、异构 scale、输出对照、恢复与线性层前反向。运行 `bash scripts/gemm-accept.sh` 执行 GPU 检查。各实现的实际验证范围见[已有验证记录](../.agent/reference.md#已有验证记录)；编译成功不能代替 GPU 数值与性能验证。
