# BF16 ABFT 算法接入教程

本教程供队员在现有 BF16 CUTLASS GEMM 上接入自己的 ABFT 算法。先运行已实现的 Classic 和 Sum，再按算法需要修改权重、阈值或 GPU 校验公式。主 GEMM、训练入口和公共环境继续复用仓库实现。

## 1. 激活环境并编译

在已配置的项目 Docker 或云容器项目 `.venv` 中，进入仓库根目录执行：

```bash
source environment/activate.sh
python environment/verify_environment.py --accelerator
python -m cutlass_gemm.backend
```

首次配置环境见[环境教程](../../environment/README.md)。当前构建目标是 SM120a，运行设备要求 SM120（如 RTX 5090）。编译命令加载固定版本的官方 CUTLASS，并输出扩展构建信息。

## 2. 先运行 Classic，再切换现有 Sum

复制公共配置，个人实验只修改副本：

```bash
mkdir -p experiments
cp configs/gemm/my_gemm.yaml experiments/bf16_classic.yaml
cp configs/gemm/my_gemm.yaml experiments/bf16_sum.yaml
```

在两个副本中修改以下值，其余设置先保留：

| 配置文件 | `abft.method` | `output` |
| --- | --- | --- |
| `experiments/bf16_classic.yaml` | `classic` | `results/gemm/bf16_classic.json` |
| `experiments/bf16_sum.yaml` | `sum` | `results/gemm/bf16_sum.json` |

确认两份配置均使用以下组合。矩阵形状、种子、容差和采样次数都在 YAML 中调整。

```yaml
formats:
  - {precision: bf16, scaling: none, global_scale: amax}
abft:
  method: sum                # Classic 副本改为 classic
  implementation: native
  reference_mode: augmented
  threshold_scale: output
  atol: 0.0001
  rtol: 0.0001
  recovery: none
  on_error: report
  diagnostics: true
  timing: false
```

上面是需要核对的配置片段，保留副本中的 `shapes`、`seeds`、预热和采样等其他字段。然后分别运行：

```bash
python -m abft.benchmark --config experiments/bf16_classic.yaml
python -m abft.benchmark --config experiments/bf16_sum.yaml
```

默认测试 `A[128,128]×B[128,128]` 和 `A[512,1024]×B[1024,1024]`，不加载 Qwen 或训练数据。控制台显示无保护 GEMM、ABFT GEMM、合并判定、总 GPU 活动时间，以及输出逐位一致性和正常报警次数；完整样本保存在各自 JSON 中。编译、输入准备和打印不计入测量。

**Sum 已经实现并注册，切换本例只需改配置。** Classic 有两路校验，Sum 只有一路全 1 校验；两者输出矩阵相同，保护关系不同。正常随机输入也可能在默认调试容差下报警，先保留残差和阈值，后续用独立正常样本校准。

## 3. 一次 GEMM 的输入、数据流和输出

Python 入口接收同设备的 `A: BF16[M,K]`、`B: BF16[K,N]`，支持 packed 行主序或列主序。`PreparedGemm.prepare()` 先准备输入，`execute_prepared()` 再执行保护。

1. **准备校验权重。** `prepare_augmented()` 返回连续、同设备的 BF16 `weights[channels,M]`。Classic 的 `channels=2`，分别为全 1 和递增行权重；Sum 的 `channels=1`，只有全 1。
2. **读取 tile 并编码。** 官方主循环仍使用 `128×128×64` tile。A/B 已进入共享内存后，额外的官方 MMA 计算 `W_tile×A_tile`，FP32 累加结果转成 BF16 校验行，放入 CTA 共享缓冲。
3. **主计算与校验传播。** 原 128 行数据照常计算；附加 MMA 用校验行与 B 计算预测校验，复用主计算已加载的 B 寄存器。Classic 逻辑上附加 2 行，Sum 附加 1 行，数据 tile 不会变成 126 行。
4. **观察输出并判定。** epilogue 从转换前的 FP32 `C_acc` 计算列校验和 `Σ|w*C_acc|`，同时检查转换后 BF16 输出的有限性。公共 finalizer 合并各 M tile，比较 FP32 残差与 `atol + rtol×Σ|w*C_acc|`。

| 返回内容 | 类型与形状 |
| --- | --- |
| `output` | BF16 `[M,N]`，正常矩阵乘法输出，不附带校验行 |
| `report.detected`、`report.nonfinite` 等标记 | GPU bool 张量；当前 native 不做定位，`located=false` |
| `report.residual`、`report.threshold` | `diagnostics: true` 时返回 FP32 `[channels,N]` |
| 原生内部 `partials` | FP32 `[3,channels,ceil(M/128),ceil(N/128)*128]`，三项依次为 observed、predicted、magnitude |

BF16 的输入和校验行使用 BF16，MMA 累加及最终比较使用 FP32。权重和中间编码转成 BF16 会引入舍入，可能改变论文的数值假设。接入论文方法时，要逐项核对校验关系、舍入位置、累加精度、检查位置和阈值尺度；使用 FP32 比较本身不等于实现了某篇论文的方法。

## 4. 按改动类型选择接口

| 要改什么 | 从哪个文件开始 |
| --- | --- |
| 权重、阈值、支持组合及方法参数 | [src/abft/methods.py](../../src/abft/methods.py) |
| GPU 编码、传播、输出观察、合并及判定公式 | [common/policy.cuh](../../src/cutlass_gemm/csrc/abft/common/policy.cuh) |
| BF16 校验片段的取数、存储或布局 | [bf16_augmented.h](../../src/cutlass_gemm/csrc/abft/fp32_bf16/bf16_augmented.h)、[bf16_augmented.cu](../../src/cutlass_gemm/csrc/abft/fp32_bf16/bf16_augmented.cu) |
| 公共证据合并和设备报告 | [common/finalize.cu](../../src/cutlass_gemm/csrc/abft/common/finalize.cu) |
| 调用编排、重试和状态提交 | [engine.py](../../src/abft/engine.py)、[native.py](../../src/abft/native.py) |
| 有状态方法实例和快照 | [session.py](../../src/abft/session.py) |

只做 BF16 native 方法时，可以继承现有 `Classic` 或 `Sum`，复用它们的参考侧方法，不必重写所有接口。保留相应的既有 GPU policy，在方法类中显式声明 `CompiledPolicy`，按需覆盖 `validate_config()`、`prepare_augmented(operands, config)`；有状态方法再实现 `commit()` 和状态保存恢复。

`prepare_augmented()` 返回本次只读 `state`：

| 状态字段 | 要求 |
| --- | --- |
| `weights` | 连续、同设备 BF16 `[channels,M]`，通道数与编译 policy 一致 |
| `thresholds`（可选） | 连续、同设备 FP32 `[channels,N]` 或 `[1]`；未提供时使用 YAML 的 `atol`、`rtol` |

`thresholds` 是本次比较使用的最终阈值，提供后会替代 `atol + rtol×magnitude`。历史阈值可变时，在准备阶段返回快照，不能让本次检测或重试过程中更新的历史改写它。

方法专用参数写入 `abft.parameters`，由 `validate_config()` 检查；内置 Classic/Sum 只接受空参数字典。初期只支持 BF16 增广的方法，声明 `supported_references = {"augmented": frozenset({"bf16"})}`。

`prepare_augmented()` 和 `commit()` 是主机侧方法，也可提交 GPU 工作；这些操作不会自动融合进 GEMM。算法增加的准备和状态更新 GPU 成本都要计入完整保护调用。

改变 GPU 公式时，需要完成两处注册：

1. 在 `common/policy.cuh` 添加 CUDA policy，并加入唯一的 `Id` / `dispatch` 注册。`encode/propagate/observe/merge/finalize` 是真实设备模板接口；继续使用现有官方 MMA。
2. 在 `methods.py` 为方法类显式声明匹配的 `CompiledPolicy(ID, channels)`，同步其合法 ID/通道数检查，再调用 `register_method("方法名", 方法类或工厂)`。现有 Classic 对应 `(0,2)`，Sum 对应 `(1,1)`，可沿着它们阅读接口。

| BF16 GPU 接口 | 修改的计算 |
| --- | --- |
| `encode(mma, fragments...)` | 用已加载的 A 与校验权重计算编码；接收官方 MMA 及寄存器片段 |
| `propagate(mma, fragments...)` | 校验编码与 B 的乘法传播 |
| `observe(weight, value, accumulator)` | 输出转换前的加权校验累加 |
| `merge(accumulator, partial)` | 合并不同 M tile 的局部校验 |
| `finalize(...)` | 根据观察值、预测值和阈值产生残差及异常标记 |

可以从现有 `Projection<channels>` 继承公共实现，再覆盖所需阶段。公共注册也会实例化其他精度的适配模板；覆盖 BF16 的 MMA 重载时，须保留基类的标量重载等接口。例如两通道 policy 覆盖 `encode` 时，可用 `using Projection<2>::encode;` 保留基类重载。Python 的支持组合限制决定该方法实际开放给哪些实验。

当前 BF16 增广 MMA 片段最多容纳 8 个校验通道，已注册的 Classic/Sum 分别使用 2/1 个。增加其他通道数需要注册并编译对应 policy；超过 8 个还需要修改适配布局，不能只改配置。

最小接入方式是在 `methods.py` 内定义并注册方法。若拆到新 Python 模块，**实际运行入口必须先导入该模块**，配置中的方法名不会自动发现任意文件。native 不执行 Python `verify()`；仅改该函数不会替换 GPU 判定公式。只有校验布局或调用点确实变化时，再修改 BF16 适配文件或 [codegen/sm80.py](../../src/cutlass_gemm/codegen/sm80.py)，保留官方主数据乘法结构。

有状态方法声明 `stateful=True`；需要用残差更新历史时，再声明 `needs_diagnostics=True`。这样即使关闭报告中的 `diagnostics`，`commit()` 仍能取得内部残差与阈值。每次调用按以下顺序处理：`prepare_*` 对历史取快照 → 执行与检测 → 必要时用同一阈值和首次参考重试 → `commit()` 更新一次 → 保存检查点。调用方不再手动调用 `commit()`，也不向算法传入故障位置或注入命中真值。

用 `MethodSession.call` 持有方法实例。下面是实验驱动中的调用结构：`config` 来自实验 YAML，`samples` 提供层名、前反向角色及矩阵对；它不是一个可单独运行的命令。

```python
from abft import execute_prepared
from abft.operands import PreparedGemm
from abft.session import MethodSession

fmt = config["formats"][0]
session = MethodSession(config["abft"], config.get("fault"))
for layer_name, role, a, b in samples:
    prepared = PreparedGemm.prepare(a, b, **fmt)
    with session.call(a, b, **fmt, layer=layer_name, role=role) as (method, context):
        output, report = execute_prepared(
            prepared, abft_config=session.config, fault_config=session.fault,
            context=context, method_instance=method,
        )
```

session 放在循环外；历史按层、角色、形状等隔离。方法实现 `state_dict/load_state_dict` 保存恢复历史及 RNG，session 保存全部实例和调用计数；训练通过 `AlgorithmState` 写入 `abft_state.pt`。当前有状态 session 使用同一 CUDA stream，不支持并发调用或 CUDA Graph 重放。

**现有 `abft.benchmark` 仅用于无状态、无故障的正常调用测量。** 它不持有 `method_instance`，不能只替换成有状态方法名；其 YAML 不接收顶层 `fault`，并要求 `recovery: none`、`on_error: report`、`timing: false`。有状态或注入实验使用上面的调用结构，实验驱动另行读取相应配置；算法尚未实现时，不预设校准和历史更新规则。

## 5. 正确读取耗时

示例的 `profile_repeats: 20` 另行采集各阶段 GPU 活动，包含准备权重、融合 GEMM 和全部合并判定内核。若算法增加历史更新 GPU 工作，也必须计入。总计已包含子阶段，不要重复相加；融合内核中的主乘法和校验并没有被分开计时。

`profile_repeats: 0` 时显示 CUDA Event 包围的完整调用区间，其中可能包含 CPU 提交间隙。两种口径分别保存，不能混用。`report.to_record()`、矩阵打印和输出比较会触发回读，应放在计时之后；`diagnostics: true` 产生的 GPU 残差、阈值写入仍属于保护成本。

完整保护开销按同一口径的 `保护时间 / 无保护时间 - 1` 计算。构造输入、输入类型转换、编译、预热和打印均排除；内置不可变权重缓存测量的是预热后的稳态，有状态算法还要单独约定校准期和测量期。

## 6. 修改后检查

修改 CUDA policy 或适配代码后重新编译，再运行已有的构造用例：

```bash
python -m cutlass_gemm.backend
python -m pytest -q \
  tests/abft/test_native.py::test_native_gpu_pipeline_output_detection_and_recovery \
  tests/cutlass_gemm/test_policy_gpu.py::test_augmented_sum_uses_one_checksum_without_changing_data_rows
```

第一个测试分别检查无故障、累加器翻转和一次重算恢复；它使用可精确表示的构造输入。第二个检查 Sum 只有一个校验通道，数据行和输出矩阵保持不变。它们与随机矩阵 benchmark 的目的不同：前者检查确定行为，后者查看正常残差和完整保护开销。新增算法再补充对应的公式、阈值和故障用例。

接入训练时，精度设为 `gemm.precision: bf16`、`gemm.scaling: none`、`training.precision: bf16`，将上述保护配置放到 **`gemm.abft`** 下；独立 benchmark 使用的是顶层 **`abft`**。`qwen3_abft_debug.yaml` 当前默认 `input_fused + reference`，个人副本需要显式改成 `native + augmented + threshold_scale: output` 才与本教程相同。训练命令和续训说明见[训练教程](../training.md)。

2026-10-22 在 RTX 5090 上验证了环境检查及扩展编译加载；上述两个测试节点合计 4 项通过。文中的 session 调用结构使用现有 Classic 连续执行两次，构造输入得到全 1 输出且无报警。

Classic/Sum 的两个默认形状共 4 组 benchmark 均输出逐位一致。随机正常输入均报警 20/20，保留原调试容差；这是固定输入重复调用记录，不是完成了阈值校准。
