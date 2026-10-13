# 算法接入

已实现 `none` 和 `classic`。`none` 使用相同量化与 GEMM 内核，跳过校验。`classic` 包含全 1 与递增加权校验、检测、单输出元素定位、修正及复检。

论文方法在各自目录实现，公共 GEMM、注入和统计不变：

| 目录 | 机制 |
| --- | --- |
| `classic/` | 确定性编码与单元素纠正 |
| `vabft/` | 残差建模与自适应阈值 |
| `gvfa/` | 高斯随机投影 |
| `strided/` | 分组编码和局部归约 |
| `aegis/` | 累加结果取数与历史残差 |

复制 [template.py](template.py)，实现 `prepare` 与 `verify`，再注册：

```python
from algorithms import register
register("my_method", "nvidia", MyDetector)
```

`prepare` 在故障注入前执行。输入为 `PackedOperand`，含码值、scale、全局 scale、逻辑尺寸与补齐尺寸；第二个操作数按 `[N,K]` 存储。参考校验直接读取 Q+S。量化、故障和投影各用独立种子。

`verify` 返回设备上的 `status、residual、threshold`。状态依次为：检测标志、定位标志、行、列、修正标志、复检失败标志。`correct=False` 不修改输出。计算和状态更新都计入完整保护时间。

有状态方法继承 `torch.nn.Module`，通过 `register_buffer` 保存投影和历史统计。训练适配器持有这个实例；独立评测可用 `gemm(..., detector=instance)` 传入同一实例。配对故障从相同的 `state_dict` 开始。论文公式、实现位置和小规模结果写在 `docs/reproduction/`。

需要融合校验时，在 `src/backends/nvidia/` 添加对应 CUDA 实现，保留公共的主计算类型、精度、位翻转位置和输出契约。先比较相同执行路径下的机制，再单独比较融合收益。
