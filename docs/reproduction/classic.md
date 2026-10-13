# 经典 ABFT 实现

采用全 1 向量和递增加权向量，分别生成两侧校验。设全 1 向量为 $u_M,u_N$，`w=(1,…,M)`，`v=(1,…,N)`：

\[
c^{pred}=\big[(u_M^TA)B,\;(w^TA)B,\;A(Bu_N),\;A(Bv)\big].
\]

对输出计算相同四组归约，残差为 `observed − predicted`。阈值采用 `atol + rtol × max(|observed|, |predicted|)`。任何残差超阈值或非有限均报警。

单输出元素错误应同时对应唯一异常行和列。普通残差给出扰动幅值，加权残差检验其坐标。通过一致性检验后，减去两侧普通残差的均值，再完整复检。

| 内容 | 实现与测试 |
| --- | --- |
| 独立 FP64 公式 | `src/algorithms/classic/reference.py` |
| 码值及 scale 读取、FP32 校验 | `src/backends/nvidia/kernels.cu` |
| 单元素定位、修正与复检 | `locate_correct`、CUDA `decide` 与 `postcheck` |
| 单点位置遍历、位翻转与传播 | `tests/gemm/test_contract.py` |
| 原生计算及零掩码一致性 | `tests/gemm/test_cuda.py` |

量化格式、阈值与执行组织属于本工作台配置。其余论文方法按各自公式单独实现，不用经典 ABFT 替代。基础文献：[Huang and Abraham, 1984](https://doi.org/10.1109/TC.1984.1676475)。
