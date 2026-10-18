# AEGIS：ABFT 公式与代码复现

## 1.核心创新点

- 架构上：采用两阶段 cSensor-cVerifier

  1. cSensor：与训练同步运行，执行轻量级检测，将检出的可能的错误打包进vTask，便于后续确认
  2. cVerifier：调度到空闲时间段重新执行高度可疑的计算

- 机制上：

  - 混合精度的算法检测

    - FlashAttention 不变量
    - Matmul 校验和**（也即ABFT优化）**

    注：原文为减小开销，采用动态采用策略，而复现时为保证后续各指标分析公平，采样率取100%

  - 自等价性检测（利用自身冗余）

重点关注部分为 Matmul 的行校验、行列校验、FP32 累加器取数接口和历史阈值更新，其他架构/机制不复现。

## 2. ABFT相关公式

### 行校验与行列校验

令有效输入为

$$
A\in\mathbb{R}^{M\times K},\quad B\in\mathbb{R}^{K\times N},\quad C=AB.
$$

“有效输入”指量化码值及其尺度确定的逻辑值。校验应使用实际参与 GEMM 的值，其中fwd、dgrad  采用行校验，wgrad采用行列校验。

记 $\mathbf{1}_N$ 为长度 $N$ 的全 1 列向量。行校验为

$$
s_r^{pred}=A(B\mathbf{1}_N),\qquad s_r^{obs}=C\mathbf{1}_N,\qquad
r_r=s_r^{obs}-s_r^{pred}. \tag{1}
$$

同理，列校验的形式为

$$
s_c^{pred}=(\mathbf{1}_M^T A)B,\qquad
s_c^{obs}=\mathbf{1}_M^TC,\qquad r_c=s_c^{obs}-s_c^{pred}. \tag{2}
$$

用最大绝对残差作为调用级统计量：

$$
\Delta_r=\max_i |r_{r,i}|,\qquad
\Delta_{rc}=\max(\max_i |r_{r,i}|,\max_j |r_{c,j}|). \tag{3}
$$

### FP32 累加器取数

![低精度输出与 FP32 累加器校验取数对比](../images/aegis-accumulator.png)

经典 ABFT 从 GEMM 的最终低精度输出取观测校验和，每个输出元素在写出时被截断一次（bf16 尾数仅 7 位，截断误差约 2⁻⁸），求和后这些噪声进入残差并可能淹没故障信号。而矩阵单元执行 GEMM 本是"低精度乘法 + FP32 片上累加"：累加器 C₀ 全程驻留 FP32 寄存器沿 K 维分块乘加，只在写出内存前才截断为低精度。

AEGIS 据此把取数点从"截断后的输出"移到"截断前的累加器"，观测校验和直接对 C₀ 求行和，校验链路精度对应地统一为 FP32、差分精度取 FP64，残差中的输出截断噪声项被消除而故障响应不变，信噪比提升约 4 个数量级。

### 自适应阈值

本次先从独立正常校准集得到统计量 $\Delta_1,\ldots,\Delta_n$，得到初始阈值：

$$
\widehat E_0=\max\{\epsilon,\Delta_{(\lceil(1-\alpha)n\rceil)}\}. \tag{7}
$$

其中 $\epsilon>0$ 是配置的数值下限，防止零阈值在乘法更新下始终为零（文中未明确给出数值）。分位数只定义经验校准方法，不保证测试集误报率必然低于 $\alpha$。

令当前阈值为 $\widehat E_t$、连续静默计数为 $q_t$，先按旧阈值判定

$$
a_t=[\Delta_t>\widehat E_t]. \tag{8}
$$

自适应更新采用

$$
(\widehat E_{t+1},q_{t+1})=
\begin{cases}
(\gamma\widehat E_t,0),& a_t=1,\\
(\max(\widehat E_t/\gamma,\widehat E_0),0),& a_t=0,\ q_t+1=W,\\
(\widehat E_t,q_t+1),& \text{其余情况}.
\end{cases} \tag{9}
$$

文中读图可取 $\gamma=2,W=50$。NaN/Inf 信号直接标为可疑，不更新有限历史阈值，这是代码中增加的异常处理。

## 3.实现代码

1. aegis.py (src/algorithms/aegis/aegis.py)：实现 ABFT 检测机制

主要函数：

- prepare(A, B)：在故障注入前生成健康参考校验。行校验为 $A(B\mathbf{1})$，开启列校验时还计算 $(\mathbf{1}^{T}A)B$。
- sense(checksums, observed)：对传入结果进行 FP32 求和，与参考校验相减，得到残差、最大绝对残差及其位置。
- calibrate()：根据正常样本的残差分位数确定初始阈值。这是复现选择。
- Threshold.update()：判断是否超阈值；自适应模式下，报警后上调阈值，连续静默后向基线衰减。也支持保存和恢复状态。

这个文件不执行主矩阵乘法，需要调用方提供被检查的计算结果。

2. reference.py (src/algorithms/aegis/reference.py)：提供 CPU 数学模拟和故障注入

主要函数：

- gemm()：按固定 K 分块计算，每块用 FP64 相乘、转为 FP32 后累加，最后转换为指定输出精度；同时返回 FP32 累加结果和最终输出。
- flip()：对 A、B 或最终累加结果的指定元素执行单比特翻转，也支持零掩码对照。
- bitwise_equal()：逐字节比较故障输出与健康输出，判断输出是否受损。

它用于小规模机制验证，不是实际 Tensor Core/CUTLASS 的计算实现。

3. main.py (src/algorithms/aegis/__main__.py)：组织完整对比实验

它读取 YAML 配置，然后依次：

1. 生成正常操作数，采集校准残差。
2. 初始化三组方法：输出取数＋固定阈值、累加器取数＋固定阈值、累加器取数＋自适应阈值。
3. 对同一组输入执行正常、零掩码和逐位故障实验。
4. 调用 prepare → gemm → sense → Threshold.update，记录检测结果。
5. 汇总误报率、受损事件检出率等，保存配置和逐事件结果到 run.json。
