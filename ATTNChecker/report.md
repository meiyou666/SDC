# ATTNChecker

* 姓名: 方泽宇
* 专业: 计算机科学与技术
* 学号: 25303050218
* 实验时间：2026/8/20～2026/9/4

[TOC]

## 环境配置

1. install docker 
2. install nvidia-container-toolkit
3. link docker and nvidia-container-toolkit
4. pull docker image

## 核心算法
ABFT -> EEC-ABFT for GEMM:(0D)
对相乘矩阵分别做校验和和加权校验和扩展得到带校验和与加权校验和的扩展结果矩阵
直接相乘得到结果矩阵再算两种校验和, 类似数学层面的算两次
比较结果拓展矩阵各列/行(d1 = 普通校验和误差, d2 = 加权校验和误差), 通过误差大小判断错误类型(none, common, NaN, INF, near-INF)
none, common: common locate (d2/d1) and correction
INF, NaN, near-INF: special locate (the max item) and correction

## 算法局限
EEC-ABFT是基于ABFT算法专门争对SDC定位与处理的改进，及对于ABFT无法正确处理的错误分类讨论并处理，
局限:
处理0D时面对SDC利用行列两种校验和实现faulty-location, 面对1C/1R时c_chk/r_chk可能false_neg
(e.g d1 false), 面对2D时则束手无策
实际情况与问题:
为了减小测试开销, 不会检测0D常规出现位，延迟检测1C/1R但是面对大规模数据可能导致location and correction 
增加开销, 而延迟检测若失误导致故障传播至2Dz则算法彻底失效无法实现纠正,综上，出于降低检测开销的延迟检测有集群数据
定位纠错和故障传播彻底失效的风险


## 完整实验

1. Computing Overhead
2. Detection and Correction Rating 
3. Adaptive ABFT Detection Frequencies
4. Recovery Overhead
5. Different Batch-size Expriments
6. Custom Optimized Encoder Kernel


## 结果分析

1. Detection and Correction Rating
Both detection and correction rating are 100% in paper.
But my correction rating is 98.967% and detection rating is 92%.
论文报道的检测率与纠正率均为 100%，而本实验测得检测率 92%，纠正率 98.967%。未达满分可能源于以下因素：
(1) 定点注入（pos.txt）可能覆盖了校验行/列或对角线元素，而这些位置的错误可能被校验和抵消或误判；
(2) CUDA 浮点运算的非结合性(如原子加、Tensor Core的wmma指令)导致校验和存在截断误差，使得 d1/d2 阈值判断出现边界失误；
(3) 注入错误幅值过小，被数值噪声淹没。
后续需引入随机位置注入并统计置信区间，以验证算法的鲁棒性。(实验问题)

2. Adaptive ABFT Detection Frequencies
mode1_training overhead > mode2_training overhead
对比两种检测频率策略：mode1（全频率检测）的训练开销X，mode2（自适应降低频率）的训练开销，且 X > Y。这表明自适应检测频率能够有效降低 ABFT 带来的累计开销，与算法设计目标一致。

3. Recovery Overhead
Overhead of Checkpointing:  22.754563590047233
ATTNChecker kernel correction time:  0.5386708320002072 
ATTNChecker kernel is up to 34× overhead reduction compared with CR in paper.
ATTNChecker kernel is up to 42.24205x overhead reduction compared with CR in my experiment.
在相同故障场景下，Checkpointing (CR) 的恢复开销为 22.75 s，而 ATTNChecker 内核纠正时间仅为 0.539 s，加速比达 42.2×，优于论文报道的 34×。CR 时间偏高可能源于实现中未优化的冗余数据加载或状态重置（如 cleanRecords 未清空），而 ATTNChecker 的轻量级纠正避免了全局同步，因此开销更低。但需注意，本实验的纠正率未达 100%，部分故障可能未被计入纠正时间，该加速比应在完整纠正率下重新评估。

4. Different Batch-size Expriments
batch_size = 16                             
Attention Mechanism Overhead:  -0.08521124187890557  
当 batch_size=16 时，测得 Attention 机制的 ABFT 开销为 -0.085 s，出现负值。这并非算法降低了计算量，而是由于 GPU 异步执行、CUDA kernel 缓存预热或测量误差导致。

5. Custom Optimized Encoder Kernel                         
v4_avg < v3_avg
对比自定义优化编码器内核(v4)与基线版本(v3)的平均开销，v4_avg < v3_avg，表明针对校验和编码的专用内核能够有效减少ABFT的扩展计算开销，验证了内核级优化的必要性。

## 实验局限
检测率和纠正率实验中简单基于源码的pos.txt定点注入不具备全面性和随机性
源码中缺少总时间的记录脚本无法计算开销百分比，缺少直观量化
