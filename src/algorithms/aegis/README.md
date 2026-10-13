# AEGIS 校验

实现 FP32 累加结果检查与历史残差阈值：[论文](https://www.usenix.org/conference/osdi26/presentation/lei)。

保留检查位置与状态更新规则。历史统计注册为 buffer；配对实验恢复相同初始状态，记录在线更新成本。
