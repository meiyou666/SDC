# 论文检测算法判定逻辑梳理与源码对照

本文档梳理论文 Section VI "Detection" 的检测算法判定逻辑，并逐项对照本仓库 `torchrun_main.py` 中的实现位置。所有行号以当前版本（2026-09）为准。

## 1. 论文算法的判定逻辑

### 1.1 检测统计量 R_t 的定义（论文 Eq. 2–8）

论文基于 AdamW 的参数更新公式（Eq. 2–6），取其中的自适应分量（weight decay 与梯度无关，不受 corruption 影响，故忽略）：

```
U_{t,p} = lr * m_{t,p} / (sqrt(v_{t,p}) + eps)        （论文 Eq. 7，省略 bias correction）
```

其中 m、v 是梯度一阶/二阶指数滑动平均。论文明确说明省略 bias correction 在训练若干步后是成立的。

R_t 定义为所有 parameter group 中"均方根更新幅度"的最大值：

```
R_t = max over groups P of sqrt( (1/|P|) * sum_{p in P} U_{t,p}^2 )   （论文 Eq. 8）
```

即先对每个参数组内求 U 的 RMS（衡量该组平均更新幅度），再取所有组中的最大值。R_t 是"异常大的参数更新"的指示器。

### 1.2 跳变量与 warm-up 基线（论文 Eq. 9–11）

```
ΔR_t = R_t - R_{t-1}                                   （论文 Eq. 9）
μ̄   = (1/T) * sum_{i=1}^{T} ΔR_i,   T = min(t, t_w)    （论文 Eq. 10，t_w 为 warm-up 长度）
```

用 warm-up 阶段观察到的 ΔR 的经验均值作为基线。初始的朴素判定规则：

```
异常 ⇔ ΔR_t > μ̄ / α                                    （论文 Eq. 11）
```

α ∈ (0,1] 是可配置的敏感度参数：α 越大阈值越低、检测越灵敏；α 越小越保守。

### 1.3 辅助信号：梯度范数跳变（论文 Eq. 12–13）

仅靠 ΔR_t 在敏感度/特异度之间有 tradeoff，因为正常训练动态也会产生跳变。论文引入全局梯度范数（clipping 前）的正跳变作为辅助信号：

```
G_t = max(0, G_t^pre - G_{t-1}^pre)                    （论文 Eq. 12）
```

最终判定规则：

```
异常 ⇔ ΔR_t > μ̄ / (α * sqrt(G_t))                      （论文 Eq. 13）
```

含义：G_t 大（伴随梯度尖峰）时检测器对 ΔR_t 更敏感；G_t 小时降低响应以避免误报。除以 sqrt(G_t) 是因为 R_t 是 RMS 量，随底层更新幅度开方缩放。

### 1.4 非有限梯度范数的硬判定（论文 Section VI 末段）

```
若 G_t^pre 非有限（inf / NaN）⇒ 直接判为异常
```

理由（论文 Section IV）：梯度范数为 inf 时 clipping 会把梯度缩成 0（该 mini-batch 实际被丢弃），NaN 会直接中断训练，两者都是失效状态。

### 1.5 检测后的 recompute 验证与缓解（论文 Section VI 末段）

检测到异常后，重算当前 training step 以验证检测并缓解影响：

- 健康条件下重算应得到相同的 R_t（相同数据 + 相同模型状态 ⇒ 相同输出）；
- 若原计算被 corruption 污染，由于硬件调度非确定性，重算的 R_t 几乎必然不同；
- recompute 时不再注入故障。

论文还讨论了分布式兼容性：指标可在每个 worker 本地从梯度和优化器状态计算（梯度传播前检测并定位故障设备），也可在梯度聚合后计算（各 worker 判定一致，避免独立误报）。

## 2. 源码对照（torchrun_main.py）

### 2.1 R_t 计算 —— 行 564–591

```python
rt = torch.tensor(0.0, device=device)
for group in optimizer.param_groups:
    lr = group["lr"]
    beta1, beta2 = group["betas"]
    for p in group["params"]:
        ...
        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * (g * g)
        s_new = lr * m_new / (torch.sqrt(v_new) + 1e-8)
        layer_rms = torch.sqrt((s_new * s_new).mean())
        rt = torch.maximum(rt, layer_rms)
```

对应 Eq. 7–8。注意：**不调用 optimizer.step() 来拿更新量，而是手工用 betas 从 exp_avg/exp_avg_sq 推下一步的 m/v**，与论文"省略 bias correction 的更新量"口径一致；eps 取 1e-8。

### 2.2 ΔR_t 计算 —— 行 492、593–596

```python
rt_prev = rt                      # 行 492，step 开始时先保存上一步的 rt
...
if global_step < 2:
    rt_jump = torch.tensor(0.0, device=device)
else:
    rt_jump = torch.abs(rt - rt_prev)   # 行 596
```

对应 Eq. 9。**实现用的是绝对值**，论文是带符号差分（见第 3 节偏差说明）。

### 2.3 warm-up 基线 μ̄（window_mean）—— 行 376、496、619–620

```python
rt_jump_history = torch.zeros(args.warmup_steps, device=device)   # 行 376
...
if global_step < args.warmup_steps:
    rt_jump_history[global_step] = rt_jump.detach()               # 行 496
...
if global_step < args.warmup_steps:                               # 行 619
    window_mean = rt_jump_history[:global_step + 1].mean().detach()  # 行 620
```

对应 Eq. 10：warm-up 期内用截至当前 step 的运行均值，T = min(t, t_w)。**warm-up 结束后 window_mean 不再更新**，冻结为 warm-up 末期值，作为后续全部训练的基线。

细节：行 496 在更新 rt_jump 之前执行，因此 history[t] 实际存入的是 step t-1 的 ΔR，window_mean 是"截至上一步的 ΔR 均值"，与论文口径有一个 step 的索引偏移，无实质影响。

### 2.4 主判定逻辑 —— 行 603–631

```python
anomaly = False
if not torch.isfinite(gradient_norm_pre):        # 行 606：论文"非有限梯度范数硬判定"
    anomaly = True
else:
    if global_step >= 10:                        # 行 610：前 10 步不检测
        gradient_norm_jump = gradient_norm_pre - gradient_norm_pre_prev
        gradient_norm_jump = torch.clamp_min(gradient_norm_jump, 0)   # 行 612：Eq. 12
        if gradient_norm_pre_prev.eq(0):         # 行 614：上一步范数为 0 的守卫
            anomaly = False
        elif gradient_norm_jump.eq(0):           # 行 616：无梯度跳变 ⇒ 不判异常
            anomaly = False
        else:
            if global_step < args.warmup_steps:
                window_mean = rt_jump_history[:global_step + 1].mean().detach()
            gradient_norm_jump_sqrt = torch.sqrt(gradient_norm_jump)
            threshold = window_mean / (args.fi_nvbit_alpha * gradient_norm_jump_sqrt)  # 行 623：Eq. 13
            anomaly = rt_jump > threshold        # 行 624
```

`gradient_norm_pre` 在行 546 于 clipping 之前计算（`compute_gradient_norm`），对应论文 G_t^pre。

判定真值表：

| 条件 | 判定 |
|---|---|
| G_t^pre 非有限 | 异常（跳过后续所有条件） |
| global_step < 10 | 不检测 |
| G_{t-1}^pre = 0 | 不检测 |
| G_t = 0（无正跳变） | 不检测 |
| 否则 | ΔR_t > μ̄ / (α·√G_t) 则异常 |

α 来自命令行 `--fi_nvbit_alpha`（行 86），默认 0.05，无范围校验。

### 2.5 recompute 验证与缓解 —— 行 498、634–668

检测与 recompute 包在 `while do_recompute:` 循环（行 498）内：

```python
if args.fi_nvbit_recompute and anomaly:   # 行 659
    trigger_nvbit = False                 # 行 660：recompute 期间关闭注入
    do_recompute = True
    rt_jump_prev = rt_jump.detach()       # 行 662：记录原 R_t 供对照
    optimizer.zero_grad()                 # 行 663：清梯度后重算当前 step
```

重算一轮后（行 634–643）：若仍判异常 → `count_incorrect_recompute`，日志 "Anomaly persists after recomputation"；否则 → `count_correct_recompute`，"Resolved after recomputation"。与论文"重算验证检测"的口径一致（论文用重算前后 R_t 是否一致，代码用重算后是否仍超阈值，等价）。

### 2.6 混淆矩阵统计 —— 行 381–388、644–655

以 `trigger_nvbit`（本 step 是否处于注入窗口）为真值标签：

```python
if anomaly:
    confusion_count['tp' if trigger_nvbit else 'fp'] += 1
else:
    confusion_count['fn' if trigger_nvbit else 'tn'] += 1
```

对应论文 Section VII-A 的 detection rate / recompute precision 统计。

### 2.7 日志输出

- 检测事件：`logger.warning(f"[AD] Detected anomaly {global_step}: {warning}")`（行 646），warning 内含 rt_jump、window_mean、α、√G_t 的数值（行 627–631）。
- 逐 step 指标写入 metrics.jsonl（行 759–775）：`gradient_norm_pre/post`、`gradient_norm_jump`、`rt/rt`、`rt/rt_jump`、`detected_anomaly`、`trigger_nvbit`，供事后画论文 Fig. 3/4 及 TP/FP/FN/TN 分析。

## 3. 实现与论文的口径差异（写报告时需注意）

1. **ΔR_t 取绝对值**（行 596）：论文 Eq. 9 是带符号差分 R_t − R_{t−1}，负跳变（R_t 回落）在论文口径下会降低判定值，本地实现把回落也当作跳变参与判定。对"尖峰检测"目的而言本地口径更保守。
2. **max 的粒度是单个参数张量而非参数组**（行 567–591）：论文 Eq. 8 是"每组内求 RMS 再对组取 max"，本地对 optimizer.param_groups 内每个参数张量单独求 RMS 后取全局 max。因为优化器按张量建 state，粒度更细，数值上会 ≥ 论文口径。
3. **G_t = 0 时不检测**（行 616–617）：论文 Eq. 13 在 G_t = 0 时阈值无穷大、数学上等价不检测，代码显式短路，口径一致；但这也意味着**检测器只在梯度范数有正跳变的 step 才可能触发**，纯 R_t 异常而梯度平稳的事件会被漏掉。
4. **warm-up 后基线冻结**：window_mean 在 warmup_steps 之后不再更新。若训练后期正常更新幅度整体漂移，基线会过期——论文 Eq. 10 的 T = min(t, t_w) 表述同样意味着只用 warm-up 段，口径一致。
5. **前 10 步与 G_{t−1}=0 守卫**是论文没有的工程保护。
6. **bias correction**：两边都省略；本地额外注意 m_new/v_new 是"下一步"的矩估计（用当前 g 再推一步），不是 optimizer 内部已 bias-corrected 的 m̂/v̂。
7. 非有限判定用的是 clipping 前的 `gradient_norm_pre`（行 606），与论文一致；这与论文 Section IV 的"inf 梯度范数使 clipped 范数塌缩为 0"的分析链条对应。

## 4. 相关文件

| 内容 | 位置 |
|---|---|
| 检测主逻辑 | `torchrun_main.py:603-668` |
| R_t 计算 | `torchrun_main.py:564-591` |
| warm-up 历史 | `torchrun_main.py:376, 496, 619-620` |
| recompute 循环 | `torchrun_main.py:498-668` |
| α 命令行参数 | `torchrun_main.py:86`（`--fi_nvbit_alpha`，默认 0.05） |
| 梯度范数（pre/post clipping） | `torchrun_main.py:546, 561-562` |
| 混淆计数与日志 | `torchrun_main.py:381-388, 644-655` |
| metrics 输出 | `torchrun_main.py:759-775` |
| 无检测基线对照 | `torchrun_main_no_detection.py`（无 anomaly detection 路径，用于 overhead 对比） |
