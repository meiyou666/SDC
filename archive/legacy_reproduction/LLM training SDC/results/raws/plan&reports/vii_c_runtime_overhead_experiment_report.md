# VII-C Runtime Overhead 复现实验报告

## 1. 实验定位

本报告记录论文 VII-C runtime overhead 的本地复现结果。VII-C 的目标不是评估 fault injection 或 recompute 效果，而是在 **fault-free** 条件下比较 detection-on 与 detection-off 两条训练代码路径的执行速度。

论文报告 detector 的训练开销约为 1%。本轮本地实验采用 60M、seed 42、batch 512 的同配置样本，比较 step 1-2000 的 per-update 时间，并过滤前 100 step warmup。

## 2. 实验设计

两组样本如下：

| 条件 | 入口 | 数据来源 | Fault injection | 说明 |
|---|---|---|---|---|
| detection-on | `torchrun_main.py` | 复用 VII-B fault-free baseline 首段 | disabled | 有 Rt/detector 代码路径 |
| detection-off | `torchrun_main_no_detection.py` | 新增匹配 2,000-step run | disabled | 移除 detector 代码路径 |

共同训练配置如下：

| 项目 | 配置 |
|---|---|
| Model | `llama_60m` |
| Seed | 42 |
| Compared updates | 1-2000 |
| Warmup filtered | first 100 updates |
| Sampled deltas | 1900 per run |
| Eval frequency | every 1,000 updates |
| Max sequence length | 256 |
| Micro batch size | 256 |
| Effective batch size | 512 |
| Optimizer | AdamW |
| Max LR | 1e-3 |
| Warmup schedule | 1,000 steps |
| Precision | bfloat16 |
| Gradient clipping | global norm 1.0 |
| Dataset repeat | 1 |
| Attention metrics | enabled |

计时口径来自 `metrics.jsonl` 中带有 `loss` 和 `_time` 的 core training row。对相邻 step 的 `_time` 做差，保留 step 101-2000 的正有限 delta，取 median s/it。Overhead 公式为：

```text
overhead_pct = (median_s_per_it_detection_on - median_s_per_it_detection_off)
               / median_s_per_it_detection_off * 100
```

## 3. 数据来源与完整性

数据路径如下：

| 类型 | 路径 |
|---|---|
| Detection-on reused run | `checkpoints/vii_b_mb256_3906/baseline/seed_42` |
| Detection-off run | `checkpoints/vii_c_mb256/no_detection_match_vii_b/seed_42` |
| Summary CSV | `logs/vii_c_mb256/overhead_summary.csv` |
| Detection-off log | `logs/vii_c_mb256/no_detection_match_vii_b_seed_42.log` |

完整性检查：

| 条件 | Core steps present | Step range | Sampled deltas | Eval @1000 | Eval @2000 | Final checkpoint |
|---|---:|---|---:|---:|---:|---|
| detection-on | 2000 | 1-2000 | 1900 | 4.317315102 | 3.901448965 | reused source has 3906-step checkpoint |
| detection-off | 2000 | 1-2000 | 1900 | 4.317315102 | 3.901448965 | yes, `model_2000` |

两组 eval loss 在 step 1000 和 step 2000 完全一致，说明 detection-off run 与复用的 detection-on baseline 在训练数据、seed、batch、LR schedule 和模型配置上对齐。Detection-off 的 `training_state.json` 显示 `update_step=2000`、`global_step=2000`、`batch_idx=4000`，run 完整结束。

## 4. Runtime 结果

`overhead_summary.csv` 的主结果如下：

| 条件 | Median s/it | Steps sampled | Step range | Warmup filtered |
|---|---:|---:|---|---:|
| detection-on reused VII-B | 1.642826080 | 1900 | 1-2000 | 100 |
| detection-off | 1.674168587 | 1900 | 1-2000 | 100 |
| overhead | -1.872123671% | - | - | - |

补充统计如下：

| 条件 | Mean s/it | P95 s/it |
|---|---:|---:|
| detection-on reused VII-B | 1.673503682 | 1.762562275 |
| detection-off | 1.705237464 | 1.805291891 |

Median、mean 和 p95 三个统计都显示 detection-on 样本略快于 detection-off 样本。按脚本定义的 median 口径，本轮 overhead 为 **-1.87%**，即未观察到 detection-on 相比 detection-off 的正向 slowdown。

## 5. 结果解释

本轮数据不能支持“本机实测 detector 开销为 +1%”这个更强表述；它支持的是：在 60M、batch 512、2,000-step、单 seed 的本地样本中，detector 路径没有产生可测量的正向开销，观测差异约为 -1.9%。

这个负 overhead 不应解释为 detector 能加速训练。更合理的解释是：检测代码路径的真实开销很小，已经低于单次非同时运行带来的环境抖动、GPU boost 状态、I/O/eval 附带扰动和系统负载差异。由于 detection-on 样本复用 VII-B 已保存首段，而 detection-off 是之后单独新增运行，两者不是同一时间窗口内 back-to-back 交替测量；当目标差异只有约 1% 时，这种运行间噪声足以改变符号。

尽管如此，本轮设计仍然比直接复用 VII-B 10,000-step 总时间更严谨：它只比较 fault-free 的 detection-on/off 两条路径，只取相同步数范围，只使用 core training row 的 step-to-step `_time` 差分，并过滤前 100 step warmup。它避免了 VII-B 中 fault injection、recompute、checkpoint、不同 run 长度和数据 repeat 后段对 runtime 的混杂影响。

## 6. 结论

本地 VII-C 结果如下：

1. Detection-on 与 detection-off 两组在 step 1-2000 的 eval loss 完全一致，配置和数据顺序对齐。
2. 按 step 101-2000 median s/it 计算，detection-on 为 1.642826 s/it，detection-off 为 1.674169 s/it，overhead 为 -1.87%。
3. 本轮没有复现论文中约 +1% 的正开销，但结果说明 detector 路径的开销在本地设置下处于运行噪声量级，没有观察到明显 slowdown。
4. 若需要严格验证 1% 量级，应在云服务器上做 back-to-back 多轮交替测量，例如 on/off/on/off 各 3 次，固定 GPU clocks，并关闭不必要的外部负载。

