# 原始与汇总结果说明

## 主实验

- `main_18k_v1/campaign.json`：硬件、软件、模型、数据、采样与模式的冻结身份。
- `main_18k_v1/selection.json`：50 个实际使用的 prompts。
- `main_18k_v1/manifests/*.json`：全部 6,000 个唯一故障规格；三个保护模式共用相同规格，所以逻辑上对应 18,000 次故障运行。
- `main_18k_v1/audit.json`、`completion.json`：完成与不变量审计。
- `main_18k_v1/summary.json`：原始流水线汇总，保留用于追溯，不作为报告主结果。
- `main_18k_author_logic/`：不修改原始运行的作者逻辑重评分；`per_cell.csv` 是报告主结果表。

GitHub 交付中没有展开约 487 MiB 的 18,000 份逐次 JSON，以避免仓库膨胀；用于精确重放的故障坐标、位位置、随机种子和模型/数据哈希均在 manifests 与 campaign 中，完整逐次结果仍由实验服务器上的 audited campaign 保留。提交的主汇总含 45 个完整实验格及校验和，不包含人工填充值。

## 补充实验

- `scaling_factor/`：各 factor 的结果、95% Wilson 区间、配对检验、审计与日志。
- `critical_layers/`：逐投影 omission 的结果、直接落在目标投影的故障子集、配对检验、审计与日志。
- `overhead/`：若微基准成功执行，包含逐次时间/显存记录和汇总；口径是同一 FT2 hook 框架内的增量。

`checksums.sha256` 只覆盖各实验生成时列入清单的核心文件；报告图可由 `code/analysis/make_report_figures.py` 从 CSV/JSON 重新生成。
