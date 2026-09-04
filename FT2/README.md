# FT2 关键层在线容错缩减复现

本目录是基于论文 *FT2: First-Token-Inspired Online Fault Tolerance on Critical Layers for Generative Large Language Models*、作者公开仓库和三次 RTX 4090 实验整理的可审计交付物。

- `report.md`：实验报告正文。
- `code/`：作者代码摘录、复现实现、实验入口、评分与绘图脚本。
- `results/figures/`：由表格数据可重复生成的图。
- `results/raw/`：冻结协议、故障清单、汇总表、审计、校验和与日志。

主结果以作者意图的二元判定重新评分：归一化后的参考答案 token recall 等于 1 视为任务正确；所有 18,000 次已完成故障运行都进入分母，不再附加 clean-evaluability gate。

## 最短复核路径

1. 阅读 `report.md` 的“结论摘要”和“结果分析”。
2. 核对 `results/raw/main_18k_author_logic/per_cell.csv`、`results/raw/scaling_factor/per_factor.csv`、`results/raw/critical_layers/per_mode.csv`。
3. 运行 `python code/analysis/make_report_figures.py` 重建所有结果图。
4. 若具备相同模型/数据缓存和 CUDA 环境，按 `code/README.md` 中的命令重跑实验。

论文 DOI：<https://doi.org/10.1145/3731545.3731570>  
作者仓库快照：<https://github.com/pipijing13/FT2-LLM-inference-protection/tree/90510aec5d26850739cf583c97c65cbd71256cb1>
