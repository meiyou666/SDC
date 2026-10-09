# 代码说明与重跑入口

## 目录

- `upstream_reference/`：作者仓库 commit `90510aec…` 中与 OPT/Qwen 实验直接相关的公开脚本和模型实现摘录，仅作逻辑核对。
- `reproduction/src/ft2_formal/`：本次正式、可审计的故障注入与 FT2 引擎。
- `reproduction/run_main_18k.py`：18,000 次主实验入口。
- `reproduction/experiments/run_scaling_ablation_v2.py`：scaling factor 实验。
- `reproduction/experiments/run_critical_layer_identification.py`：关键层 omission 实验。
- `reproduction/analysis/reanalyze_author_logic.py`：按作者判定逻辑重评分。
- `experiments/run_overhead_benchmark.py`：补充的 RTX 4090 成对时间/显存基准。
- `analysis/make_report_figures.py`：从 CSV/JSON 重建报告图。

自行新增或重构部分的动机和与作者实现的对应关系见 `MODIFICATIONS.md`。正式实现以 `ft2_formal` 为准；`ft2_repro` 是早期验证代码，保留用于溯源，不应与正式 API 混用。

## 主实验环境

主 campaign 固定并记录：Python 3.10.21、PyTorch 2.3.0+cu118、CUDA toolkit 11.8、Transformers 4.42.4、Datasets 2.15.0、NumPy 1.26.2、RTX 4090、driver 550.142。模型和数据版本见 `../results/raw/main_18k_v1/campaign.json`。

## 典型命令

以下命令应从作者仓库根目录执行；先把本目录的 `reproduction/` 放到该仓库根目录，并确保冻结模型及数据缓存路径与配置一致。

```bash
python reproduction/run_main_18k.py
python reproduction/analysis/reanalyze_author_logic.py \
  --results-dir reproduction/results/main_18k_v1 \
  --output-dir reproduction/results/main_18k_v1/reanalysis_author_logic_v2
python reproduction/experiments/run_scaling_ablation_v2.py
python reproduction/experiments/run_critical_layer_identification.py
```

开销基准比较同一 hook 框架内 `no_protection` 与 first-token factor 2.0 的增量，不是原生 Transformers 无 hook 基线：

```bash
python code/experiments/run_overhead_benchmark.py \
  --main-root reproduction/results/main_18k_v1 \
  --output-dir results/raw/overhead \
  --repeats 2
```

重建图表：

```bash
python code/analysis/make_report_figures.py
```

所有正式实验都支持已有 artifact 的审计式续跑；不要删除 lock、protocol、manifest 或 completion 文件后混跑不同配置。
