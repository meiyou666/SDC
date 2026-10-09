# llm-sdc-training-qr

Offline reproduction code for *Exploring Silent Data Corruption as a Reliability Challenge in LLM Training*. The project studies how GPU-level silent data corruption (SDC), injected into matrix-multiply instructions with NVBit, affects LLaMA-style pretraining and how a lightweight runtime detector plus recomputation can mitigate harmful updates.

This repository is adapted from the paper's GaLore-based training code for an offline CUDA 13.0 environment. See [MODIFICATIONS.md](MODIFICATIONS.md) for the full list of changes from the original code.

## Features

- LLaMA-style causal language model training from scratch on local C4-formatted data.
- Runtime anomaly detection based on Adam update statistics (`R_t`) and gradient-norm jumps.
- Optional recompute-on-detection mitigation.
- NVBit-based register bit-flip fault injection for CUDA GEMM/HMMA kernels.
- Attention-score introspection for corruption signature analysis.
- Offline dataset, tokenizer, and metrics logging paths for air-gapped runs.

## Repository Layout

```text
.
├── configs/                      # LLaMA model configs used by experiments
├── metrics/                      # Gradient, weight, and parameter-difference helpers
├── nvbit/
│   ├── 1.7.6_nvbit_release/      # Expected location for the NVBit 1.7.6 release
│   │   └── tools/fault_injection/# Fault-injection tool source
│   └── README_NVBit_1.7.6.md     # NVBit adaptation notes
├── scripts/
│   ├── base/                     # Baseline training entry points
│   ├── configs/                  # Shared experiment configuration
│   ├── experiments/              # Paper-section reproduction campaigns
│   └── fi/                       # Fault-injection entry points
├── training/                     # Dataloader, model, scheduler, and local logger code
├── MODIFICATIONS.md              # Changes from the original paper code
├── requirements.txt
├── torchrun_main.py              # Trainer with anomaly detection
└── torchrun_main_no_detection.py # Trainer without anomaly detection
```

Generated experiment outputs are written under `checkpoints/` and `logs/`.

## Requirements

The tested reproduction environment is:

- Linux host or Linux container for training and NVBit execution.
- Python 3.12.3.
- CUDA 13.0-capable PyTorch environment.
- NVIDIA GPU with bfloat16 support. Local experiments used an RTX 4090; the paper used an L40S.
- NVBit 1.7.6 installed under `nvbit/1.7.6_nvbit_release/`.
- `nvdisasm` available in `PATH` when building or running NVBit tools.

Python dependencies are pinned in [requirements.txt](requirements.txt):

```bash
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Offline Assets

This repository does not require network access at training time, but the required assets must already be present locally.

Place local C4-style JSONL files at:

```text
data/c4_subset_2M.json.gz
data/c4_validation.jsonl
```

Each line should be a JSON object with at least a `text` field. The training file may be plain JSONL or gzip-compressed JSONL.

Place a downloaded T5 tokenizer at:

```text
tokenizer/t5-base/
├── tokenizer.json
├── tokenizer_config.json
├── spiece.model
└── special_tokens_map.json
```

The training scripts set `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`, and the trainer loads the tokenizer with `local_files_only=True`.

## NVBit Setup

Place the official NVBit 1.7.6 release contents under:

```text
nvbit/1.7.6_nvbit_release/
```

The checked-in `tools/fault_injection/` directory contains the adapted fault-injection tool source. The NVBit `core/` directory and `libnvbit.a` must come from the official release.

Build the fault injector:

```bash
cd nvbit/1.7.6_nvbit_release/tools/fault_injection
make clean && make
cd ../../../..
```

If the default architecture does not match your GPU, override it:

```bash
make ARCH=sm_89
```

Fault injection is enabled through `LD_PRELOAD` and controlled by `SIGUSR1`/`SIGUSR2` from the Python trainer. The default NVBit environment is defined in [scripts/configs/nvbit_default.sh](scripts/configs/nvbit_default.sh).

## Quick Start

Edit [scripts/configs/common.sh](scripts/configs/common.sh) first:

```bash
export SEED=351344
export MODEL=llama_60m
export BATCH_SIZE=256
export LR=1e-3
export TOKENIZER_PATH="./tokenizer/t5-base"
export TRAIN_DATA_PATH="./data/c4_subset_2M.json.gz"
export VAL_DATA_PATH="./data/c4_validation.jsonl"
```

Run a baseline with anomaly detection:

```bash
./scripts/base/train_baseline.sh
```

Run the matching no-detection baseline for overhead comparison:

```bash
./scripts/base/train_baseline_no_detection.sh
```

Run fault injection without recompute:

```bash
./scripts/fi/train_fi.sh
```

Run fault injection with recompute-on-detection:

```bash
./scripts/fi/train_fi_recompute.sh
```

## Main Training Entrypoints

Use `torchrun_main.py` for the full training path with detection:

```bash
python torchrun_main.py \
  --single_gpu \
  --model_config configs/llama_60m.json \
  --tokenizer_path ./tokenizer/t5-base \
  --train_data_path ./data/c4_subset_2M.json.gz \
  --val_data_path ./data/c4_validation.jsonl \
  --batch_size 256 \
  --total_batch_size 512 \
  --num_training_steps 100000 \
  --exit_after 1000 \
  --dtype bfloat16 \
  --optimizer adamw \
  --grad_clipping 1.0 \
  --save_dir checkpoints/example_detection
```

Use `torchrun_main_no_detection.py` when measuring detector overhead.

Key fault-injection flags:

| Flag | Meaning |
|---|---|
| `--fi_nvbit_enable` | Enables trainer-side NVBit trigger logic. |
| `--fi_nvbit_location` | Injection window: `forward`, `backward`, `forward_backward`, `clipping`, or `optimizer`. |
| `--fi_nvbit_trigger_rate` | Average trigger interval in update steps. |
| `--fi_nvbit_steps` | Explicit trigger steps; use `-1` to disable explicit steps. |
| `--fi_nvbit_duration` | Number of consecutive update steps per injection event. |
| `--fi_nvbit_duration_random` | Randomize duration from 1 to `--fi_nvbit_duration`. |
| `--fi_nvbit_target_funcs` | Candidate NVBit function IDs; use `-1` for all matching functions. |
| `--fi_nvbit_recompute` | Recompute the step after a detected anomaly. |
| `--fi_nvbit_alpha` | Detector threshold parameter. |

## Reproducing Paper Sections

The `scripts/experiments/` directory contains higher-level campaigns and summarizers used for the local reproduction reports:

| Script | Purpose |
|---|---|
| `profile_iv_a_kernels.sh` | Profile GEMM/HMMA kernels for IV-A style campaigns. |
| `run_iv_a_bit_sensitivity.sh` | Bit sensitivity experiments. |
| `run_iv_b_kernel_sensitivity.sh` | Kernel sensitivity experiments. |
| `run_iv_e_spatial_sweep.sh` | SM/lane spatial sensitivity sweeps. |
| `run_iv_f_rate_all_hmma.sh` | Injection-rate experiments. |
| `run_iv_f_duration_all_hmma.sh` | Injection-duration experiments. |
| `run_vii_a_detection.sh` | Detection/recompute campaign. |
| `run_vii_b_scale_60m.sh` | 60M-scale recovery comparison. |
| `run_vii_c_overhead.sh` | Detection runtime overhead comparison. |

Most campaign defaults live in [scripts/configs](scripts/configs). Review those files before launching long runs.

## Outputs

Each run writes local artifacts under its `--save_dir`:

- `metrics.jsonl`: one JSON object per logging event.
- `summary.json`: final run summary, including detection confusion counts where applicable.
- `run_config.json`: complete resolved run configuration.
- `model_<step>/`: Hugging Face model checkpoint plus `optimizer.pt` and `training_state.json`.

Campaign scripts also write stdout/stderr logs and summary CSV files under `logs/`.

## Important Deviations From the Original Code

This fork includes several engineering changes required by the local reproduction environment:

- Offline local data and tokenizer loading.
- Local JSONL/JSON metrics instead of wandb.
- NVBit 1.7.6 path and build adaptation for CUDA 13.0.
- NVBit callback performance optimization to avoid repeated no-op instrumentation calls.
- Chunked language-model loss computation to reduce peak logits memory.
- Optional training-data repetition for experiments whose step count exceeds the local C4 subset.

See [MODIFICATIONS.md](MODIFICATIONS.md) for detailed rationale and report-ready wording.

## Safety Notes

NVBit fault injection uses `LD_PRELOAD`, installs signal handlers, and modifies GPU register values at runtime. Run these experiments only on trusted, disposable research machines or containers.

When `--fi_nvbit_enable` is used, the Python process sends `SIGUSR1` and `SIGUSR2` to itself. If the NVBit shared object is not preloaded, these signals can terminate the process instead of toggling injection.

Fault injection may produce NaNs, Infs, divergent parameters, CUDA errors, or process crashes. Keep checkpoints and logs isolated from production workloads.

## Troubleshooting

- If NVBit does not build, confirm that the official 1.7.6 `core/` directory exists and that `nvdisasm` is in `PATH`.
- If no faults are injected, check `LD_PRELOAD`, `TARGET_OP`, `TARGET_FUNC_CONTAINS`, `TARGET_FUNC`, and `TARGET_INSTR` in `scripts/configs/nvbit_default.sh`.
- If training exits around 3,906 steps with `Script finished successfully`, the local training dataset was exhausted. Use `--train_data_repeat` for longer campaigns and document the data-reuse deviation.
- If tokenizer loading fails, verify that `tokenizer/t5-base/` contains the downloaded tokenizer files and that the offline paths in `common.sh` are correct.

## Citation

If you use this code, cite the original paper:

```text
Exploring Silent Data Corruption as a Reliability Challenge in LLM Training
```

This repository is a reproduction/adaptation of the original research code, not the canonical upstream release.
