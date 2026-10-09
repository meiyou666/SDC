#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-BASELINE][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${IV_A_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

require_file() {
  [[ -f "$1" ]] || { echo "[IV-BASELINE][ERROR] Required file not found: $1" >&2; exit 1; }
}

[[ "$IV_A_CONFIG_REVISION" -ge 5 ]] || { echo "[IV-BASELINE][ERROR] IV-A config revision 5 or newer is required." >&2; exit 1; }
[[ "$IV_A_BATCH_SIZE" -eq 256 && "$IV_A_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-BASELINE][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IV_A_WARMUP_STEPS" -eq "$IV_A_EXIT_AFTER" ]] || { echo "[IV-BASELINE][ERROR] Expected 1,000 warmup steps in 1,000 training updates." >&2; exit 1; }
[[ "$IV_A_EVAL_EVERY" -gt "$IV_A_EXIT_AFTER" ]] || { echo "[IV-BASELINE][ERROR] Validation must run only once after training." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-BASELINE][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${IV_A_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"

# Baseline runs must not inherit an NVBit preload or targeting variables.
unset LD_PRELOAD TOOL_VERBOSE TARGET_FUNC TARGET_INSTR TARGET_SMID TARGET_LANEID
unset TARGET_REGISTER TARGET_BITMASK TARGET_OP TARGET_FUNC_CONTAINS TARGET_EVERY
export PYTHONUNBUFFERED=1

save_dir="$IV_A_BASELINE_DIR"
log_dir="$IV_A_LOG_ROOT/baseline"
log_file="$log_dir/seed_${IV_A_SEED}.log"
run_name="iv_paired_baseline_seed_${IV_A_SEED}"
mkdir -p "$log_dir" "$(dirname "$save_dir")"

echo "============================================================"
echo "IV paired baseline: single seed, ${IV_A_EXIT_AFTER} steps"
echo "Seed: $IV_A_SEED"
echo "Micro batch: $IV_A_BATCH_SIZE; total batch: $IV_A_TOTAL_BATCH_SIZE"
echo "Validation: final evaluation only"
echo "Checkpoint: $save_dir/model_${IV_A_EXIT_AFTER}"
echo "Log: $log_file"
echo "============================================================"

CURRENT_RUN="$run_name"
if is_complete "$save_dir"; then
  echo "[IV-BASELINE] SKIP completed baseline: $save_dir"
  exit 0
fi
if [[ -e "$save_dir" || -e "$log_file" ]]; then
  echo "[IV-BASELINE][ERROR] Partial output already exists:" >&2
  echo "  $save_dir" >&2
  echo "  $log_file" >&2
  echo "Move or rename the partial output before resuming; it will not be overwritten." >&2
  exit 1
fi

python torchrun_main.py \
  --single_gpu \
  --seed "$IV_A_SEED" \
  --model_config "configs/${IV_A_MODEL}.json" \
  --max_length "$IV_A_MAX_LENGTH" \
  --lr "$IV_A_LR" \
  --scheduler cosine \
  --batch_size "$IV_A_BATCH_SIZE" \
  --total_batch_size "$IV_A_TOTAL_BATCH_SIZE" \
  --num_training_steps "$IV_A_TRAINING_SCHEDULE_STEPS" \
  --warmup_steps "$IV_A_WARMUP_STEPS" \
  --eval_every "$IV_A_EVAL_EVERY" \
  --exit_after "$IV_A_EXIT_AFTER" \
  --save_every "$IV_A_SAVE_EVERY" \
  --grad_clipping 1.0 \
  --weight_decay 0.01 \
  --dtype bfloat16 \
  --optimizer adamw \
  --workers "$IV_A_WORKERS" \
  --record_attn_metrics \
  --tokenizer_path "$TOKENIZER_PATH" \
  --train_data_path "$TRAIN_DATA_PATH" \
  --val_data_path "$VAL_DATA_PATH" \
  --save_dir "$save_dir" \
  --name "$run_name" \
  2>&1 | tee "$log_file"

is_complete "$save_dir" || { echo "[IV-BASELINE][ERROR] $run_name ended without a complete checkpoint." >&2; exit 1; }

echo
echo "[IV-BASELINE] Baseline complete: $save_dir"
