#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_f.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-F][ERROR] ${CURRENT_RUN} failed with exit code ${status}. Check its log before resuming." >&2' ERR

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${IVF_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/optimizer.pt" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" ]]
}

require_file() {
  [[ -f "$1" ]] || { echo "[IV-F][ERROR] Required file not found: $1" >&2; exit 1; }
}

[[ ${#IVF_SEEDS[@]} -eq 3 ]] || { echo "[IV-F][ERROR] IVF_SEEDS must contain exactly 3 seeds." >&2; exit 1; }
[[ "$IVF_CONFIG_REVISION" -ge 3 ]] || { echo "[IV-F][ERROR] Old IV-F config loaded; revision 3 or newer is required." >&2; exit 1; }
[[ "$IVF_BATCH_SIZE" -eq 256 && "$IVF_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-F][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IVF_WARMUP_STEPS" -eq "$IVF_EXIT_AFTER" ]] || { echo "[IV-F][ERROR] The 1,000-step campaign must include exactly 1,000 warmup steps." >&2; exit 1; }
[[ "$IVF_EVAL_EVERY" -gt "$IVF_EXIT_AFTER" ]] || { echo "[IV-F][ERROR] IVF_EVAL_EVERY must exceed IVF_EXIT_AFTER so validation runs only once at the end." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-F][ERROR] python is not available; activate the project virtualenv first." >&2; exit 1; }
require_file "configs/${IVF_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"

# Baselines must not accidentally inherit an NVBit preload from the shell.
unset LD_PRELOAD TOOL_VERBOSE TARGET_FUNC TARGET_INSTR TARGET_SMID TARGET_LANEID
unset TARGET_REGISTER TARGET_BITMASK TARGET_OP TARGET_FUNC_CONTAINS
export PYTHONUNBUFFERED=1

mkdir -p "$IVF_CHECKPOINT_ROOT/baseline" "$IVF_LOG_ROOT/baseline"

echo "============================================================"
echo "IV-F baseline campaign: 3 runs, each ${IVF_EXIT_AFTER} steps"
echo "Seeds: ${IVF_SEEDS[*]}"
echo "Config revision: $IVF_CONFIG_REVISION"
echo "Warmup/training updates: ${IVF_WARMUP_STEPS}/${IVF_EXIT_AFTER}"
echo "Validation: final evaluation only (not counted as a training step)"
echo "Outputs: $IVF_CHECKPOINT_ROOT/baseline"
echo "Best seed rule: lowest final evaluation loss"
echo "============================================================"

run_index=0
for seed in "${IVF_SEEDS[@]}"; do
  run_index=$((run_index + 1))
  run_name="iv_f_baseline_seed_${seed}"
  save_dir="$IVF_CHECKPOINT_ROOT/baseline/seed_${seed}"
  log_file="$IVF_LOG_ROOT/baseline/seed_${seed}.log"
  CURRENT_RUN="$run_name"

  if is_complete "$save_dir"; then
    echo "[IV-F][$run_index/3] SKIP completed baseline: seed=$seed"
    continue
  fi
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[IV-F][ERROR] Partial output already exists for seed=$seed:" >&2
    echo "  $save_dir" >&2
    echo "  $log_file" >&2
    echo "Move or rename the partial output before resuming; it will not be overwritten." >&2
    exit 1
  fi

  echo
  echo "[IV-F][$run_index/3] START baseline seed=$seed"
  echo "Checkpoint: $save_dir/model_${IVF_EXIT_AFTER}"
  echo "Log: $log_file"

  python torchrun_main.py \
    --single_gpu \
    --seed "$seed" \
    --model_config "configs/${IVF_MODEL}.json" \
    --max_length "$IVF_MAX_LENGTH" \
    --lr "$IVF_LR" \
    --scheduler cosine \
    --batch_size "$IVF_BATCH_SIZE" \
    --total_batch_size "$IVF_TOTAL_BATCH_SIZE" \
    --num_training_steps "$IVF_TRAINING_SCHEDULE_STEPS" \
    --warmup_steps "$IVF_WARMUP_STEPS" \
    --eval_every "$IVF_EVAL_EVERY" \
    --exit_after "$IVF_EXIT_AFTER" \
    --save_every "$IVF_SAVE_EVERY" \
    --grad_clipping 1.0 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$IVF_WORKERS" \
    --record_attn_metrics \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --save_dir "$save_dir" \
    --name "$run_name" \
    2>&1 | tee "$log_file"

  is_complete "$save_dir" || { echo "[IV-F][ERROR] $run_name ended without a complete checkpoint." >&2; exit 1; }
  echo "[IV-F][$run_index/3] DONE baseline seed=$seed"
done

CURRENT_RUN="baseline summary"
python scripts/experiments/summarize_iv_f.py \
  --checkpoint-root "$IVF_CHECKPOINT_ROOT" \
  --output "$IVF_LOG_ROOT/iv_f_summary.csv" \
  --aggregate-output "$IVF_LOG_ROOT/iv_f_aggregate.csv" \
  --best-seed-file "$IVF_CHECKPOINT_ROOT/baseline/best_seed.txt" \
  --expected-step "$IVF_EXIT_AFTER"

echo
echo "[IV-F] All three baselines are complete."
echo "[IV-F] Best seed: $(tr -d '[:space:]' < "$IVF_CHECKPOINT_ROOT/baseline/best_seed.txt")"
