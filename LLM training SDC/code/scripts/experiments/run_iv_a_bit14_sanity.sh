#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh

BIT=14
BITMASK=$((1 << BIT))
GROUP="bit_14"
BIT_LABEL="bit_14"
RUN_NAME="iv_a_${GROUP}_${BIT_LABEL}_seed_${IV_A_SEED}"
SAVE_DIR="$IV_A_CHECKPOINT_ROOT/seed_${IV_A_SEED}/${GROUP}/${BIT_LABEL}"
LOG_DIR="$IV_A_LOG_ROOT/seed_${IV_A_SEED}/${GROUP}"
LOG_FILE="$LOG_DIR/${BIT_LABEL}.log"
GROUP_MANIFEST="$IV_A_CHECKPOINT_ROOT/seed_${IV_A_SEED}/${GROUP}/group_manifest.txt"

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-A-BIT14][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

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
  [[ -f "$1" ]] || { echo "[IV-A-BIT14][ERROR] Required file not found: $1" >&2; exit 1; }
}

[[ "$IV_A_CONFIG_REVISION" -ge 6 ]] || { echo "[IV-A-BIT14][ERROR] IV-A config revision 6 or newer is required." >&2; exit 1; }
[[ "$IV_A_BATCH_SIZE" -eq 256 && "$IV_A_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-A-BIT14][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IV_A_WARMUP_STEPS" -eq "$IV_A_EXIT_AFTER" ]] || { echo "[IV-A-BIT14][ERROR] Expected 1,000 warmup steps in 1,000 training updates." >&2; exit 1; }
[[ "$IV_A_EVAL_EVERY" -gt "$IV_A_EXIT_AFTER" ]] || { echo "[IV-A-BIT14][ERROR] Validation must run only once after training." >&2; exit 1; }
[[ "$IV_A_KEEP_OPTIMIZER_PT" == "0" || "$IV_A_KEEP_OPTIMIZER_PT" == "1" ]] || { echo "[IV-A-BIT14][ERROR] IV_A_KEEP_OPTIMIZER_PT must be 0 or 1." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-A-BIT14][ERROR] Activate the project Python environment first." >&2; exit 1; }

require_file "configs/${IV_A_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$IV_A_BASELINE_DIR/metrics.jsonl"
require_file "$IV_A_BASELINE_DIR/model_${IV_A_EXIT_AFTER}/model.safetensors"

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-A-BIT14][ERROR] NVBit source is newer than the .so. Rebuild it before running bit 14." >&2
  exit 1
fi

if is_complete "$SAVE_DIR"; then
  echo "[IV-A-BIT14] SKIP completed bit=14: $SAVE_DIR"
  exit 0
fi
if [[ -e "$SAVE_DIR" || -e "$LOG_FILE" ]]; then
  echo "[IV-A-BIT14][ERROR] Partial output already exists; not overwriting:" >&2
  echo "  $SAVE_DIR" >&2
  echo "  $LOG_FILE" >&2
  echo "Move or rename the partial output before resuming." >&2
  exit 1
fi

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY
export TOOL_VERBOSE="${IV_A_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$IV_A_TARGET_REGISTER"
export TARGET_OP="$IV_A_TARGET_OP"
export TARGET_SMID="$IV_A_TARGET_SMID"
export TARGET_LANEID="$IV_A_TARGET_LANEID"
export TARGET_FUNC="$IV_A_TARGET_FUNC"
export TARGET_FUNC_CONTAINS="$IV_A_TARGET_FUNC_CONTAINS"
export TARGET_INSTR="$IV_A_BIT_SWEEP_TARGET_INSTR"
export PYTHONUNBUFFERED=1

mkdir -p "$SAVE_DIR" "$LOG_DIR"

cat > "$GROUP_MANIFEST" <<EOF
campaign=iv_a_bit_position_sensitivity
group=$GROUP
seed=$IV_A_SEED
bit_start=$BIT
bit_end=$BIT
selected_bits=$BIT
trigger_rate=$IV_A_TRIGGER_RATE
duration=$IV_A_DURATION
location=$IV_A_BIT_SWEEP_LOCATION
target_func_contains=$IV_A_TARGET_FUNC_CONTAINS
target_instr=$IV_A_BIT_SWEEP_TARGET_INSTR
target_register=$IV_A_TARGET_REGISTER
target_smid=$IV_A_TARGET_SMID
target_laneid=$IV_A_TARGET_LANEID
EOF

echo "============================================================"
echo "IV-A bit-position sanity run: bit 14"
echo "Seed: $IV_A_SEED"
echo "Bitmask: $BITMASK"
echo "Micro batch: $IV_A_BATCH_SIZE; total batch: $IV_A_TOTAL_BATCH_SIZE"
echo "Kernel filter: $TARGET_FUNC_CONTAINS"
echo "Target instruction: $TARGET_INSTR"
echo "Location: $IV_A_BIT_SWEEP_LOCATION"
echo "Injection rate: average 1/${IV_A_TRIGGER_RATE} updates; duration=${IV_A_DURATION}"
if [[ "$IV_A_SEED" -eq 42 ]]; then
  echo "Expected seed-42 schedule: 91 triggered updates, first Step 2, last Step 1000"
fi
echo "Checkpoint: $SAVE_DIR/model_${IV_A_EXIT_AFTER}"
echo "Log: $LOG_FILE"
echo "============================================================"

CURRENT_RUN="$RUN_NAME"
trap - ERR
set +e
TARGET_BITMASK="$BITMASK" LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
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
  --save_dir "$SAVE_DIR" \
  --name "$RUN_NAME" \
  --fi_nvbit_enable \
  --fi_nvbit_location "$IV_A_BIT_SWEEP_LOCATION" \
  --fi_nvbit_trigger_rate "$IV_A_TRIGGER_RATE" \
  --fi_nvbit_target_funcs -1 \
  --fi_nvbit_duration "$IV_A_DURATION" \
  --fi_nvbit_steps -1 \
  2>&1 | tee "$LOG_FILE"
trainer_status=${PIPESTATUS[0]}
set -e
trap 'status=$?; echo "[IV-A-BIT14][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

if [[ "$trainer_status" -ne 0 ]] || ! is_complete "$SAVE_DIR"; then
  printf 'trainer_exit_code=%s\n' "$trainer_status" > "$SAVE_DIR/FAILED"
  echo "[IV-A-BIT14][ERROR] bit 14 run failed or ended without a complete checkpoint." >&2
  exit 1
fi

if [[ "$IV_A_KEEP_OPTIMIZER_PT" == "0" ]]; then
  rm -f -- "$SAVE_DIR/model_${IV_A_EXIT_AFTER}/optimizer.pt"
  echo "[IV-A-BIT14] Removed optimizer.pt (not needed for offline analysis)."
fi

summary_args=(
  --checkpoint-root "$IV_A_CHECKPOINT_ROOT/seed_${IV_A_SEED}/${GROUP}"
  --bit-start "$BIT"
  --bit-end "$BIT"
  --baseline-dir "$IV_A_BASELINE_DIR"
  --compute-parameter-difference
  --output "$LOG_DIR/iv_a_summary_raw.csv"
  --expected-step "$IV_A_EXIT_AFTER"
)
if [[ "$IV_A_SEED" -eq 42 ]]; then
  summary_args+=(
    --expected-trigger-count 91
    --expected-first-trigger 2
    --expected-last-trigger 1000
  )
fi

CURRENT_RUN="IV-A bit 14 summary"
python scripts/experiments/summarize_iv_a.py "${summary_args[@]}"

echo
echo "[IV-A-BIT14] Complete: $SAVE_DIR"
echo "[IV-A-BIT14] Raw summary: $LOG_DIR/iv_a_summary_raw.csv"
