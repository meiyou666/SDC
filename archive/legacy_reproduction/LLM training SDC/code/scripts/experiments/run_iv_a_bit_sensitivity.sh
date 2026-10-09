#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-A][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/experiments/run_iv_a_bit_sensitivity.sh bit_13
  bash scripts/experiments/run_iv_a_bit_sensitivity.sh bits_00_15
  bash scripts/experiments/run_iv_a_bit_sensitivity.sh bits_16_31

Runs the IV-A bit-position sensitivity sweep for one half of the 32-bit
packed BF16 register. Bits listed in IV_A_SKIP_BITS are intentionally skipped
because they are handled by separate sanity runs.
EOF
}

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
  [[ -f "$1" ]] || { echo "[IV-A][ERROR] Required file not found: $1" >&2; exit 1; }
}

contains_bit() {
  local needle="$1"
  local value
  for value in "${IV_A_SKIP_BITS[@]}"; do
    [[ "$value" -eq "$needle" ]] && return 0
  done
  return 1
}

group="${1:-}"
if [[ "$group" == "-h" || "$group" == "--help" ]]; then
  usage
  exit 0
fi
[[ -n "$group" ]] || { usage; exit 1; }

case "$group" in
  bit_13)
    bit_start=13
    bit_end=13
    ;;
  bits_00_15)
    bit_start=0
    bit_end=15
    ;;
  bits_16_31)
    bit_start=16
    bit_end=31
    ;;
  *)
    usage
    exit 1
    ;;
esac

[[ "$IV_A_CONFIG_REVISION" -ge 6 ]] || { echo "[IV-A][ERROR] IV-A config revision 6 or newer is required." >&2; exit 1; }
[[ "$IV_A_BATCH_SIZE" -eq 256 && "$IV_A_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-A][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IV_A_WARMUP_STEPS" -eq "$IV_A_EXIT_AFTER" ]] || { echo "[IV-A][ERROR] Expected 1,000 warmup steps in 1,000 training updates." >&2; exit 1; }
[[ "$IV_A_EVAL_EVERY" -gt "$IV_A_EXIT_AFTER" ]] || { echo "[IV-A][ERROR] Validation must run only once after training." >&2; exit 1; }
[[ "$IV_A_KEEP_OPTIMIZER_PT" == "0" || "$IV_A_KEEP_OPTIMIZER_PT" == "1" ]] || { echo "[IV-A][ERROR] IV_A_KEEP_OPTIMIZER_PT must be 0 or 1." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-A][ERROR] Activate the project Python environment first." >&2; exit 1; }
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
  echo "[IV-A][ERROR] NVBit source is newer than the .so. Rebuild it before running IV-A." >&2
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

group_checkpoint_dir="$IV_A_CHECKPOINT_ROOT/seed_${IV_A_SEED}/$group"
group_log_dir="$IV_A_LOG_ROOT/seed_${IV_A_SEED}/$group"
group_manifest="$group_checkpoint_dir/group_manifest.txt"
mkdir -p "$group_checkpoint_dir" "$group_log_dir"

bit_list=()
for bit in $(seq "$bit_start" "$bit_end"); do
  if [[ "$group" != "bit_13" ]] && contains_bit "$bit"; then
    continue
  fi
  bit_list+=("$bit")
done
[[ "${#bit_list[@]}" -gt 0 ]] || { echo "[IV-A][ERROR] No bits selected after skip filter." >&2; exit 1; }

expected_manifest=$(cat <<EOF
campaign=iv_a_bit_position_sensitivity
group=$group
seed=$IV_A_SEED
bit_start=$bit_start
bit_end=$bit_end
skip_bits=${IV_A_SKIP_BITS[*]}
selected_bits=${bit_list[*]}
trigger_rate=$IV_A_TRIGGER_RATE
duration=$IV_A_DURATION
location=$IV_A_BIT_SWEEP_LOCATION
target_func_contains=$IV_A_TARGET_FUNC_CONTAINS
target_instr=$IV_A_BIT_SWEEP_TARGET_INSTR
target_register=$IV_A_TARGET_REGISTER
target_smid=$IV_A_TARGET_SMID
target_laneid=$IV_A_TARGET_LANEID
EOF
)
if [[ -f "$group_manifest" ]]; then
  recorded_manifest="$(cat "$group_manifest")"
  [[ "$recorded_manifest" == "$expected_manifest" ]] || {
    echo "[IV-A][ERROR] Existing group manifest does not match current configuration: $group_manifest" >&2
    exit 1
  }
else
  printf '%s\n' "$expected_manifest" > "$group_manifest"
fi

echo "============================================================"
echo "IV-A bit-position sensitivity: $group"
echo "Fixed seed: $IV_A_SEED"
echo "Bits selected: ${bit_list[*]}"
echo "Bits skipped: ${IV_A_SKIP_BITS[*]}"
echo "Kernel filter: $TARGET_FUNC_CONTAINS"
echo "Target instruction: $TARGET_INSTR"
echo "Location: $IV_A_BIT_SWEEP_LOCATION"
echo "Injection rate: average 1/${IV_A_TRIGGER_RATE} updates; duration=${IV_A_DURATION}"
if [[ "$IV_A_SEED" -eq 42 ]]; then
  echo "Expected seed-42 schedule: 91 triggered updates, first Step 2, last Step 1000"
else
  echo "Trigger schedule: random schedule controlled by seed $IV_A_SEED"
fi
echo "Checkpoint root: $group_checkpoint_dir"
echo "Log root: $group_log_dir"
echo "============================================================"

failed_runs=()
run_index=0
selected_count="${#bit_list[@]}"
for bit in "${bit_list[@]}"; do
  run_index=$((run_index + 1))
  bitmask=$((1 << bit))
  bit_label=$(printf 'bit_%02d' "$bit")
  run_name="iv_a_${group}_${bit_label}_seed_${IV_A_SEED}"
  save_dir="$group_checkpoint_dir/$bit_label"
  log_file="$group_log_dir/${bit_label}.log"
  CURRENT_RUN="$run_name"

  if is_complete "$save_dir"; then
    echo "[IV-A][$run_index/$selected_count] SKIP completed bit=$bit"
    continue
  fi
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[IV-A][$run_index/$selected_count] WARN partial output exists; not overwriting bit=$bit" >&2
    failed_runs+=("${bit_label}:partial")
    continue
  fi

  echo
  echo "[IV-A][$run_index/$selected_count] START bit=$bit bitmask=$bitmask"
  echo "Checkpoint: $save_dir/model_${IV_A_EXIT_AFTER}"
  echo "Log: $log_file"

  trap - ERR
  set +e
  TARGET_BITMASK="$bitmask" LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
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
    --fi_nvbit_enable \
    --fi_nvbit_location "$IV_A_BIT_SWEEP_LOCATION" \
    --fi_nvbit_trigger_rate "$IV_A_TRIGGER_RATE" \
    --fi_nvbit_target_funcs -1 \
    --fi_nvbit_duration "$IV_A_DURATION" \
    --fi_nvbit_steps -1 \
    2>&1 | tee "$log_file"
  trainer_status=${PIPESTATUS[0]}
  set -e
  trap 'status=$?; echo "[IV-A][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

  if [[ "$trainer_status" -ne 0 ]] || ! is_complete "$save_dir"; then
    mkdir -p "$save_dir"
    printf 'trainer_exit_code=%s\n' "$trainer_status" > "$save_dir/FAILED"
    failed_runs+=("${bit_label}:exit_${trainer_status}")
    echo "[IV-A][$run_index/$selected_count] FAILED bit=$bit; continuing with the next bit." >&2
    continue
  fi
  if [[ "$IV_A_KEEP_OPTIMIZER_PT" == "0" ]]; then
    rm -f -- "$save_dir/model_${IV_A_EXIT_AFTER}/optimizer.pt"
    echo "[IV-A][$run_index/$selected_count] Removed optimizer.pt (not needed for offline analysis)."
  fi
  echo "[IV-A][$run_index/$selected_count] DONE bit=$bit"
done

summary_args=(
  --checkpoint-root "$group_checkpoint_dir"
  --bit-start "$bit_start"
  --bit-end "$bit_end"
  --baseline-dir "$IV_A_BASELINE_DIR"
  --compute-parameter-difference
  --output "$group_log_dir/iv_a_summary_raw.csv"
  --expected-step "$IV_A_EXIT_AFTER"
)
for skipped_bit in "${IV_A_SKIP_BITS[@]}"; do
  if [[ "$group" != "bit_13" ]]; then
    summary_args+=(--skip-bit "$skipped_bit")
  fi
done
if [[ "$IV_A_SEED" -eq 42 ]]; then
  summary_args+=(
    --expected-trigger-count 91
    --expected-first-trigger 2
    --expected-last-trigger 1000
  )
fi

CURRENT_RUN="IV-A bit-position summary"
python scripts/experiments/summarize_iv_a.py "${summary_args[@]}"

echo
if [[ ${#failed_runs[@]} -eq 0 ]]; then
  echo "[IV-A] Selected bit-position runs completed for $group."
else
  echo "[IV-A] $group finished with ${#failed_runs[@]} failed/partial run(s): ${failed_runs[*]}" >&2
  echo "[IV-A] Completed runs and the raw summary were preserved." >&2
  exit 1
fi
echo "[IV-A] Raw summary: $group_log_dir/iv_a_summary_raw.csv"
