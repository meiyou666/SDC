#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh
source scripts/configs/iv_b.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-B][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/experiments/run_iv_b_kernel_sensitivity.sh bit_10
  bash scripts/experiments/run_iv_b_kernel_sensitivity.sh bit_13
  bash scripts/experiments/run_iv_b_kernel_sensitivity.sh bit_14

Runs one bit batch of the simplified IV-B kernel-sensitivity matrix. Each batch
covers every kernel row in logs/iv_a_mb256/kernel_manifest.tsv, using the row's
forward/backward location and kernel-name filter. Within the selected kernel,
all matching HMMA instructions are instrumented.
EOF
}

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${IV_B_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

require_file() {
  [[ -f "$1" ]] || { echo "[IV-B][ERROR] Required file not found: $1" >&2; exit 1; }
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

contains_bit() {
  local needle="$1"
  local value
  for value in "${IV_B_BITS[@]}"; do
    [[ "$value" -eq "$needle" ]] && return 0
  done
  return 1
}

validate_manifest_row() {
  local location="$1"
  local kernel_label="$2"
  local hmma_instr_count="$3"
  local target_func_contains="$4"
  [[ "$location" == "forward" || "$location" == "backward" ]] || { echo "[IV-B][ERROR] Invalid location in manifest row: $kernel_label ($location)" >&2; exit 1; }
  [[ -n "$kernel_label" ]] || { echo "[IV-B][ERROR] Manifest row has no kernel_label." >&2; exit 1; }
  [[ -n "$target_func_contains" ]] || { echo "[IV-B][ERROR] Manifest row has no target_func_contains: $kernel_label" >&2; exit 1; }
  is_positive_integer "$hmma_instr_count" || { echo "[IV-B][ERROR] Manifest row has invalid HMMA instruction count: $kernel_label ($hmma_instr_count)" >&2; exit 1; }
}

group="${1:-}"
if [[ "$group" == "-h" || "$group" == "--help" ]]; then
  usage
  exit 0
fi
[[ -n "$group" ]] || { usage; exit 1; }

case "$group" in
  bit_10) bit=10 ;;
  bit_13) bit=13 ;;
  bit_14) bit=14 ;;
  *) usage; exit 1 ;;
esac
contains_bit "$bit" || { echo "[IV-B][ERROR] Bit $bit is not in IV_B_BITS: ${IV_B_BITS[*]}" >&2; exit 1; }
bitmask=$((1 << bit))

[[ "$IV_B_CONFIG_REVISION" -ge 2 ]] || { echo "[IV-B][ERROR] IV-B config revision 2 or newer is required." >&2; exit 1; }
[[ "$IV_A_CONFIG_REVISION" -ge 6 ]] || { echo "[IV-B][ERROR] IV-A config revision 6 or newer is required." >&2; exit 1; }
[[ "$IV_B_BATCH_SIZE" -eq 256 && "$IV_B_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-B][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IV_B_WARMUP_STEPS" -eq "$IV_B_EXIT_AFTER" ]] || { echo "[IV-B][ERROR] Expected 1,000 warmup steps in 1,000 training updates." >&2; exit 1; }
[[ "$IV_B_EVAL_EVERY" -gt "$IV_B_EXIT_AFTER" ]] || { echo "[IV-B][ERROR] Validation must run only once after training." >&2; exit 1; }
[[ "$IV_B_KEEP_OPTIMIZER_PT" == "0" || "$IV_B_KEEP_OPTIMIZER_PT" == "1" ]] || { echo "[IV-B][ERROR] IV_B_KEEP_OPTIMIZER_PT must be 0 or 1." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-B][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${IV_B_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$IV_B_KERNEL_MANIFEST"
require_file "$IV_B_BASELINE_DIR/metrics.jsonl"
require_file "$IV_B_BASELINE_DIR/model_${IV_B_EXIT_AFTER}/model.safetensors"

expected_header=$'location\tkernel_label\tfunction_id_first_seen\tlaunch_count\thmma_instr_count\thmma_instr_indexes\ttarget_instr\tkernel_name\ttarget_func_contains'
actual_header="$(head -n 1 "$IV_B_KERNEL_MANIFEST")"
[[ "$actual_header" == "$expected_header" ]] || {
  echo "[IV-B][ERROR] Unexpected kernel manifest header. Regenerate it with profile_iv_a_kernels.sh." >&2
  echo "Expected: $expected_header" >&2
  echo "Actual:   $actual_header" >&2
  exit 1
}
manifest_sha256="$(python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$IV_B_KERNEL_MANIFEST")"

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-B][ERROR] NVBit source is newer than the .so. Rebuild it before running IV-B." >&2
  exit 1
fi

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export TOOL_VERBOSE="${IV_B_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$IV_B_TARGET_REGISTER"
export TARGET_OP="$IV_B_TARGET_OP"
export TARGET_SMID="$IV_B_TARGET_SMID"
export TARGET_LANEID="$IV_B_TARGET_LANEID"
export TARGET_FUNC="$IV_B_TARGET_FUNC"
export PYTHONUNBUFFERED=1

batch_checkpoint_dir="$IV_B_CHECKPOINT_ROOT/seed_${IV_B_SEED}/$group"
batch_log_dir="$IV_B_LOG_ROOT/seed_${IV_B_SEED}/$group"
batch_manifest="$batch_checkpoint_dir/batch_manifest.txt"
mkdir -p "$batch_checkpoint_dir" "$batch_log_dir"

selected_count=0
while IFS=$'\t' read -r location kernel_label function_id launch_count hmma_instr_count hmma_instr_indexes target_instr kernel_name target_func_contains; do
  [[ "$location" == "location" ]] && continue
  [[ -n "$location" ]] || continue
  validate_manifest_row "$location" "$kernel_label" "$hmma_instr_count" "$target_func_contains"
  selected_count=$((selected_count + 1))
done < "$IV_B_KERNEL_MANIFEST"
[[ "$selected_count" -gt 0 ]] || { echo "[IV-B][ERROR] No manifest rows selected." >&2; exit 1; }

expected_manifest=$(cat <<EOF
campaign=iv_b_kernel_sensitivity
seed=$IV_B_SEED
bit=$bit
bitmask=$bitmask
kernels=$selected_count
trigger_rate=$IV_B_TRIGGER_RATE
duration=$IV_B_DURATION
target_register=$IV_B_TARGET_REGISTER
target_smid=$IV_B_TARGET_SMID
target_laneid=$IV_B_TARGET_LANEID
target_op=$IV_B_TARGET_OP
target_func=$IV_B_TARGET_FUNC
target_instr=-1_all_matching_hmma
manifest=$IV_B_KERNEL_MANIFEST
manifest_sha256=$manifest_sha256
EOF
)
if [[ -f "$batch_manifest" ]]; then
  recorded_manifest="$(cat "$batch_manifest")"
  [[ "$recorded_manifest" == "$expected_manifest" ]] || {
    echo "[IV-B][ERROR] Existing batch manifest does not match current configuration: $batch_manifest" >&2
    exit 1
  }
else
  printf '%s\n' "$expected_manifest" > "$batch_manifest"
fi

echo "============================================================"
echo "IV-B kernel sensitivity: $group"
echo "Fixed seed: $IV_B_SEED"
echo "Bit position: $bit (bitmask=$bitmask)"
echo "Kernel rows: $selected_count from $IV_B_KERNEL_MANIFEST"
echo "Baseline: $IV_B_BASELINE_DIR"
echo "Target instruction: all matching HMMA instructions in the selected kernel"
echo "Injection rate: average 1/${IV_B_TRIGGER_RATE} updates; duration=${IV_B_DURATION}"
if [[ "$IV_B_SEED" -eq 42 ]]; then
  echo "Expected seed-42 schedule: 91 triggered updates, first Step 2, last Step 1000"
else
  echo "Trigger schedule: random schedule controlled by seed $IV_B_SEED"
fi
echo "Location/kernel filter come from manifest; SM=$TARGET_SMID; lane=$TARGET_LANEID; input register=$TARGET_REGISTER"
echo "Checkpoint root: $batch_checkpoint_dir"
echo "Log root: $batch_log_dir"
echo "============================================================"

failed_runs=()
run_index=0
while IFS=$'\t' read -r location kernel_label function_id launch_count hmma_instr_count hmma_instr_indexes target_instr kernel_name target_func_contains; do
  [[ "$location" == "location" ]] && continue
  [[ -n "$location" ]] || continue
  validate_manifest_row "$location" "$kernel_label" "$hmma_instr_count" "$target_func_contains"

  run_index=$((run_index + 1))
  safe_label="${kernel_label//[^A-Za-z0-9_]/_}"
  run_name="iv_b_${group}_${kernel_label}_seed_${IV_B_SEED}"
  save_dir="$batch_checkpoint_dir/$safe_label"
  log_file="$batch_log_dir/${safe_label}.log"
  CURRENT_RUN="$run_name"

  if is_complete "$save_dir"; then
    echo "[IV-B][$run_index/$selected_count] SKIP completed kernel=$kernel_label location=$location bit=$bit"
    continue
  fi
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[IV-B][$run_index/$selected_count] WARN partial output exists; not overwriting kernel=$kernel_label bit=$bit" >&2
    failed_runs+=("${kernel_label}:partial")
    continue
  fi

  echo
  echo "[IV-B][$run_index/$selected_count] START kernel=$kernel_label location=$location bit=$bit"
  echo "Kernel: $kernel_name"
  echo "Target contains: $target_func_contains"
  echo "HMMA instr count: $hmma_instr_count; TARGET_INSTR=-1 (all matching HMMA)"
  echo "Checkpoint: $save_dir/model_${IV_B_EXIT_AFTER}"
  echo "Log: $log_file"

  trap - ERR
  set +e
  TARGET_BITMASK="$bitmask" TARGET_FUNC_CONTAINS="$target_func_contains" TARGET_INSTR=-1 LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
    --single_gpu \
    --seed "$IV_B_SEED" \
    --model_config "configs/${IV_B_MODEL}.json" \
    --max_length "$IV_B_MAX_LENGTH" \
    --lr "$IV_B_LR" \
    --scheduler cosine \
    --batch_size "$IV_B_BATCH_SIZE" \
    --total_batch_size "$IV_B_TOTAL_BATCH_SIZE" \
    --num_training_steps "$IV_B_TRAINING_SCHEDULE_STEPS" \
    --warmup_steps "$IV_B_WARMUP_STEPS" \
    --eval_every "$IV_B_EVAL_EVERY" \
    --exit_after "$IV_B_EXIT_AFTER" \
    --save_every "$IV_B_SAVE_EVERY" \
    --grad_clipping 1.0 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$IV_B_WORKERS" \
    --record_attn_metrics \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --save_dir "$save_dir" \
    --name "$run_name" \
    --fi_nvbit_enable \
    --fi_nvbit_location "$location" \
    --fi_nvbit_trigger_rate "$IV_B_TRIGGER_RATE" \
    --fi_nvbit_target_funcs -1 \
    --fi_nvbit_duration "$IV_B_DURATION" \
    --fi_nvbit_steps -1 \
    2>&1 | tee "$log_file"
  trainer_status=${PIPESTATUS[0]}
  set -e
  trap 'status=$?; echo "[IV-B][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

  if [[ "$trainer_status" -ne 0 ]] || ! is_complete "$save_dir"; then
    mkdir -p "$save_dir"
    printf 'trainer_exit_code=%s\n' "$trainer_status" > "$save_dir/FAILED"
    failed_runs+=("${kernel_label}:exit_${trainer_status}")
    echo "[IV-B][$run_index/$selected_count] FAILED kernel=$kernel_label bit=$bit; continuing with the next kernel." >&2
    continue
  fi
  if [[ "$IV_B_KEEP_OPTIMIZER_PT" == "0" ]]; then
    rm -f -- "$save_dir/model_${IV_B_EXIT_AFTER}/optimizer.pt"
    echo "[IV-B][$run_index/$selected_count] Removed optimizer.pt (not needed for offline analysis)."
  fi
  echo "[IV-B][$run_index/$selected_count] DONE kernel=$kernel_label bit=$bit"
done < "$IV_B_KERNEL_MANIFEST"

summary_args=(
  --checkpoint-root "$batch_checkpoint_dir"
  --manifest "$IV_B_KERNEL_MANIFEST"
  --bit "$bit"
  --baseline-dir "$IV_B_BASELINE_DIR"
  --compute-parameter-difference
  --output "$batch_log_dir/iv_b_summary_raw.csv"
  --expected-step "$IV_B_EXIT_AFTER"
)
if [[ "$IV_B_SEED" -eq 42 ]]; then
  summary_args+=(
    --expected-trigger-count 91
    --expected-first-trigger 2
    --expected-last-trigger 1000
  )
fi

CURRENT_RUN="IV-B kernel sensitivity summary"
python scripts/experiments/summarize_iv_b.py "${summary_args[@]}"

echo
if [[ ${#failed_runs[@]} -eq 0 ]]; then
  echo "[IV-B] All kernel-sensitivity runs completed for $group."
else
  echo "[IV-B] $group finished with ${#failed_runs[@]} failed/partial run(s): ${failed_runs[*]}" >&2
  echo "[IV-B] Completed runs and the raw summary were preserved." >&2
  exit 1
fi
echo "[IV-B] Raw summary: $batch_log_dir/iv_b_summary_raw.csv"
