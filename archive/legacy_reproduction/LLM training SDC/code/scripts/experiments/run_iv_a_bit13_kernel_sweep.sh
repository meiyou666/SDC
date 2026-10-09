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
  bash scripts/experiments/profile_iv_a_kernels.sh
  bash scripts/experiments/run_iv_a_bit13_kernel_sweep.sh [kernel_manifest.tsv] [FP1 BP1 ...]

The sweep script runs bit position 13 across profiled forward/backward GEMM
kernels one kernel at a time. This is a diagnostic kernel-sensitivity probe,
not the main IV-A bit-position sweep.
EOF
}

manifest_path="${1:-$IV_A_KERNEL_MANIFEST}"
if [[ "$manifest_path" == "-h" || "$manifest_path" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

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

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

validate_manifest_row() {
  local location="$1"
  local kernel_label="$2"
  local hmma_instr_count="$3"
  local target_instr="$4"
  [[ "$location" == "forward" || "$location" == "backward" ]] || { echo "[IV-A][ERROR] Invalid location in manifest row: $kernel_label ($location)" >&2; exit 1; }
  [[ -n "$target_instr" ]] || { echo "[IV-A][ERROR] Manifest row has no selected target_instr: $kernel_label" >&2; exit 1; }
  is_positive_integer "$hmma_instr_count" || { echo "[IV-A][ERROR] Manifest row has invalid HMMA instruction count: $kernel_label ($hmma_instr_count)" >&2; exit 1; }
}

[[ "$IV_A_CONFIG_REVISION" -ge 6 ]] || { echo "[IV-A][ERROR] IV-A config revision 6 or newer is required." >&2; exit 1; }
[[ "$IV_A_BATCH_SIZE" -eq 256 && "$IV_A_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-A][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IV_A_BIT" -eq 13 && "$IV_A_TARGET_BITMASK" -eq 8192 ]] || { echo "[IV-A][ERROR] This diagnostic probe must use bit position 13 / bitmask 8192." >&2; exit 1; }
[[ "$IV_A_WARMUP_STEPS" -eq "$IV_A_EXIT_AFTER" ]] || { echo "[IV-A][ERROR] Expected 1,000 warmup steps in 1,000 training updates." >&2; exit 1; }
[[ "$IV_A_EVAL_EVERY" -gt "$IV_A_EXIT_AFTER" ]] || { echo "[IV-A][ERROR] Validation must run only once after training." >&2; exit 1; }
[[ "$IV_A_KEEP_OPTIMIZER_PT" == "0" || "$IV_A_KEEP_OPTIMIZER_PT" == "1" ]] || { echo "[IV-A][ERROR] IV_A_KEEP_OPTIMIZER_PT must be 0 or 1." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-A][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${IV_A_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$manifest_path"
require_file "$IV_A_BASELINE_DIR/metrics.jsonl"
require_file "$IV_A_BASELINE_DIR/model_${IV_A_EXIT_AFTER}/model.safetensors"

expected_header=$'location	kernel_label	function_id_first_seen	launch_count	hmma_instr_count	hmma_instr_indexes	target_instr	kernel_name	target_func_contains'
actual_header="$(head -n 1 "$manifest_path")"
[[ "$actual_header" == "$expected_header" ]] || {
  echo "[IV-A][ERROR] Unexpected kernel manifest header. Regenerate it with profile_iv_a_kernels.sh." >&2
  echo "Expected: $expected_header" >&2
  echo "Actual:   $actual_header" >&2
  exit 1
}
manifest_sha256="$(python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$manifest_path")"

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-A][ERROR] NVBit source is newer than the .so. Rebuild it before running IV-A." >&2
  exit 1
fi

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export TOOL_VERBOSE="${IV_A_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$IV_A_TARGET_REGISTER"
export TARGET_OP="$IV_A_TARGET_OP"
export TARGET_SMID="$IV_A_TARGET_SMID"
export TARGET_LANEID="$IV_A_TARGET_LANEID"
export TARGET_FUNC="$IV_A_TARGET_FUNC"
export PYTHONUNBUFFERED=1

declare -A label_filter=()
if [[ "$#" -gt 0 ]]; then
  for label in "$@"; do
    label_filter["$label"]=1
  done
fi

run_checkpoint_dir="$IV_A_CHECKPOINT_ROOT/seed_${IV_A_SEED}/bit_13"
run_log_dir="$IV_A_LOG_ROOT/seed_${IV_A_SEED}/bit_13"
run_manifest="$run_checkpoint_dir/run_manifest.txt"
mkdir -p "$run_checkpoint_dir" "$run_log_dir"

expected_manifest=$(cat <<EOF
campaign=bit_13_kernel_diagnostic_sweep
seed=$IV_A_SEED
bit=$IV_A_BIT
bitmask=$IV_A_TARGET_BITMASK
target_instr=manifest_selected_hmma
instr_selection_seed=$IV_A_INSTR_SELECTION_SEED
trigger_rate=$IV_A_TRIGGER_RATE
duration=$IV_A_DURATION
manifest=$manifest_path
manifest_sha256=$manifest_sha256
EOF
)
if [[ -f "$run_manifest" ]]; then
  recorded_manifest="$(cat "$run_manifest")"
  [[ "$recorded_manifest" == "$expected_manifest" ]] || {
    echo "[IV-A][ERROR] Existing run manifest does not match current configuration: $run_manifest" >&2
    exit 1
  }
else
  printf '%s\n' "$expected_manifest" > "$run_manifest"
fi

selected_count=0
while IFS=$'\t' read -r location kernel_label function_id launch_count hmma_instr_count hmma_instr_indexes target_instr kernel_name target_func_contains; do
  [[ "$location" == "location" ]] && continue
  [[ -n "$location" ]] || continue
  validate_manifest_row "$location" "$kernel_label" "$hmma_instr_count" "$target_instr"
  if [[ ${#label_filter[@]} -gt 0 && -z "${label_filter[$kernel_label]:-}" ]]; then
    continue
  fi
  selected_count=$((selected_count + 1))
done < "$manifest_path"
[[ "$selected_count" -gt 0 ]] || { echo "[IV-A][ERROR] No manifest rows selected." >&2; exit 1; }

echo "============================================================"
echo "IV-A bit-13 kernel diagnostic sweep"
echo "Fixed seed: $IV_A_SEED"
echo "Bit position: $IV_A_BIT (bitmask=$IV_A_TARGET_BITMASK)"
echo "Kernel manifest: $manifest_path"
echo "Baseline: $IV_A_BASELINE_DIR"
echo "Target instruction: selected per manifest row (seed=$IV_A_INSTR_SELECTION_SEED)"
echo "Injection rate: average 1/${IV_A_TRIGGER_RATE} updates; duration=${IV_A_DURATION}"
if [[ "$IV_A_SEED" -eq 42 ]]; then
  echo "Expected seed-42 schedule: 91 triggered updates, first Step 2, last Step 1000"
else
  echo "Trigger schedule: random schedule controlled by seed $IV_A_SEED"
fi
echo "Locations come from manifest; SM=$TARGET_SMID; lane=$TARGET_LANEID; input register=$TARGET_REGISTER"
echo "Checkpoint root: $run_checkpoint_dir"
echo "Log root: $run_log_dir"
echo "============================================================"

failed_runs=()
run_index=0
while IFS=$'\t' read -r location kernel_label function_id launch_count hmma_instr_count hmma_instr_indexes target_instr kernel_name target_func_contains; do
  [[ "$location" == "location" ]] && continue
  [[ -n "$location" ]] || continue
  validate_manifest_row "$location" "$kernel_label" "$hmma_instr_count" "$target_instr"
  if [[ ${#label_filter[@]} -gt 0 && -z "${label_filter[$kernel_label]:-}" ]]; then
    continue
  fi

  run_index=$((run_index + 1))
  safe_label="${kernel_label//[^A-Za-z0-9_]/_}"
  run_name="iv_a_bit13_${kernel_label}_seed_${IV_A_SEED}"
  save_dir="$run_checkpoint_dir/$safe_label"
  log_file="$run_log_dir/${safe_label}.log"
  CURRENT_RUN="$run_name"

  if is_complete "$save_dir"; then
    echo "[IV-A][$run_index/$selected_count] SKIP completed kernel=$kernel_label location=$location"
    continue
  fi
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[IV-A][$run_index/$selected_count] WARN partial output exists; not overwriting kernel=$kernel_label" >&2
    failed_runs+=("${kernel_label}:partial")
    continue
  fi

  echo
  echo "[IV-A][$run_index/$selected_count] START kernel=$kernel_label location=$location"
  echo "Kernel: $kernel_name"
  echo "Target contains: $target_func_contains"
  echo "HMMA instr count: $hmma_instr_count; selected TARGET_INSTR=$target_instr"
  echo "Checkpoint: $save_dir/model_${IV_A_EXIT_AFTER}"
  echo "Log: $log_file"

  trap - ERR
  set +e
  TARGET_BITMASK="$IV_A_TARGET_BITMASK" TARGET_FUNC_CONTAINS="$target_func_contains" TARGET_INSTR="$target_instr" LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
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
    --fi_nvbit_location "$location" \
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
    failed_runs+=("${kernel_label}:exit_${trainer_status}")
    echo "[IV-A][$run_index/$selected_count] FAILED kernel=$kernel_label; continuing with the next kernel." >&2
    continue
  fi
  if [[ "$IV_A_KEEP_OPTIMIZER_PT" == "0" ]]; then
    rm -f -- "$save_dir/model_${IV_A_EXIT_AFTER}/optimizer.pt"
    echo "[IV-A][$run_index/$selected_count] Removed optimizer.pt (not needed for offline analysis)."
  fi
  echo "[IV-A][$run_index/$selected_count] DONE kernel=$kernel_label"
done < "$manifest_path"

summary_args=(
  --checkpoint-root "$run_checkpoint_dir"
  --manifest "$manifest_path"
  --baseline-dir "$IV_A_BASELINE_DIR"
  --compute-parameter-difference
  --output "$run_log_dir/iv_a_bit13_kernel_summary_raw.csv"
  --expected-step "$IV_A_EXIT_AFTER"
)
if [[ "$IV_A_SEED" -eq 42 ]]; then
  summary_args+=(
    --expected-trigger-count 91
    --expected-first-trigger 2
    --expected-last-trigger 1000
  )
fi

CURRENT_RUN="IV-A bit-13 kernel diagnostic summary"
python scripts/experiments/summarize_iv_a_kernel_sweep.py "${summary_args[@]}"

echo
if [[ ${#failed_runs[@]} -eq 0 ]]; then
  echo "[IV-A] All selected bit-13 kernel diagnostic runs completed."
else
  echo "[IV-A] bit-13 diagnostic sweep finished with ${#failed_runs[@]} failed/partial run(s): ${failed_runs[*]}" >&2
  echo "[IV-A] Completed runs and the raw summary were preserved." >&2
  exit 1
fi
echo "[IV-A] Raw summary: $run_log_dir/iv_a_bit13_kernel_summary_raw.csv"
