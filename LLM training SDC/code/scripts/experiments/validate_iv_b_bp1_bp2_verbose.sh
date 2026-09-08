#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh
source scripts/configs/iv_b.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-B-VALIDATE][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

require_file() {
  [[ -f "$1" ]] || { echo "[IV-B-VALIDATE][ERROR] Required file not found: $1" >&2; exit 1; }
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${VALIDATE_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

[[ "$IV_B_CONFIG_REVISION" -ge 2 ]] || { echo "[IV-B-VALIDATE][ERROR] IV-B config revision 2 or newer is required." >&2; exit 1; }
[[ "$IV_A_CONFIG_REVISION" -ge 6 ]] || { echo "[IV-B-VALIDATE][ERROR] IV-A config revision 6 or newer is required." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-B-VALIDATE][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${IV_B_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$IV_B_KERNEL_MANIFEST"

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-B-VALIDATE][ERROR] NVBit source is newer than the .so. Rebuild it before validation." >&2
  exit 1
fi

VALIDATE_BIT=14
VALIDATE_BITMASK=$((1 << VALIDATE_BIT))
VALIDATE_EXIT_AFTER=5
VALIDATE_TRIGGER_STEP=2
VALIDATE_KERNELS=(BP1 BP2)
VALIDATE_ROOT="$IV_B_CHECKPOINT_ROOT/validation/bp1_bp2_bit14_all_hmma_verbose"
VALIDATE_LOG_ROOT="$IV_B_LOG_ROOT/validation/bp1_bp2_bit14_all_hmma_verbose"
mkdir -p "$VALIDATE_ROOT" "$VALIDATE_LOG_ROOT"

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export TOOL_VERBOSE=1
export TARGET_REGISTER="$IV_B_TARGET_REGISTER"
export TARGET_OP="$IV_B_TARGET_OP"
export TARGET_SMID="$IV_B_TARGET_SMID"
export TARGET_LANEID="$IV_B_TARGET_LANEID"
export TARGET_FUNC="$IV_B_TARGET_FUNC"
export TARGET_INSTR=-1
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "IV-B BP1/BP2 verbose validation"
echo "Purpose: verify BP1 and BP2 map to distinct kernel filters and HMMA instructions"
echo "Bit: $VALIDATE_BIT (bitmask=$VALIDATE_BITMASK)"
echo "TARGET_INSTR=-1, TOOL_VERBOSE=1"
echo "Forced injection step: $VALIDATE_TRIGGER_STEP; exit_after=$VALIDATE_EXIT_AFTER"
echo "Kernels: ${VALIDATE_KERNELS[*]}"
echo "Checkpoint root: $VALIDATE_ROOT"
echo "Log root: $VALIDATE_LOG_ROOT"
echo "============================================================"

for kernel_label in "${VALIDATE_KERNELS[@]}"; do
  location=""
  hmma_instr_count=""
  target_func_contains=""
  kernel_name=""
  while IFS=$'\t' read -r row_location row_label function_id launch_count row_hmma_instr_count hmma_instr_indexes target_instr row_kernel_name row_target_func_contains; do
    [[ "$row_location" == "location" ]] && continue
    if [[ "$row_label" == "$kernel_label" ]]; then
      location="$row_location"
      hmma_instr_count="$row_hmma_instr_count"
      target_func_contains="$row_target_func_contains"
      kernel_name="$row_kernel_name"
      break
    fi
  done < "$IV_B_KERNEL_MANIFEST"

  [[ "$location" == "backward" ]] || { echo "[IV-B-VALIDATE][ERROR] Expected $kernel_label to be a backward row, got: $location" >&2; exit 1; }
  [[ -n "$target_func_contains" ]] || { echo "[IV-B-VALIDATE][ERROR] Manifest row not found or empty target filter for $kernel_label" >&2; exit 1; }
  is_positive_integer "$hmma_instr_count" || { echo "[IV-B-VALIDATE][ERROR] Invalid HMMA instruction count for $kernel_label: $hmma_instr_count" >&2; exit 1; }

  save_dir="$VALIDATE_ROOT/$kernel_label"
  log_file="$VALIDATE_LOG_ROOT/${kernel_label}.log"
  CURRENT_RUN="validate_${kernel_label}"

  if is_complete "$save_dir"; then
    echo "[IV-B-VALIDATE] SKIP completed $kernel_label: $save_dir"
    continue
  fi
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[IV-B-VALIDATE][ERROR] Partial validation output exists; not overwriting:" >&2
    echo "  $save_dir" >&2
    echo "  $log_file" >&2
    exit 1
  fi

  echo
  echo "[IV-B-VALIDATE] START $kernel_label"
  echo "Kernel: $kernel_name"
  echo "Target contains: $target_func_contains"
  echo "Profiled HMMA instruction count: $hmma_instr_count; TARGET_INSTR=-1"

  TARGET_BITMASK="$VALIDATE_BITMASK" TARGET_FUNC_CONTAINS="$target_func_contains" LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
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
    --exit_after "$VALIDATE_EXIT_AFTER" \
    --save_every "$VALIDATE_EXIT_AFTER" \
    --grad_clipping 1.0 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$IV_B_WORKERS" \
    --record_attn_metrics \
    --disable_final_evaluation \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --save_dir "$save_dir" \
    --name "iv_b_validate_${kernel_label}_bit14_all_hmma_seed_${IV_B_SEED}" \
    --fi_nvbit_enable \
    --fi_nvbit_location backward \
    --fi_nvbit_trigger_rate "$IV_B_TRIGGER_RATE" \
    --fi_nvbit_target_funcs -1 \
    --fi_nvbit_duration 1 \
    --fi_nvbit_steps "$VALIDATE_TRIGGER_STEP" \
    2>&1 | tee "$log_file"

  if ! is_complete "$save_dir"; then
    echo "[IV-B-VALIDATE][ERROR] $kernel_label validation ended without a complete checkpoint." >&2
    exit 1
  fi

  echo "[IV-B-VALIDATE] Evidence counts for $kernel_label:"
  grep -F "TARGET_FUNC_CONTAINS" "$log_file" | head -n 1 || true
  printf '  inspecting lines: '
  grep -cF "[NVBit Fault Injector] inspecting" "$log_file" || true
  printf '  HMMA instrumented idx lines: '
  grep -cF "idx:" "$log_file" || true
  echo "[IV-B-VALIDATE] DONE $kernel_label"
done

python scripts/experiments/summarize_iv_b_verbose_validation.py \
  --log "$VALIDATE_LOG_ROOT/BP1.log" \
  --log "$VALIDATE_LOG_ROOT/BP2.log" \
  --output "$VALIDATE_LOG_ROOT/verbose_validation_summary.tsv"

echo
echo "[IV-B-VALIDATE] Compare the summary and logs:"
echo "  $VALIDATE_LOG_ROOT/verbose_validation_summary.tsv"
echo "  $VALIDATE_LOG_ROOT/BP1.log"
echo "  $VALIDATE_LOG_ROOT/BP2.log"
echo "Expected evidence: different TARGET_FUNC_CONTAINS values and different verbose HMMA idx sets."
