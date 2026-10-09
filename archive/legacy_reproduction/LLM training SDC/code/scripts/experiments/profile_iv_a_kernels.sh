#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-A][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

require_file() {
  [[ -f "$1" ]] || { echo "[IV-A][ERROR] Required file not found: $1" >&2; exit 1; }
}

[[ "$IV_A_CONFIG_REVISION" -ge 5 ]] || { echo "[IV-A][ERROR] IV-A config revision 5 or newer is required." >&2; exit 1; }
[[ "$IV_A_BATCH_SIZE" -eq 256 && "$IV_A_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-A][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-A][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${IV_A_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-A][ERROR] NVBit source is newer than the .so. Rebuild it before profiling." >&2
  exit 1
fi

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY
export TOOL_VERBOSE=1
export TARGET_REGISTER="$IV_A_TARGET_REGISTER"
export TARGET_OP="$IV_A_TARGET_OP"
export TARGET_SMID="$IV_A_TARGET_SMID"
export TARGET_LANEID="$IV_A_TARGET_LANEID"
# Kernel profiling only needs NVBit launch/instrumentation logs. Use a zero
# bitmask so the short profiling runs enumerate kernels without corrupting the
# model state before the real bit-13 sweep.
export TARGET_BITMASK=0
export TARGET_FUNC="$IV_A_TARGET_FUNC"
export TARGET_INSTR=-1
export TARGET_FUNC_CONTAINS="$IV_A_PROFILE_FUNC_CONTAINS"
export PYTHONUNBUFFERED=1

profile_checkpoint_dir="$IV_A_CHECKPOINT_ROOT/profile"
profile_log_dir="$IV_A_LOG_ROOT/profile"
mkdir -p "$profile_checkpoint_dir" "$profile_log_dir"

echo "============================================================"
echo "IV-A kernel profiling"
echo "Micro batch: $IV_A_BATCH_SIZE; total batch: $IV_A_TOTAL_BATCH_SIZE"
echo "Locations: ${IV_A_LOCATIONS[*]}"
echo "Profile step: $IV_A_PROFILE_TRIGGER_STEP; exit after: $IV_A_PROFILE_EXIT_AFTER"
echo "Function filter: $TARGET_FUNC_CONTAINS"
echo "Manifest output: $IV_A_KERNEL_MANIFEST"
echo "============================================================"

profile_logs=()
for location in "${IV_A_LOCATIONS[@]}"; do
  CURRENT_RUN="profile_$location"
  save_dir="$profile_checkpoint_dir/$location"
  log_file="$profile_log_dir/profile_${location}.log"
  profile_logs+=("$log_file")

  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[IV-A][ERROR] Existing profile output would be overwritten:" >&2
    echo "  $save_dir" >&2
    echo "  $log_file" >&2
    echo "Move or rename it before profiling again." >&2
    exit 1
  fi

  echo
  echo "[IV-A] Profiling $location kernels"
  LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
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
    --exit_after "$IV_A_PROFILE_EXIT_AFTER" \
    --save_every "$IV_A_PROFILE_EXIT_AFTER" \
    --grad_clipping 1.0 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$IV_A_WORKERS" \
    --record_attn_metrics \
    --disable_final_evaluation \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --save_dir "$save_dir" \
    --name "iv_a_profile_${location}_seed_${IV_A_SEED}" \
    --fi_nvbit_enable \
    --fi_nvbit_location "$location" \
    --fi_nvbit_steps "$IV_A_PROFILE_TRIGGER_STEP" \
    --fi_nvbit_target_funcs -1 \
    --fi_nvbit_duration 1 \
    2>&1 | tee "$log_file"
done

CURRENT_RUN="extract_kernel_manifest"
extract_args=()
for log_file in "${profile_logs[@]}"; do
  extract_args+=(--log "$log_file")
done
python scripts/experiments/extract_iv_a_kernels.py "${extract_args[@]}" --output "$IV_A_KERNEL_MANIFEST" --seed "$IV_A_INSTR_SELECTION_SEED"

echo
echo "[IV-A] Kernel profiling complete. Review: $IV_A_KERNEL_MANIFEST"
echo "[IV-A] Treat the manifest as the evidence for how many FP/BP kernels exist on this local stack."
echo "[IV-A] Review kernel names and launch counts before starting the full bit-13 sweep."
