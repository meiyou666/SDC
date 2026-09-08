#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh
source scripts/configs/iv_b.sh
source scripts/configs/iv_cd.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-CD][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/experiments/run_iv_cd_clipping_comparison.sh [KERNEL_LABEL] [BIT]

Runs a focused IV-D clipping comparison for one backward kernel and one bit.
The clipped condition is reused from the completed IV-B run; this script only
runs the no-clipping condition with --grad_clipping 0.0.

Defaults come from scripts/configs/iv_cd.sh. The kernel label must exist in
logs/iv_a_mb256/kernel_manifest.tsv and must be a backward kernel.
EOF
}

require_file() {
  [[ -f "$1" ]] || { echo "[IV-CD][ERROR] Required file not found: $1" >&2; exit 1; }
}

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${IV_CD_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

is_ivb_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${IV_CD_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

kernel_label="${1:-$IV_CD_DEFAULT_CLIPPING_KERNEL}"
bit="${2:-$IV_CD_DEFAULT_CLIPPING_BIT}"
if [[ "$kernel_label" == "-h" || "$kernel_label" == "--help" ]]; then
  usage
  exit 0
fi
[[ "$bit" =~ ^[0-9]+$ && "$bit" -ge 0 && "$bit" -le 31 ]] || { usage; exit 1; }
bitmask=$((1 << bit))
bit_label=$(printf 'bit_%02d' "$bit")

[[ "$IV_CD_CONFIG_REVISION" -ge 1 ]] || { echo "[IV-CD][ERROR] IV-CD config revision 1 or newer is required." >&2; exit 1; }
[[ "$IV_B_CONFIG_REVISION" -ge 2 ]] || { echo "[IV-CD][ERROR] IV-B config revision 2 or newer is required." >&2; exit 1; }
[[ "$IV_A_CONFIG_REVISION" -ge 6 ]] || { echo "[IV-CD][ERROR] IV-A config revision 6 or newer is required." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-CD][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${IV_CD_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$IV_CD_KERNEL_MANIFEST"
require_file "$IV_CD_BASELINE_DIR/metrics.jsonl"

location=""
target_func_contains=""
kernel_name=""
hmma_instr_count=""
while IFS=$'\t' read -r row_location row_label function_id launch_count row_hmma_instr_count hmma_instr_indexes target_instr row_kernel_name row_target_func_contains; do
  [[ "$row_location" == "location" ]] && continue
  if [[ "$row_label" == "$kernel_label" ]]; then
    location="$row_location"
    hmma_instr_count="$row_hmma_instr_count"
    kernel_name="$row_kernel_name"
    target_func_contains="$row_target_func_contains"
    break
  fi
done < "$IV_CD_KERNEL_MANIFEST"

[[ "$location" == "backward" ]] || { echo "[IV-CD][ERROR] Expected a backward kernel label for clipping comparison, got $kernel_label location=$location" >&2; exit 1; }
[[ -n "$target_func_contains" ]] || { echo "[IV-CD][ERROR] Kernel label not found in manifest: $kernel_label" >&2; exit 1; }

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-CD][ERROR] NVBit source is newer than the .so. Rebuild it before running IV-C/D." >&2
  exit 1
fi

run_root="$IV_CD_CHECKPOINT_ROOT/clipping/seed_${IV_CD_SEED}/${bit_label}/${kernel_label}"
log_root="$IV_CD_LOG_ROOT/clipping/seed_${IV_CD_SEED}/${bit_label}/${kernel_label}"
clipped_dir="$IV_B_CHECKPOINT_ROOT/seed_${IV_CD_SEED}/${bit_label}/${kernel_label}"
mkdir -p "$run_root" "$log_root"
is_ivb_complete "$clipped_dir" || { echo "[IV-CD][ERROR] Completed IV-B clipped run not found: $clipped_dir" >&2; exit 1; }

expected_manifest=$(cat <<EOF
campaign=iv_cd_clipping_comparison
seed=$IV_CD_SEED
kernel_label=$kernel_label
bit=$bit
bitmask=$bitmask
trigger_rate=$IV_CD_TRIGGER_RATE
duration=$IV_CD_DURATION
location=backward
target_func_contains=$target_func_contains
target_instr=-1_all_matching_hmma
target_register=$IV_CD_TARGET_REGISTER
target_smid=$IV_CD_TARGET_SMID
target_laneid=$IV_CD_TARGET_LANEID
target_op=$IV_CD_TARGET_OP
hmma_instr_count=$hmma_instr_count
kernel_name=$kernel_name
EOF
)
manifest_file="$run_root/experiment_manifest.txt"
if [[ -f "$manifest_file" ]]; then
  recorded_manifest="$(cat "$manifest_file")"
  [[ "$recorded_manifest" == "$expected_manifest" ]] || {
    echo "[IV-CD][ERROR] Existing experiment manifest does not match current configuration: $manifest_file" >&2
    exit 1
  }
else
  printf '%s\n' "$expected_manifest" > "$manifest_file"
fi

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export TOOL_VERBOSE="${IV_CD_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$IV_CD_TARGET_REGISTER"
export TARGET_OP="$IV_CD_TARGET_OP"
export TARGET_SMID="$IV_CD_TARGET_SMID"
export TARGET_LANEID="$IV_CD_TARGET_LANEID"
export TARGET_FUNC="$IV_CD_TARGET_FUNC"
export TARGET_FUNC_CONTAINS="$target_func_contains"
export TARGET_INSTR=-1
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "IV-C/D clipping comparison"
echo "Kernel: $kernel_label ($kernel_name)"
echo "Bit: $bit (bitmask=$bitmask)"
echo "TARGET_INSTR=-1, HMMA instr count from manifest=$hmma_instr_count"
echo "Clipped condition reused from: $clipped_dir"
echo "Condition to run: no_clipping"
echo "Checkpoint root: $run_root"
echo "Log root: $log_root"
echo "============================================================"

failed_runs=()
condition=no_clipping
grad_clipping=0.0
save_dir="$run_root/$condition"
log_file="$log_root/${condition}.log"
CURRENT_RUN="iv_cd_${kernel_label}_${bit_label}_${condition}"

if is_complete "$save_dir"; then
  echo "[IV-CD] SKIP completed condition=$condition"
elif [[ -e "$save_dir" || -e "$log_file" ]]; then
  echo "[IV-CD] WARN partial output exists; not overwriting condition=$condition" >&2
  failed_runs+=("${condition}:partial")
else

  echo
  echo "[IV-CD] START condition=$condition grad_clipping=$grad_clipping"
  trap - ERR
  set +e
  TARGET_BITMASK="$bitmask" LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
    --single_gpu \
    --seed "$IV_CD_SEED" \
    --model_config "configs/${IV_CD_MODEL}.json" \
    --max_length "$IV_CD_MAX_LENGTH" \
    --lr "$IV_CD_LR" \
    --scheduler cosine \
    --batch_size "$IV_CD_BATCH_SIZE" \
    --total_batch_size "$IV_CD_TOTAL_BATCH_SIZE" \
    --num_training_steps "$IV_CD_TRAINING_SCHEDULE_STEPS" \
    --warmup_steps "$IV_CD_WARMUP_STEPS" \
    --eval_every "$IV_CD_EVAL_EVERY" \
    --exit_after "$IV_CD_EXIT_AFTER" \
    --save_every "$IV_CD_SAVE_EVERY" \
    --grad_clipping "$grad_clipping" \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$IV_CD_WORKERS" \
    --record_attn_metrics \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --save_dir "$save_dir" \
    --name "iv_cd_${kernel_label}_${bit_label}_${condition}_seed_${IV_CD_SEED}" \
    --fi_nvbit_enable \
    --fi_nvbit_location backward \
    --fi_nvbit_trigger_rate "$IV_CD_TRIGGER_RATE" \
    --fi_nvbit_target_funcs -1 \
    --fi_nvbit_duration "$IV_CD_DURATION" \
    --fi_nvbit_steps -1 \
    2>&1 | tee "$log_file"
  trainer_status=${PIPESTATUS[0]}
  set -e
  trap 'status=$?; echo "[IV-CD][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

  if [[ "$trainer_status" -ne 0 ]] || ! is_complete "$save_dir"; then
    mkdir -p "$save_dir"
    printf 'trainer_exit_code=%s\n' "$trainer_status" > "$save_dir/FAILED"
    failed_runs+=("${condition}:exit_${trainer_status}")
    echo "[IV-CD] FAILED condition=$condition; preserving output." >&2
  else
    if [[ "$IV_CD_KEEP_OPTIMIZER_PT" == "0" ]]; then
      rm -f -- "$save_dir/model_${IV_CD_EXIT_AFTER}/optimizer.pt"
      echo "[IV-CD] Removed optimizer.pt for condition=$condition."
    fi
    echo "[IV-CD] DONE condition=$condition"
  fi
fi

python scripts/experiments/summarize_iv_cd_clipping.py \
  --clipped-dir "$clipped_dir" \
  --no-clipping-dir "$save_dir" \
  --baseline-dir "$IV_CD_BASELINE_DIR" \
  --kernel-label "$kernel_label" \
  --bit "$bit" \
  --output "$log_root/clipping_summary.csv" \
  --expected-step "$IV_CD_EXIT_AFTER"

echo
if [[ ${#failed_runs[@]} -eq 0 ]]; then
  echo "[IV-CD] Clipping comparison completed."
else
  echo "[IV-CD] Finished with ${#failed_runs[@]} failed/partial condition(s): ${failed_runs[*]}" >&2
  exit 1
fi
echo "[IV-CD] Summary: $log_root/clipping_summary.csv"
