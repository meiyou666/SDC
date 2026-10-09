#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh
source scripts/configs/iv_f_rate_all_hmma.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-F-RATE][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/experiments/run_iv_f_rate_all_hmma.sh probe BP8
  bash scripts/experiments/run_iv_f_rate_all_hmma.sh serial BP8
  bash scripts/experiments/run_iv_f_rate_all_hmma.sh full BP8

Modes:
  probe    Run only every-step injection, used as the first stress probe.
  serial   Run every 1, every 100, and every 1000 steps in one serial pass.
  full     Run every 1, 10, 100, and 1000 steps in one serial pass.

The kernel label must exist in logs/iv_a_mb256/kernel_manifest.tsv and must be a
backward kernel (BP*). Injection uses bit 13, TARGET_REGISTER=1, TARGET_INSTR=-1.
EOF
}

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${IVF_RATE_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

require_file() {
  [[ -f "$1" ]] || { echo "[IV-F-RATE][ERROR] Required file not found: $1" >&2; exit 1; }
}

safe_name() {
  local value="$1"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s' "$value"
}

mode="${1:-}"
kernel_label="${2:-}"
if [[ "$mode" == "-h" || "$mode" == "--help" ]]; then
  usage
  exit 0
fi
[[ -n "$mode" && -n "$kernel_label" ]] || { usage; exit 1; }

case "$mode" in
  probe) rates=("${IVF_RATE_PROBE_RATES[@]}") ;;
  serial) rates=("${IVF_RATE_SERIAL_RATES[@]}") ;;
  full) rates=("${IVF_RATE_FULL_RATES[@]}") ;;
  *) usage; exit 1 ;;
esac

[[ "$IV_A_CONFIG_REVISION" -ge 6 ]] || { echo "[IV-F-RATE][ERROR] IV-A config revision 6 or newer is required." >&2; exit 1; }
[[ "$IVF_RATE_CONFIG_REVISION" -ge 1 ]] || { echo "[IV-F-RATE][ERROR] IV-F rate config revision 1 or newer is required." >&2; exit 1; }
[[ "$IVF_RATE_BIT" -eq 13 ]] || { echo "[IV-F-RATE][ERROR] This rerun is fixed to bit 13." >&2; exit 1; }
[[ "$IVF_RATE_TARGET_INSTR" -eq -1 ]] || { echo "[IV-F-RATE][ERROR] TARGET_INSTR must be -1 for all matching HMMA instructions." >&2; exit 1; }
[[ "$IVF_RATE_BATCH_SIZE" -eq 256 && "$IVF_RATE_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-F-RATE][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IVF_RATE_WARMUP_STEPS" -eq "$IVF_RATE_EXIT_AFTER" ]] || { echo "[IV-F-RATE][ERROR] Expected 1,000 warmup steps in 1,000 training updates." >&2; exit 1; }
[[ "$IVF_RATE_EVAL_EVERY" -gt "$IVF_RATE_EXIT_AFTER" ]] || { echo "[IV-F-RATE][ERROR] Validation must run only once after training." >&2; exit 1; }
[[ "$IVF_RATE_KEEP_OPTIMIZER_PT" == "0" || "$IVF_RATE_KEEP_OPTIMIZER_PT" == "1" ]] || { echo "[IV-F-RATE][ERROR] IVF_RATE_KEEP_OPTIMIZER_PT must be 0 or 1." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-F-RATE][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${IVF_RATE_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$IVF_RATE_KERNEL_MANIFEST"
require_file "$IVF_RATE_BASELINE_DIR/metrics.jsonl"
require_file "$IVF_RATE_BASELINE_DIR/model_${IVF_RATE_EXIT_AFTER}/model.safetensors"

expected_header=$'location\tkernel_label\tfunction_id_first_seen\tlaunch_count\thmma_instr_count\thmma_instr_indexes\ttarget_instr\tkernel_name\ttarget_func_contains'
actual_header="$(head -n 1 "$IVF_RATE_KERNEL_MANIFEST")"
[[ "$actual_header" == "$expected_header" ]] || {
  echo "[IV-F-RATE][ERROR] Unexpected kernel manifest header. Regenerate it with profile_iv_a_kernels.sh." >&2
  exit 1
}

manifest_row="$(awk -F '\t' -v label="$kernel_label" '$2 == label {print; exit}' "$IVF_RATE_KERNEL_MANIFEST")"
[[ -n "$manifest_row" ]] || { echo "[IV-F-RATE][ERROR] Kernel label not found in manifest: $kernel_label" >&2; exit 1; }
IFS=$'\t' read -r location selected_label function_id launch_count hmma_instr_count hmma_instr_indexes manifest_target_instr kernel_name target_func_contains <<< "$manifest_row"
[[ "$location" == "backward" ]] || { echo "[IV-F-RATE][ERROR] IV-F rate rerun expects a backward kernel label (BP*), got $kernel_label with location=$location." >&2; exit 1; }
[[ -n "$target_func_contains" ]] || { echo "[IV-F-RATE][ERROR] Manifest row has no target_func_contains for $kernel_label." >&2; exit 1; }
[[ "$hmma_instr_count" =~ ^[1-9][0-9]*$ ]] || { echo "[IV-F-RATE][ERROR] Invalid HMMA instruction count for $kernel_label: $hmma_instr_count" >&2; exit 1; }
manifest_sha256="$(python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$IVF_RATE_KERNEL_MANIFEST")"

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-F-RATE][ERROR] NVBit source is newer than the .so. Rebuild it before running this campaign." >&2
  exit 1
fi

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export TOOL_VERBOSE="${IVF_RATE_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$IVF_RATE_TARGET_REGISTER"
export TARGET_OP="$IVF_RATE_TARGET_OP"
export TARGET_SMID="$IVF_RATE_TARGET_SMID"
export TARGET_LANEID="$IVF_RATE_TARGET_LANEID"
export TARGET_BITMASK="$IVF_RATE_TARGET_BITMASK"
export TARGET_FUNC="$IVF_RATE_TARGET_FUNC"
export PYTHONUNBUFFERED=1

safe_label="$(safe_name "$kernel_label")"
kernel_root="$IVF_RATE_CHECKPOINT_ROOT/$safe_label"
kernel_log_root="$IVF_RATE_LOG_ROOT/$safe_label"
campaign_manifest="$kernel_root/campaign_manifest.txt"
mkdir -p "$kernel_root" "$kernel_log_root"

expected_manifest=$(cat <<EOF
campaign=iv_f_rate_all_hmma
seed=$IVF_RATE_SEED
bit=$IVF_RATE_BIT
bitmask=$IVF_RATE_TARGET_BITMASK
kernel_label=$kernel_label
location=$location
target_func_contains=$target_func_contains
hmma_instr_count=$hmma_instr_count
target_register=$IVF_RATE_TARGET_REGISTER
target_smid=$IVF_RATE_TARGET_SMID
target_laneid=$IVF_RATE_TARGET_LANEID
target_op=$IVF_RATE_TARGET_OP
target_func=$IVF_RATE_TARGET_FUNC
target_instr=-1_all_matching_hmma
duration=$IVF_RATE_DURATION
manifest=$IVF_RATE_KERNEL_MANIFEST
manifest_sha256=$manifest_sha256
baseline_dir=$IVF_RATE_BASELINE_DIR
EOF
)
if [[ -f "$campaign_manifest" ]]; then
  recorded_manifest="$(cat "$campaign_manifest")"
  [[ "$recorded_manifest" == "$expected_manifest" ]] || {
    echo "[IV-F-RATE][ERROR] Existing campaign manifest does not match current configuration: $campaign_manifest" >&2
    exit 1
  }
else
  printf '%s\n' "$expected_manifest" > "$campaign_manifest"
fi

echo "============================================================"
echo "IV-F rate rerun: mode=$mode kernel=$kernel_label"
echo "Seed: $IVF_RATE_SEED"
echo "Bit position: $IVF_RATE_BIT (bitmask=$TARGET_BITMASK)"
echo "Kernel: $kernel_name"
echo "Target contains: $target_func_contains"
echo "HMMA instr count: $hmma_instr_count; TARGET_INSTR=-1 (all matching HMMA)"
echo "Injection target: location=backward SM=$TARGET_SMID lane=$TARGET_LANEID register=$TARGET_REGISTER"
echo "Rates: ${rates[*]}"
echo "Baseline: $IVF_RATE_BASELINE_DIR"
echo "Checkpoint root: $kernel_root"
echo "Log root: $kernel_log_root"
echo "============================================================"

run_index=0
total_runs=${#rates[@]}
for rate in "${rates[@]}"; do
  run_index=$((run_index + 1))
  setting="every_${rate}"
  run_name="iv_f_rate_all_hmma_${kernel_label}_${setting}_seed_${IVF_RATE_SEED}"
  save_dir="$kernel_root/rate/$setting/seed_${IVF_RATE_SEED}"
  log_file="$kernel_log_root/rate/${setting}_seed_${IVF_RATE_SEED}.log"
  CURRENT_RUN="$run_name"

  if is_complete "$save_dir"; then
    echo "[IV-F-RATE][$run_index/$total_runs] SKIP completed: kernel=$kernel_label rate=1/$rate"
    continue
  fi
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[IV-F-RATE][ERROR] Partial output already exists for kernel=$kernel_label rate=1/$rate:" >&2
    echo "  $save_dir" >&2
    echo "  $log_file" >&2
    echo "Move or rename the partial output before resuming; it will not be overwritten." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$save_dir")" "$(dirname "$log_file")"

  echo
  echo "[IV-F-RATE][$run_index/$total_runs] START kernel=$kernel_label rate=1/$rate"
  echo "Checkpoint: $save_dir/model_${IVF_RATE_EXIT_AFTER}"
  echo "Log: $log_file"

  TARGET_FUNC_CONTAINS="$target_func_contains" TARGET_INSTR=-1 LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
    --single_gpu \
    --seed "$IVF_RATE_SEED" \
    --model_config "configs/${IVF_RATE_MODEL}.json" \
    --max_length "$IVF_RATE_MAX_LENGTH" \
    --lr "$IVF_RATE_LR" \
    --scheduler cosine \
    --batch_size "$IVF_RATE_BATCH_SIZE" \
    --total_batch_size "$IVF_RATE_TOTAL_BATCH_SIZE" \
    --num_training_steps "$IVF_RATE_TRAINING_SCHEDULE_STEPS" \
    --warmup_steps "$IVF_RATE_WARMUP_STEPS" \
    --eval_every "$IVF_RATE_EVAL_EVERY" \
    --exit_after "$IVF_RATE_EXIT_AFTER" \
    --save_every "$IVF_RATE_SAVE_EVERY" \
    --grad_clipping 1.0 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$IVF_RATE_WORKERS" \
    --record_attn_metrics \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --base_model_path "$IVF_RATE_BASELINE_DIR" \
    --compare_every "$IVF_RATE_EXIT_AFTER" \
    --save_dir "$save_dir" \
    --name "$run_name" \
    --fi_nvbit_enable \
    --fi_nvbit_location backward \
    --fi_nvbit_trigger_rate "$rate" \
    --fi_nvbit_target_funcs -1 \
    --fi_nvbit_duration "$IVF_RATE_DURATION" \
    --fi_nvbit_steps -1 \
    2>&1 | tee "$log_file"

  is_complete "$save_dir" || { echo "[IV-F-RATE][ERROR] $run_name ended without a complete checkpoint." >&2; exit 1; }
  if [[ "$IVF_RATE_KEEP_OPTIMIZER_PT" == "0" ]]; then
    rm -f -- "$save_dir/model_${IVF_RATE_EXIT_AFTER}/optimizer.pt"
    echo "[IV-F-RATE][$run_index/$total_runs] Removed optimizer.pt (not needed for offline analysis)."
  fi
  echo "[IV-F-RATE][$run_index/$total_runs] DONE kernel=$kernel_label rate=1/$rate"
done

echo
echo "[IV-F-RATE] Mode $mode complete for kernel=$kernel_label."
echo "[IV-F-RATE] Outputs: $kernel_root"

CURRENT_RUN="IV-F rate summary"
python scripts/experiments/summarize_iv_f_rate_all_hmma.py \
  --checkpoint-root "$IVF_RATE_CHECKPOINT_ROOT" \
  --baseline-dir "$IVF_RATE_BASELINE_DIR" \
  --output "$IVF_RATE_LOG_ROOT/iv_f_rate_all_hmma_summary.csv" \
  --expected-step "$IVF_RATE_EXIT_AFTER" \
  --reuse-every10-dir "checkpoints/iv_b_mb256/seed_${IVF_RATE_SEED}/bit_${IVF_RATE_BIT}/${kernel_label}"

echo "[IV-F-RATE] Summary: $IVF_RATE_LOG_ROOT/iv_f_rate_all_hmma_summary.csv"
