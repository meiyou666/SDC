#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh
source scripts/configs/iv_f_duration_all_hmma.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-F-DURATION][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/experiments/run_iv_f_duration_all_hmma.sh probe BP8
  bash scripts/experiments/run_iv_f_duration_all_hmma.sh main BP8
  bash scripts/experiments/run_iv_f_duration_all_hmma.sh full BP8

Modes:
  probe    Run duration=3 for seed 42. This is the minimum run needed for a
           Fig.3-style loss/Rt time-series sanity check.
  main     Run the paper duration axis 1,3,5,7,9 for seed 42.
  full     Run the same duration axis for all configured seeds. Matching mb256
           baselines must already exist under checkpoints/iv_f_mb256/baseline.

The kernel label must exist in logs/iv_a_mb256/kernel_manifest.tsv and must be a
backward kernel (BP*). Injection uses bit 13, TARGET_REGISTER=1, TARGET_INSTR=-1.
EOF
}

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${IVF_DURATION_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

require_file() {
  [[ -f "$1" ]] || { echo "[IV-F-DURATION][ERROR] Required file not found: $1" >&2; exit 1; }
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
  probe)
    durations=("${IVF_DURATION_PROBE_VALUES[@]}")
    seeds=("${IVF_DURATION_MAIN_SEEDS[@]}")
    ;;
  main)
    durations=("${IVF_DURATION_MAIN_VALUES[@]}")
    seeds=("${IVF_DURATION_MAIN_SEEDS[@]}")
    ;;
  full)
    durations=("${IVF_DURATION_FULL_VALUES[@]}")
    seeds=("${IVF_DURATION_FULL_SEEDS[@]}")
    ;;
  *) usage; exit 1 ;;
esac

[[ "$IV_A_CONFIG_REVISION" -ge 6 ]] || { echo "[IV-F-DURATION][ERROR] IV-A config revision 6 or newer is required." >&2; exit 1; }
[[ "$IVF_DURATION_CONFIG_REVISION" -ge 1 ]] || { echo "[IV-F-DURATION][ERROR] IV-F duration config revision 1 or newer is required." >&2; exit 1; }
[[ "$IVF_DURATION_BIT" -eq 13 ]] || { echo "[IV-F-DURATION][ERROR] This rerun is fixed to bit 13." >&2; exit 1; }
[[ "$IVF_DURATION_TARGET_INSTR" -eq -1 ]] || { echo "[IV-F-DURATION][ERROR] TARGET_INSTR must be -1 for all matching HMMA instructions." >&2; exit 1; }
[[ "$IVF_DURATION_TRIGGER_STEP" -ge 1 && "$IVF_DURATION_TRIGGER_STEP" -le "$IVF_DURATION_EXIT_AFTER" ]] || { echo "[IV-F-DURATION][ERROR] Trigger step must be inside the training window." >&2; exit 1; }
[[ "$IVF_DURATION_BATCH_SIZE" -eq 256 && "$IVF_DURATION_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-F-DURATION][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IVF_DURATION_WARMUP_STEPS" -eq "$IVF_DURATION_EXIT_AFTER" ]] || { echo "[IV-F-DURATION][ERROR] Expected 1,000 warmup steps in 1,000 training updates." >&2; exit 1; }
[[ "$IVF_DURATION_EVAL_EVERY" -gt "$IVF_DURATION_EXIT_AFTER" ]] || { echo "[IV-F-DURATION][ERROR] Validation must run only once after training." >&2; exit 1; }
[[ "$IVF_DURATION_KEEP_OPTIMIZER_PT" == "0" || "$IVF_DURATION_KEEP_OPTIMIZER_PT" == "1" ]] || { echo "[IV-F-DURATION][ERROR] IVF_DURATION_KEEP_OPTIMIZER_PT must be 0 or 1." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-F-DURATION][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${IVF_DURATION_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$IVF_DURATION_KERNEL_MANIFEST"

expected_header=$'location\tkernel_label\tfunction_id_first_seen\tlaunch_count\thmma_instr_count\thmma_instr_indexes\ttarget_instr\tkernel_name\ttarget_func_contains'
actual_header="$(head -n 1 "$IVF_DURATION_KERNEL_MANIFEST")"
[[ "$actual_header" == "$expected_header" ]] || {
  echo "[IV-F-DURATION][ERROR] Unexpected kernel manifest header. Regenerate it with profile_iv_a_kernels.sh." >&2
  exit 1
}

manifest_row="$(awk -F '\t' -v label="$kernel_label" '$2 == label {print; exit}' "$IVF_DURATION_KERNEL_MANIFEST")"
[[ -n "$manifest_row" ]] || { echo "[IV-F-DURATION][ERROR] Kernel label not found in manifest: $kernel_label" >&2; exit 1; }
IFS=$'\t' read -r location selected_label function_id launch_count hmma_instr_count hmma_instr_indexes manifest_target_instr kernel_name target_func_contains <<< "$manifest_row"
[[ "$location" == "backward" ]] || { echo "[IV-F-DURATION][ERROR] IV-F duration rerun expects a backward kernel label (BP*), got $kernel_label with location=$location." >&2; exit 1; }
[[ -n "$target_func_contains" ]] || { echo "[IV-F-DURATION][ERROR] Manifest row has no target_func_contains for $kernel_label." >&2; exit 1; }
[[ "$hmma_instr_count" =~ ^[1-9][0-9]*$ ]] || { echo "[IV-F-DURATION][ERROR] Invalid HMMA instruction count for $kernel_label: $hmma_instr_count" >&2; exit 1; }
manifest_sha256="$(python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$IVF_DURATION_KERNEL_MANIFEST")"

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-F-DURATION][ERROR] NVBit source is newer than the .so. Rebuild it before running this campaign." >&2
  exit 1
fi

for seed in "${seeds[@]}"; do
  baseline_dir="$IVF_DURATION_BASELINE_ROOT/seed_${seed}"
  require_file "$baseline_dir/metrics.jsonl"
  require_file "$baseline_dir/model_${IVF_DURATION_EXIT_AFTER}/model.safetensors"
done

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export TOOL_VERBOSE="${IVF_DURATION_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$IVF_DURATION_TARGET_REGISTER"
export TARGET_OP="$IVF_DURATION_TARGET_OP"
export TARGET_SMID="$IVF_DURATION_TARGET_SMID"
export TARGET_LANEID="$IVF_DURATION_TARGET_LANEID"
export TARGET_BITMASK="$IVF_DURATION_TARGET_BITMASK"
export TARGET_FUNC="$IVF_DURATION_TARGET_FUNC"
export PYTHONUNBUFFERED=1

safe_label="$(safe_name "$kernel_label")"
kernel_root="$IVF_DURATION_CHECKPOINT_ROOT/$safe_label"
kernel_log_root="$IVF_DURATION_LOG_ROOT/$safe_label"
campaign_manifest="$kernel_root/campaign_manifest.txt"
mkdir -p "$kernel_root" "$kernel_log_root"

expected_manifest=$(cat <<EOF
campaign=iv_f_duration_all_hmma
bit=$IVF_DURATION_BIT
bitmask=$IVF_DURATION_TARGET_BITMASK
kernel_label=$kernel_label
location=$location
target_func_contains=$target_func_contains
hmma_instr_count=$hmma_instr_count
target_register=$IVF_DURATION_TARGET_REGISTER
target_smid=$IVF_DURATION_TARGET_SMID
target_laneid=$IVF_DURATION_TARGET_LANEID
target_op=$IVF_DURATION_TARGET_OP
target_func=$IVF_DURATION_TARGET_FUNC
target_instr=-1_all_matching_hmma
trigger_step=$IVF_DURATION_TRIGGER_STEP
duration_values=${IVF_DURATION_FULL_VALUES[*]}
manifest=$IVF_DURATION_KERNEL_MANIFEST
manifest_sha256=$manifest_sha256
baseline_root=$IVF_DURATION_BASELINE_ROOT
EOF
)
if [[ -f "$campaign_manifest" ]]; then
  recorded_manifest="$(cat "$campaign_manifest")"
  [[ "$recorded_manifest" == "$expected_manifest" ]] || {
    echo "[IV-F-DURATION][ERROR] Existing campaign manifest does not match current configuration: $campaign_manifest" >&2
    exit 1
  }
else
  printf '%s\n' "$expected_manifest" > "$campaign_manifest"
fi

total_runs=$((${#durations[@]} * ${#seeds[@]}))
echo "============================================================"
echo "IV-F duration all-HMMA rerun: mode=$mode kernel=$kernel_label"
echo "Seeds: ${seeds[*]}"
echo "Durations: ${durations[*]} consecutive update steps"
echo "Trigger step: $IVF_DURATION_TRIGGER_STEP"
echo "Bit position: $IVF_DURATION_BIT (bitmask=$TARGET_BITMASK)"
echo "Kernel: $kernel_name"
echo "Target contains: $target_func_contains"
echo "HMMA instr count: $hmma_instr_count; TARGET_INSTR=-1 (all matching HMMA)"
echo "Injection target: location=backward SM=$TARGET_SMID lane=$TARGET_LANEID register=$TARGET_REGISTER"
echo "Baseline root: $IVF_DURATION_BASELINE_ROOT"
echo "Checkpoint root: $kernel_root"
echo "Log root: $kernel_log_root"
echo "============================================================"

run_index=0
for duration in "${durations[@]}"; do
  setting="steps_${duration}"
  for seed in "${seeds[@]}"; do
    run_index=$((run_index + 1))
    run_name="iv_f_duration_all_hmma_${kernel_label}_${setting}_seed_${seed}"
    save_dir="$kernel_root/duration/$setting/seed_${seed}"
    log_file="$kernel_log_root/duration/${setting}_seed_${seed}.log"
    baseline_dir="$IVF_DURATION_BASELINE_ROOT/seed_${seed}"
    CURRENT_RUN="$run_name"

    if is_complete "$save_dir"; then
      echo "[IV-F-DURATION][$run_index/$total_runs] SKIP completed: kernel=$kernel_label duration=$duration seed=$seed"
      continue
    fi
    if [[ -e "$save_dir" || -e "$log_file" ]]; then
      echo "[IV-F-DURATION][ERROR] Partial output already exists for kernel=$kernel_label duration=$duration seed=$seed:" >&2
      echo "  $save_dir" >&2
      echo "  $log_file" >&2
      echo "Move or rename the partial output before resuming; it will not be overwritten." >&2
      exit 1
    fi
    mkdir -p "$(dirname "$save_dir")" "$(dirname "$log_file")"

    echo
    echo "[IV-F-DURATION][$run_index/$total_runs] START kernel=$kernel_label duration=$duration seed=$seed"
    echo "Trigger window: $IVF_DURATION_TRIGGER_STEP-$((IVF_DURATION_TRIGGER_STEP + duration - 1))"
    echo "Checkpoint: $save_dir/model_${IVF_DURATION_EXIT_AFTER}"
    echo "Log: $log_file"

    TARGET_FUNC_CONTAINS="$target_func_contains" TARGET_INSTR=-1 LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
      --single_gpu \
      --seed "$seed" \
      --model_config "configs/${IVF_DURATION_MODEL}.json" \
      --max_length "$IVF_DURATION_MAX_LENGTH" \
      --lr "$IVF_DURATION_LR" \
      --scheduler cosine \
      --batch_size "$IVF_DURATION_BATCH_SIZE" \
      --total_batch_size "$IVF_DURATION_TOTAL_BATCH_SIZE" \
      --num_training_steps "$IVF_DURATION_TRAINING_SCHEDULE_STEPS" \
      --warmup_steps "$IVF_DURATION_WARMUP_STEPS" \
      --eval_every "$IVF_DURATION_EVAL_EVERY" \
      --exit_after "$IVF_DURATION_EXIT_AFTER" \
      --save_every "$IVF_DURATION_SAVE_EVERY" \
      --grad_clipping 1.0 \
      --weight_decay 0.01 \
      --dtype bfloat16 \
      --optimizer adamw \
      --workers "$IVF_DURATION_WORKERS" \
      --record_attn_metrics \
      --tokenizer_path "$TOKENIZER_PATH" \
      --train_data_path "$TRAIN_DATA_PATH" \
      --val_data_path "$VAL_DATA_PATH" \
      --base_model_path "$baseline_dir" \
      --compare_every "$IVF_DURATION_EXIT_AFTER" \
      --save_dir "$save_dir" \
      --name "$run_name" \
      --fi_nvbit_enable \
      --fi_nvbit_location backward \
      --fi_nvbit_target_funcs -1 \
      --fi_nvbit_duration "$duration" \
      --fi_nvbit_steps "$IVF_DURATION_TRIGGER_STEP" \
      2>&1 | tee "$log_file"

    is_complete "$save_dir" || { echo "[IV-F-DURATION][ERROR] $run_name ended without a complete checkpoint." >&2; exit 1; }
    if [[ "$IVF_DURATION_KEEP_OPTIMIZER_PT" == "0" ]]; then
      rm -f -- "$save_dir/model_${IVF_DURATION_EXIT_AFTER}/optimizer.pt"
      echo "[IV-F-DURATION][$run_index/$total_runs] Removed optimizer.pt (not needed for offline analysis)."
    fi
    echo "[IV-F-DURATION][$run_index/$total_runs] DONE kernel=$kernel_label duration=$duration seed=$seed"
  done
done

echo
echo "[IV-F-DURATION] Mode $mode complete for kernel=$kernel_label."
echo "[IV-F-DURATION] Outputs: $kernel_root"

CURRENT_RUN="IV-F duration summary"
python scripts/experiments/summarize_iv_f_duration_all_hmma.py \
  --checkpoint-root "$IVF_DURATION_CHECKPOINT_ROOT" \
  --baseline-root "$IVF_DURATION_BASELINE_ROOT" \
  --output "$IVF_DURATION_LOG_ROOT/iv_f_duration_all_hmma_summary.csv" \
  --aggregate-output "$IVF_DURATION_LOG_ROOT/iv_f_duration_all_hmma_aggregate.csv" \
  --fig3-output "$IVF_DURATION_LOG_ROOT/iv_f_duration_all_hmma_fig3_timeseries.csv" \
  --fig3-kernel "$kernel_label" \
  --fig3-duration "$IVF_DURATION_FIG3_DURATION" \
  --fig3-seed "$IVF_DURATION_FIG3_SEED" \
  --window-before "$IVF_DURATION_FIG3_WINDOW_BEFORE" \
  --window-after "$IVF_DURATION_FIG3_WINDOW_AFTER" \
  --trigger-step "$IVF_DURATION_TRIGGER_STEP" \
  --expected-step "$IVF_DURATION_EXIT_AFTER" \
  --compute-parameter-difference

echo "[IV-F-DURATION] Summary: $IVF_DURATION_LOG_ROOT/iv_f_duration_all_hmma_summary.csv"
echo "[IV-F-DURATION] Aggregate: $IVF_DURATION_LOG_ROOT/iv_f_duration_all_hmma_aggregate.csv"
echo "[IV-F-DURATION] Fig.3 time series: $IVF_DURATION_LOG_ROOT/iv_f_duration_all_hmma_fig3_timeseries.csv"
