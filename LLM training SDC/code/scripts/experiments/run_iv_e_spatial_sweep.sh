#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_a.sh
source scripts/configs/iv_b.sh
source scripts/configs/iv_e.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-E][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/experiments/run_iv_e_spatial_sweep.sh lane
  bash scripts/experiments/run_iv_e_spatial_sweep.sh sm

Runs one part of the reduced IV-E spatial-effects sweep (16 runs per part).
The lane part fixes TARGET_SMID and sweeps 16 lanes with stride 2 over
0..31; the sm part fixes TARGET_LANEID and sweeps 16 SMs with stride 8 over
0..127 (128 SMs confirmed on the local RTX 4090). Both parts inject bit 13
into the BP9 kernel (all matching HMMA instructions) during the backward
pass with the standard 1/10-step, duration-1 schedule.
EOF
}

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${IV_E_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

require_file() {
  [[ -f "$1" ]] || { echo "[IV-E][ERROR] Required file not found: $1" >&2; exit 1; }
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

part="${1:-}"
if [[ "$part" == "-h" || "$part" == "--help" ]]; then
  usage
  exit 0
fi
case "$part" in
  lane|sm) ;;
  *) usage; exit 1 ;;
esac

[[ "$IV_E_CONFIG_REVISION" -ge 1 ]] || { echo "[IV-E][ERROR] IV-E config revision 1 or newer is required." >&2; exit 1; }
[[ "$IV_A_CONFIG_REVISION" -ge 6 ]] || { echo "[IV-E][ERROR] IV-A config revision 6 or newer is required." >&2; exit 1; }
[[ "$IV_B_CONFIG_REVISION" -ge 2 ]] || { echo "[IV-E][ERROR] IV-B config revision 2 or newer is required." >&2; exit 1; }
[[ "$IV_E_BATCH_SIZE" -eq 256 && "$IV_E_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-E][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IV_E_WARMUP_STEPS" -eq "$IV_E_EXIT_AFTER" ]] || { echo "[IV-E][ERROR] Expected 1,000 warmup steps in 1,000 training updates." >&2; exit 1; }
[[ "$IV_E_EVAL_EVERY" -gt "$IV_E_EXIT_AFTER" ]] || { echo "[IV-E][ERROR] Validation must run only once after training." >&2; exit 1; }
[[ "$IV_E_KEEP_OPTIMIZER_PT" == "0" || "$IV_E_KEEP_OPTIMIZER_PT" == "1" ]] || { echo "[IV-E][ERROR] IV_E_KEEP_OPTIMIZER_PT must be 0 or 1." >&2; exit 1; }
[[ "$IV_E_BIT" -eq 13 && "$IV_E_TARGET_BITMASK" -eq 8192 ]] || { echo "[IV-E][ERROR] IV-E must use the paper bit position 13 / bitmask 8192." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-E][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${IV_E_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$IV_E_KERNEL_MANIFEST"
require_file "$IV_E_BASELINE_DIR/metrics.jsonl"
require_file "$IV_E_BASELINE_DIR/model_${IV_E_EXIT_AFTER}/model.safetensors"

expected_header=$'location\tkernel_label\tfunction_id_first_seen\tlaunch_count\thmma_instr_count\thmma_instr_indexes\ttarget_instr\tkernel_name\ttarget_func_contains'
actual_header="$(head -n 1 "$IV_E_KERNEL_MANIFEST")"
[[ "$actual_header" == "$expected_header" ]] || {
  echo "[IV-E][ERROR] Unexpected kernel manifest header. Regenerate it with profile_iv_a_kernels.sh." >&2
  exit 1
}
manifest_sha256="$(python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$IV_E_KERNEL_MANIFEST")"

# Resolve the fixed BP9 row from the manifest; the kernel filter must come from
# the manifest so it stays consistent with the profiling evidence.
kernel_row="$(awk -F'\t' -v label="$IV_E_KERNEL_LABEL" 'NR > 1 && $2 == label { print; exit }' "$IV_E_KERNEL_MANIFEST")"
[[ -n "$kernel_row" ]] || { echo "[IV-E][ERROR] Kernel label $IV_E_KERNEL_LABEL not found in $IV_E_KERNEL_MANIFEST" >&2; exit 1; }
kernel_location="$(cut -f1 <<<"$kernel_row")"
kernel_name="$(cut -f8 <<<"$kernel_row")"
kernel_func_contains="$(cut -f9 <<<"$kernel_row")"
hmma_instr_count="$(cut -f5 <<<"$kernel_row")"
[[ "$kernel_location" == "$IV_E_LOCATION" ]] || { echo "[IV-E][ERROR] Manifest location for $IV_E_KERNEL_LABEL is $kernel_location, expected $IV_E_LOCATION." >&2; exit 1; }
[[ -n "$kernel_func_contains" ]] || { echo "[IV-E][ERROR] Manifest row for $IV_E_KERNEL_LABEL has no target_func_contains." >&2; exit 1; }
is_positive_integer "$hmma_instr_count" || { echo "[IV-E][ERROR] Invalid HMMA instruction count for $IV_E_KERNEL_LABEL: $hmma_instr_count" >&2; exit 1; }

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-E][ERROR] NVBit source is newer than the .so. Rebuild it before running IV-E." >&2
  exit 1
fi

# Build the sampled value list for the selected part and verify the count.
values=()
if [[ "$part" == "lane" ]]; then
  for value in $(seq "$IV_E_LANE_START" "$IV_E_LANE_STRIDE" "$IV_E_LANE_END"); do
    values+=("$value")
  done
else
  for value in $(seq "$IV_E_SM_START" "$IV_E_SM_STRIDE" "$IV_E_SM_END"); do
    values+=("$value")
  done
fi
[[ "${#values[@]}" -eq "$IV_E_RUNS_PER_PART" ]] || {
  echo "[IV-E][ERROR] Expected $IV_E_RUNS_PER_PART $part runs, got ${#values[@]} (stride/config mismatch)." >&2
  exit 1
}

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export TOOL_VERBOSE="${IV_E_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$IV_E_TARGET_REGISTER"
export TARGET_OP="$IV_E_TARGET_OP"
export TARGET_FUNC="$IV_E_TARGET_FUNC"
export PYTHONUNBUFFERED=1

part_checkpoint_dir="$IV_E_CHECKPOINT_ROOT/seed_${IV_E_SEED}/$part"
part_log_dir="$IV_E_LOG_ROOT/seed_${IV_E_SEED}/$part"
part_manifest="$part_checkpoint_dir/part_manifest.txt"
mkdir -p "$part_checkpoint_dir" "$part_log_dir"

if [[ "$part" == "lane" ]]; then
  fixed_key=smid
  fixed_value="$IV_E_LANE_FIXED_SMID"
  varied_key=laneid
  value_list="$(printf '%s,' "${values[@]}")"
  value_list="${value_list%,}"
  expected_manifest=$(cat <<EOF
campaign=iv_e_spatial_effects
part=lane
seed=$IV_E_SEED
bit=$IV_E_BIT
bitmask=$IV_E_TARGET_BITMASK
kernel_label=$IV_E_KERNEL_LABEL
fixed_smid=$IV_E_LANE_FIXED_SMID
lane_stride=$IV_E_LANE_STRIDE
lanes=$value_list
runs=${#values[@]}
trigger_rate=$IV_E_TRIGGER_RATE
duration=$IV_E_DURATION
target_register=$IV_E_TARGET_REGISTER
target_op=$IV_E_TARGET_OP
target_func=$IV_E_TARGET_FUNC
target_instr=-1_all_matching_hmma
manifest=$IV_E_KERNEL_MANIFEST
manifest_sha256=$manifest_sha256
EOF
)
else
  fixed_key=laneid
  fixed_value="$IV_E_SM_FIXED_LANEID"
  varied_key=smid
  value_list="$(printf '%s,' "${values[@]}")"
  value_list="${value_list%,}"
  expected_manifest=$(cat <<EOF
campaign=iv_e_spatial_effects
part=sm
seed=$IV_E_SEED
bit=$IV_E_BIT
bitmask=$IV_E_TARGET_BITMASK
kernel_label=$IV_E_KERNEL_LABEL
fixed_laneid=$IV_E_SM_FIXED_LANEID
sm_count=$IV_E_SM_COUNT
sm_stride=$IV_E_SM_STRIDE
sms=$value_list
runs=${#values[@]}
trigger_rate=$IV_E_TRIGGER_RATE
duration=$IV_E_DURATION
target_register=$IV_E_TARGET_REGISTER
target_op=$IV_E_TARGET_OP
target_func=$IV_E_TARGET_FUNC
target_instr=-1_all_matching_hmma
manifest=$IV_E_KERNEL_MANIFEST
manifest_sha256=$manifest_sha256
EOF
)
fi
if [[ -f "$part_manifest" ]]; then
  recorded_manifest="$(cat "$part_manifest")"
  [[ "$recorded_manifest" == "$expected_manifest" ]] || {
    echo "[IV-E][ERROR] Existing part manifest does not match current configuration: $part_manifest" >&2
    exit 1
  }
else
  printf '%s\n' "$expected_manifest" > "$part_manifest"
fi

echo "============================================================"
echo "IV-E spatial effects: $part sweep (${#values[@]} runs)"
echo "Fixed seed: $IV_E_SEED"
echo "Bit position: $IV_E_BIT (bitmask=$IV_E_TARGET_BITMASK)"
echo "Kernel: $IV_E_KERNEL_LABEL ($kernel_name)"
echo "Target contains: $kernel_func_contains"
echo "HMMA instr count: $hmma_instr_count; TARGET_INSTR=-1 (all matching HMMA)"
echo "Fixed $fixed_key=$fixed_value; sweeping $varied_key: ${values[*]}"
echo "Location: $IV_E_LOCATION"
echo "Injection rate: average 1/${IV_E_TRIGGER_RATE} updates; duration=${IV_E_DURATION}"
if [[ "$IV_E_SEED" -eq 42 ]]; then
  echo "Expected seed-42 schedule: 91 triggered updates, first Step 2, last Step 1000"
fi
echo "Checkpoint root: $part_checkpoint_dir"
echo "Log root: $part_log_dir"
echo "============================================================"

failed_runs=()
run_index=0
for value in "${values[@]}"; do
  run_index=$((run_index + 1))
  if [[ "$part" == "lane" ]]; then
    value_label=$(printf 'lane_%02d' "$value")
  else
    value_label=$(printf 'sm_%03d' "$value")
  fi
  run_name="iv_e_${part}_${value_label}_seed_${IV_E_SEED}"
  save_dir="$part_checkpoint_dir/$value_label"
  log_file="$part_log_dir/${value_label}.log"
  CURRENT_RUN="$run_name"

  if is_complete "$save_dir"; then
    echo "[IV-E][$run_index/${#values[@]}] SKIP completed $varied_key=$value"
    continue
  fi
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[IV-E][$run_index/${#values[@]}] WARN partial output exists; not overwriting $varied_key=$value" >&2
    failed_runs+=("${value_label}:partial")
    continue
  fi

  echo
  echo "[IV-E][$run_index/${#values[@]}] START $varied_key=$value (fixed $fixed_key=$fixed_value)"
  echo "Checkpoint: $save_dir/model_${IV_E_EXIT_AFTER}"
  echo "Log: $log_file"

  if [[ "$part" == "lane" ]]; then
    run_env=(TARGET_BITMASK="$IV_E_TARGET_BITMASK" TARGET_FUNC_CONTAINS="$kernel_func_contains" TARGET_INSTR="$IV_E_TARGET_INSTR" TARGET_SMID="$IV_E_LANE_FIXED_SMID" TARGET_LANEID="$value")
  else
    run_env=(TARGET_BITMASK="$IV_E_TARGET_BITMASK" TARGET_FUNC_CONTAINS="$kernel_func_contains" TARGET_INSTR="$IV_E_TARGET_INSTR" TARGET_SMID="$value" TARGET_LANEID="$IV_E_SM_FIXED_LANEID")
  fi

  trap - ERR
  set +e
  env "${run_env[@]}" LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
    --single_gpu \
    --seed "$IV_E_SEED" \
    --model_config "configs/${IV_E_MODEL}.json" \
    --max_length "$IV_E_MAX_LENGTH" \
    --lr "$IV_E_LR" \
    --scheduler cosine \
    --batch_size "$IV_E_BATCH_SIZE" \
    --total_batch_size "$IV_E_TOTAL_BATCH_SIZE" \
    --num_training_steps "$IV_E_TRAINING_SCHEDULE_STEPS" \
    --warmup_steps "$IV_E_WARMUP_STEPS" \
    --eval_every "$IV_E_EVAL_EVERY" \
    --exit_after "$IV_E_EXIT_AFTER" \
    --save_every "$IV_E_SAVE_EVERY" \
    --grad_clipping 1.0 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$IV_E_WORKERS" \
    --record_attn_metrics \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --save_dir "$save_dir" \
    --name "$run_name" \
    --fi_nvbit_enable \
    --fi_nvbit_location "$IV_E_LOCATION" \
    --fi_nvbit_trigger_rate "$IV_E_TRIGGER_RATE" \
    --fi_nvbit_target_funcs -1 \
    --fi_nvbit_duration "$IV_E_DURATION" \
    --fi_nvbit_steps -1 \
    2>&1 | tee "$log_file"
  trainer_status=${PIPESTATUS[0]}
  set -e
  trap 'status=$?; echo "[IV-E][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

  if [[ "$trainer_status" -ne 0 ]] || ! is_complete "$save_dir"; then
    mkdir -p "$save_dir"
    printf 'trainer_exit_code=%s\n' "$trainer_status" > "$save_dir/FAILED"
    failed_runs+=("${value_label}:exit_${trainer_status}")
    echo "[IV-E][$run_index/${#values[@]}] FAILED $varied_key=$value; continuing with the next value." >&2
    continue
  fi
  if [[ "$IV_E_KEEP_OPTIMIZER_PT" == "0" ]]; then
    rm -f -- "$save_dir/model_${IV_E_EXIT_AFTER}/optimizer.pt"
    echo "[IV-E][$run_index/${#values[@]}] Removed optimizer.pt (not needed for offline analysis)."
  fi
  echo "[IV-E][$run_index/${#values[@]}] DONE $varied_key=$value"
done

summary_args=(
  --checkpoint-root "$part_checkpoint_dir"
  --part "$part"
  --baseline-dir "$IV_E_BASELINE_DIR"
  --compute-parameter-difference
  --output "$part_log_dir/iv_e_${part}_summary_raw.csv"
  --expected-step "$IV_E_EXIT_AFTER"
)
if [[ "$IV_E_SEED" -eq 42 ]]; then
  summary_args+=(
    --expected-trigger-count 91
    --expected-first-trigger 2
    --expected-last-trigger 1000
  )
fi

CURRENT_RUN="IV-E $part summary"
python scripts/experiments/summarize_iv_e.py "${summary_args[@]}"

echo
if [[ ${#failed_runs[@]} -eq 0 ]]; then
  echo "[IV-E] All $part sweep runs completed (${#values[@]} runs)."
else
  echo "[IV-E] $part sweep finished with ${#failed_runs[@]} failed/partial run(s): ${failed_runs[*]}" >&2
  echo "[IV-E] Completed runs and the raw summary were preserved." >&2
  exit 1
fi
echo "[IV-E] Raw summary: $part_log_dir/iv_e_${part}_summary_raw.csv"
