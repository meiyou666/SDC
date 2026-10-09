#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_f.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[IV-F][ERROR] ${CURRENT_RUN} failed with exit code ${status}. Check its log before resuming." >&2' ERR

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${IVF_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/optimizer.pt" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" ]]
}

[[ ${#IVF_SEEDS[@]} -eq 3 ]] || { echo "[IV-F][ERROR] IVF_SEEDS must contain exactly 3 seeds." >&2; exit 1; }
[[ "$IVF_CONFIG_REVISION" -ge 3 ]] || { echo "[IV-F][ERROR] Old IV-F config loaded; revision 3 or newer is required." >&2; exit 1; }
[[ "$IVF_TARGET_INSTR" -eq 312 ]] || { echo "[IV-F][ERROR] Revision 3 requires the v3-verified TARGET_INSTR=312." >&2; exit 1; }
[[ "$IVF_BATCH_SIZE" -eq 256 && "$IVF_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[IV-F][ERROR] Expected paper 60M batch structure: batch_size=256,total_batch_size=512." >&2; exit 1; }
[[ "$IVF_WARMUP_STEPS" -eq "$IVF_EXIT_AFTER" ]] || { echo "[IV-F][ERROR] The 1,000-step campaign must include exactly 1,000 warmup steps." >&2; exit 1; }
[[ "$IVF_EVAL_EVERY" -gt "$IVF_EXIT_AFTER" ]] || { echo "[IV-F][ERROR] IVF_EVAL_EVERY must exceed IVF_EXIT_AFTER so validation runs only once at the end." >&2; exit 1; }
command -v python >/dev/null || { echo "[IV-F][ERROR] python is not available; activate the project virtualenv first." >&2; exit 1; }

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
[[ -f "$NVBIT_SO" ]] || { echo "[IV-F][ERROR] NVBit tool not built: $NVBIT_SO" >&2; exit 1; }
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[IV-F][ERROR] fault_injection.cu is newer than fault_injection.so; rebuild the NVBit tool first." >&2
  exit 1
fi

for seed in "${IVF_SEEDS[@]}"; do
  baseline_dir="$IVF_CHECKPOINT_ROOT/baseline/seed_${seed}"
  is_complete "$baseline_dir" || {
    echo "[IV-F][ERROR] Missing complete baseline for seed=$seed. Run run_iv_f_baselines.sh first." >&2
    exit 1
  }
done

export TOOL_VERBOSE="${IVF_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$IVF_TARGET_REGISTER"
export TARGET_OP="$IVF_TARGET_OP"
export TARGET_SMID="$IVF_TARGET_SMID"
export TARGET_LANEID="$IVF_TARGET_LANEID"
export TARGET_BITMASK="$IVF_TARGET_BITMASK"
export TARGET_FUNC="$IVF_TARGET_FUNC"
export TARGET_INSTR="$IVF_TARGET_INSTR"
export TARGET_FUNC_CONTAINS="$IVF_TARGET_FUNC_CONTAINS"
export PYTHONUNBUFFERED=1

mkdir -p "$IVF_CHECKPOINT_ROOT/rate" "$IVF_CHECKPOINT_ROOT/duration"
mkdir -p "$IVF_LOG_ROOT/rate" "$IVF_LOG_ROOT/duration"

echo "============================================================"
echo "IV-F temporal campaign: 15 FI runs (groups 4-18 of 18)"
echo "Rate: every 1 and every 100 steps; duration fixed to 1"
echo "Duration: 1, 3, 5 consecutive steps; first trigger at Step 500"
echo "Seeds: ${IVF_SEEDS[*]}"
echo "Config revision: $IVF_CONFIG_REVISION"
echo "Warmup/training updates: ${IVF_WARMUP_STEPS}/${IVF_EXIT_AFTER}"
echo "Validation: final evaluation only (not counted as a training step)"
echo "Bit position: 13 (bitmask=$TARGET_BITMASK), backward pass"
echo "Kernel filter: $TARGET_FUNC_CONTAINS"
echo "Instruction selector: $TARGET_INSTR"
echo "Execution order: duration 1/3/5, rate 1/100, then rate 1"
echo "============================================================"

fi_index=0
run_one() {
  local family="$1"
  local setting="$2"
  local seed="$3"
  local trigger_rate="$4"
  local duration="$5"
  local explicit_step="$6"

  fi_index=$((fi_index + 1))
  local overall_index=$((fi_index + 3))
  local run_name="iv_f_${family}_${setting}_seed_${seed}"
  local save_dir="$IVF_CHECKPOINT_ROOT/$family/$setting/seed_${seed}"
  local log_file="$IVF_LOG_ROOT/$family/${setting}_seed_${seed}.log"
  local baseline_dir="$IVF_CHECKPOINT_ROOT/baseline/seed_${seed}"
  CURRENT_RUN="$run_name"

  if is_complete "$save_dir"; then
    echo "[IV-F][$overall_index/18] SKIP completed: $run_name"
    return
  fi
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[IV-F][ERROR] Partial output already exists for $run_name:" >&2
    echo "  $save_dir" >&2
    echo "  $log_file" >&2
    echo "Move or rename the partial output before resuming; it will not be overwritten." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$save_dir")" "$(dirname "$log_file")"

  local -a schedule_args
  if [[ "$explicit_step" == "-1" ]]; then
    schedule_args=(--fi_nvbit_trigger_rate "$trigger_rate" --fi_nvbit_steps -1)
  else
    schedule_args=(--fi_nvbit_steps "$explicit_step")
  fi

  echo
  echo "[IV-F][$overall_index/18] START $run_name"
  echo "Schedule: family=$family setting=$setting rate=$trigger_rate duration=$duration explicit_step=$explicit_step"
  echo "Checkpoint: $save_dir/model_${IVF_EXIT_AFTER}"
  echo "Log: $log_file"

  # Scope LD_PRELOAD to the trainer only. Exporting it globally would also
  # load the NVBit tool into tee, mkdir, and the summary process.
  LD_PRELOAD="$NVBIT_SO" python torchrun_main.py \
    --single_gpu \
    --seed "$seed" \
    --model_config "configs/${IVF_MODEL}.json" \
    --max_length "$IVF_MAX_LENGTH" \
    --lr "$IVF_LR" \
    --scheduler cosine \
    --batch_size "$IVF_BATCH_SIZE" \
    --total_batch_size "$IVF_TOTAL_BATCH_SIZE" \
    --num_training_steps "$IVF_TRAINING_SCHEDULE_STEPS" \
    --warmup_steps "$IVF_WARMUP_STEPS" \
    --eval_every "$IVF_EVAL_EVERY" \
    --exit_after "$IVF_EXIT_AFTER" \
    --save_every "$IVF_SAVE_EVERY" \
    --grad_clipping 1.0 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$IVF_WORKERS" \
    --record_attn_metrics \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --base_model_path "$baseline_dir" \
    --compare_every "$IVF_EXIT_AFTER" \
    --save_dir "$save_dir" \
    --name "$run_name" \
    --fi_nvbit_enable \
    --fi_nvbit_location backward \
    --fi_nvbit_target_funcs -1 \
    --fi_nvbit_duration "$duration" \
    "${schedule_args[@]}" \
    2>&1 | tee "$log_file"

  is_complete "$save_dir" || { echo "[IV-F][ERROR] $run_name ended without a complete checkpoint." >&2; exit 1; }
  echo "[IV-F][$overall_index/18] DONE $run_name"
}

# Fault-duration subexperiment: 3 durations x 3 seeds = 9 runs.
for duration in 1 3 5; do
  setting="steps_${duration}"
  for seed in "${IVF_SEEDS[@]}"; do
    run_one duration "$setting" "$seed" 0 "$duration" 500
  done
done

# Fault-rate subexperiment: 2 rates x 3 seeds = 6 runs. Run the expensive
# every-step condition last so cheaper runs validate the campaign first.
for rate in 100 1; do
  setting="every_${rate}"
  for seed in "${IVF_SEEDS[@]}"; do
    run_one rate "$setting" "$seed" "$rate" 1 -1
  done
done

CURRENT_RUN="temporal summary"
python scripts/experiments/summarize_iv_f.py \
  --checkpoint-root "$IVF_CHECKPOINT_ROOT" \
  --output "$IVF_LOG_ROOT/iv_f_summary.csv" \
  --aggregate-output "$IVF_LOG_ROOT/iv_f_aggregate.csv" \
  --best-seed-file "$IVF_CHECKPOINT_ROOT/baseline/best_seed.txt" \
  --expected-step "$IVF_EXIT_AFTER"

echo
echo "[IV-F] All 18 runs (3 baseline + 15 fault injection) are complete."
echo "[IV-F] Summary: $IVF_LOG_ROOT/iv_f_summary.csv"
echo "[IV-F] Aggregate mean/std: $IVF_LOG_ROOT/iv_f_aggregate.csv"
