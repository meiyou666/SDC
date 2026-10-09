#!/usr/bin/env bash
set -Eeuo pipefail

# Paper Section VII-A (downgraded): detection + recompute at alpha=0.05.
# 60M, seed 42, 2,000 steps, bits 9-12, rate 1/10, duration 1, backward
# location, random backward kernel (BP1-BP9) per trigger, all HMMA
# instructions of the chosen kernel, first input register, SM0/lane0.
#
# Runs:
#   1x fault-free baseline (detection path on)
#   4x FI + recompute (one per bit)
#   4x FI without recompute (one per bit, same trigger schedule shape)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/vi_vii.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[VII-A][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/experiments/run_vii_a_detection.sh

Single mode: runs the full VII-A campaign (baseline + 4 bits x 2 conditions)
serially, skipping runs that are already complete.
EOF
}

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${VI_VII_A_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

require_file() {
  [[ -f "$1" ]] || { echo "[VII-A][ERROR] Required file not found: $1" >&2; exit 1; }
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ "$VI_VII_CONFIG_REVISION" -ge 1 ]] || { echo "[VII-A][ERROR] vi_vii config revision 1 or newer is required." >&2; exit 1; }
[[ "$VI_VII_A_TRIGGER_RATE" -eq 10 ]] || { echo "[VII-A][ERROR] VII-A expects injection rate 1/10." >&2; exit 1; }
[[ "$VI_VII_ALPHA" == "0.05" ]] || { echo "[VII-A][ERROR] VII-A is fixed to alpha=0.05." >&2; exit 1; }
[[ "$VI_VII_TARGET_INSTR" -eq -1 ]] || { echo "[VII-A][ERROR] TARGET_INSTR must be -1 (all matching HMMA)." >&2; exit 1; }
[[ "$VI_VII_BATCH_SIZE" -eq 256 && "$VI_VII_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[VII-A][ERROR] Expected paper 60M batch structure: batch_size=256, total_batch_size=512." >&2; exit 1; }
command -v python >/dev/null || { echo "[VII-A][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${VI_VII_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$VI_VII_KERNEL_MANIFEST"

expected_header=$'location\tkernel_label\tfunction_id_first_seen\tlaunch_count\thmma_instr_count\thmma_instr_indexes\ttarget_instr\tkernel_name\ttarget_func_contains'
actual_header="$(head -n 1 "$VI_VII_KERNEL_MANIFEST")"
[[ "$actual_header" == "$expected_header" ]] || {
  echo "[VII-A][ERROR] Unexpected kernel manifest header. Regenerate it with profile_iv_a_kernels.sh." >&2
  exit 1
}

# Candidate function IDs: all backward kernels (BP*), manifest order.
backward_ids=()
while IFS=$'\t' read -r location kernel_label function_id rest; do
  [[ "$location" == "backward" ]] || continue
  [[ "$kernel_label" =~ ^BP[0-9]+$ ]] || continue
  [[ "$function_id" =~ ^[0-9]+$ ]] || { echo "[VII-A][ERROR] Bad function ID for $kernel_label: $function_id" >&2; exit 1; }
  backward_ids+=("$function_id")
done < <(tail -n +2 "$VI_VII_KERNEL_MANIFEST")
[[ "${#backward_ids[@]}" -ge 2 ]] || { echo "[VII-A][ERROR] Need at least 2 backward kernels in the manifest." >&2; exit 1; }
manifest_sha256="$(python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$VI_VII_KERNEL_MANIFEST")"

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[VII-A][ERROR] NVBit source is newer than the .so. Rebuild it before running this campaign." >&2
  exit 1
fi

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export TOOL_VERBOSE="${VI_VII_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$VI_VII_TARGET_REGISTER"
export TARGET_OP="$VI_VII_TARGET_OP"
export TARGET_SMID="$VI_VII_TARGET_SMID"
export TARGET_LANEID="$VI_VII_TARGET_LANEID"
export TARGET_FUNC=-1
export PYTHONUNBUFFERED=1

baseline_dir="$VI_VII_A_CHECKPOINT_ROOT/baseline/seed_${VI_VII_SEED}"
baseline_log="$VI_VII_A_LOG_ROOT/baseline/seed_${VI_VII_SEED}.log"
mkdir -p "$baseline_dir" "$VI_VII_A_LOG_ROOT"

echo "============================================================"
echo "VII-A detection + recompute: alpha=$VI_VII_ALPHA seed=$VI_VII_SEED"
echo "Steps: $VI_VII_A_EXIT_AFTER | rate: 1/$VI_VII_A_TRIGGER_RATE | duration: $VI_VII_A_DURATION"
echo "Backward kernel func IDs (${#backward_ids[@]}): ${backward_ids[*]}"
echo "Bits: ${VI_VII_A_BITS[*]}"
echo "Checkpoint root: $VI_VII_A_CHECKPOINT_ROOT"
echo "Log root: $VI_VII_A_LOG_ROOT"
echo "============================================================"

# ----- 1. Fault-free baseline (detection path on, no NVBit preload) -----
CURRENT_RUN="vii_a_baseline"
if is_complete "$baseline_dir"; then
  echo "[VII-A] SKIP completed baseline."
else
  if [[ -e "$baseline_dir" && -n "$(ls -A "$baseline_dir" 2>/dev/null)" ]] || [[ -e "$baseline_log" ]]; then
    echo "[VII-A][ERROR] Partial baseline output already exists; move it away before resuming." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$baseline_log")"
  echo "[VII-A] START baseline (detection on, no FI)"
  python torchrun_main.py \
    --single_gpu \
    --seed "$VI_VII_SEED" \
    --model_config "configs/${VI_VII_MODEL}.json" \
    --max_length "$VI_VII_MAX_LENGTH" \
    --lr "$VI_VII_LR" \
    --scheduler cosine \
    --batch_size "$VI_VII_BATCH_SIZE" \
    --total_batch_size "$VI_VII_TOTAL_BATCH_SIZE" \
    --num_training_steps "$VI_VII_TRAINING_SCHEDULE_STEPS" \
    --warmup_steps "$VI_VII_WARMUP_STEPS" \
    --eval_every "$VI_VII_A_EVAL_EVERY" \
    --exit_after "$VI_VII_A_EXIT_AFTER" \
    --save_every "$VI_VII_A_SAVE_EVERY" \
    --grad_clipping 1.0 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$VI_VII_WORKERS" \
    --record_attn_metrics \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --save_dir "$baseline_dir" \
    --name "vii_a_baseline_seed_${VI_VII_SEED}" \
    2>&1 | tee "$baseline_log"
  is_complete "$baseline_dir" || { echo "[VII-A][ERROR] Baseline ended without a complete checkpoint." >&2; exit 1; }
  echo "[VII-A] DONE baseline."
fi
require_file "$baseline_dir/metrics.jsonl"

# ----- 2. Per-bit FI runs (recompute and no-recompute) -----
total_runs=$(( ${#VI_VII_A_BITS[@]} * 2 ))
run_index=0
for bit in "${VI_VII_A_BITS[@]}"; do
  bitmask=$(python -c "print(1 << $bit)")
  for mode in recompute no_recompute; do
    run_index=$((run_index + 1))
    run_name="vii_a_bit_${bit}_${mode}_seed_${VI_VII_SEED}"
    save_dir="$VI_VII_A_CHECKPOINT_ROOT/bit_${bit}/${mode}"
    log_file="$VI_VII_A_LOG_ROOT/bit_${bit}/${mode}.log"
    CURRENT_RUN="$run_name"

    if is_complete "$save_dir"; then
      echo "[VII-A][$run_index/$total_runs] SKIP completed: bit=$bit mode=$mode"
      continue
    fi
    if [[ -e "$save_dir" || -e "$log_file" ]]; then
      echo "[VII-A][ERROR] Partial output already exists for bit=$bit mode=$mode:" >&2
      echo "  $save_dir" >&2
      echo "  $log_file" >&2
      echo "Move or rename the partial output before resuming; it will not be overwritten." >&2
      exit 1
    fi
    mkdir -p "$(dirname "$save_dir")" "$(dirname "$log_file")"

    extra_args=()
    if [[ "$mode" == "recompute" ]]; then
      extra_args=(--fi_nvbit_recompute --fi_nvbit_alpha "$VI_VII_ALPHA")
    fi

    echo
    echo "[VII-A][$run_index/$total_runs] START bit=$bit (bitmask=$bitmask) mode=$mode"
    echo "Checkpoint: $save_dir/model_${VI_VII_A_EXIT_AFTER}"
    echo "Log: $log_file"

    TARGET_FUNC_CONTAINS="$VI_VII_TARGET_FUNC_CONTAINS" \
    TARGET_INSTR=-1 \
    TARGET_BITMASK="$bitmask" \
    LD_PRELOAD="$NVBIT_SO" \
    python torchrun_main.py \
      --single_gpu \
      --seed "$VI_VII_SEED" \
      --model_config "configs/${VI_VII_MODEL}.json" \
      --max_length "$VI_VII_MAX_LENGTH" \
      --lr "$VI_VII_LR" \
      --scheduler cosine \
      --batch_size "$VI_VII_BATCH_SIZE" \
      --total_batch_size "$VI_VII_TOTAL_BATCH_SIZE" \
      --num_training_steps "$VI_VII_TRAINING_SCHEDULE_STEPS" \
      --warmup_steps "$VI_VII_WARMUP_STEPS" \
      --eval_every "$VI_VII_A_EVAL_EVERY" \
      --exit_after "$VI_VII_A_EXIT_AFTER" \
      --save_every "$VI_VII_A_SAVE_EVERY" \
      --grad_clipping 1.0 \
      --weight_decay 0.01 \
      --dtype bfloat16 \
      --optimizer adamw \
      --workers "$VI_VII_WORKERS" \
      --record_attn_metrics \
      --tokenizer_path "$TOKENIZER_PATH" \
      --train_data_path "$TRAIN_DATA_PATH" \
      --val_data_path "$VAL_DATA_PATH" \
      --base_model_path "$baseline_dir" \
      --compare_every "$VI_VII_A_EXIT_AFTER" \
      --save_dir "$save_dir" \
      --name "$run_name" \
      --fi_nvbit_enable \
      --fi_nvbit_location "$VI_VII_LOCATION" \
      --fi_nvbit_trigger_rate "$VI_VII_A_TRIGGER_RATE" \
      --fi_nvbit_target_funcs "${backward_ids[@]}" \
      --fi_nvbit_duration "$VI_VII_A_DURATION" \
      --fi_nvbit_steps -1 \
      "${extra_args[@]}" \
      2>&1 | tee "$log_file"

    is_complete "$save_dir" || { echo "[VII-A][ERROR] $run_name ended without a complete checkpoint." >&2; exit 1; }
    if [[ "$VI_VII_KEEP_OPTIMIZER_PT" == "0" ]]; then
      rm -f -- "$save_dir/model_${VI_VII_A_EXIT_AFTER}/optimizer.pt"
      echo "[VII-A][$run_index/$total_runs] Removed optimizer.pt (not needed for offline analysis)."
    fi
    echo "[VII-A][$run_index/$total_runs] DONE bit=$bit mode=$mode"
  done
done

echo
echo "[VII-A] Campaign complete. Outputs: $VI_VII_A_CHECKPOINT_ROOT"

CURRENT_RUN="VII-A summary"
python scripts/experiments/summarize_vii_a.py \
  --checkpoint-root "$VI_VII_A_CHECKPOINT_ROOT" \
  --baseline-dir "$baseline_dir" \
  --output "$VI_VII_A_LOG_ROOT/vii_a_summary.csv" \
  --expected-step "$VI_VII_A_EXIT_AFTER" \
  --alpha "$VI_VII_ALPHA"

echo "[VII-A] Summary: $VI_VII_A_LOG_ROOT/vii_a_summary.csv"
