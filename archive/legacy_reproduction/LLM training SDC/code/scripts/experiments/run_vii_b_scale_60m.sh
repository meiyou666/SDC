#!/usr/bin/env bash
set -Eeuo pipefail

# Paper Section VII-B (downgraded to the 60M column of Table I): baseline,
# fault injection, and fault injection + recompute at alpha=0.05.
# 60M, seed 42, 10,000 steps, eval every 1,000 steps. Faults: BP8/BP9 chosen
# at random per trigger, rate 1/100, duration random 1-5, bit 13, backward
# location, all HMMA instructions of the chosen kernel, first input register,
# SM0/lane0. Local deviation from paper (bit 12, random kernels): IV-B shows
# bit 13 + BP8/BP9 are the strongest non-saturating local target; the
# paper-config bit-12 run showed no measurable degradation (null result).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/vi_vii.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[VII-B][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/experiments/run_vii_b_scale_60m.sh

Single mode: runs baseline, FI, and FI+recompute serially, skipping runs
that are already complete.
EOF
}

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${VI_VII_B_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

require_file() {
  [[ -f "$1" ]] || { echo "[VII-B][ERROR] Required file not found: $1" >&2; exit 1; }
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ "$VI_VII_CONFIG_REVISION" -ge 1 ]] || { echo "[VII-B][ERROR] vi_vii config revision 1 or newer is required." >&2; exit 1; }
[[ "$VI_VII_TARGET_INSTR" -eq -1 ]] || { echo "[VII-B][ERROR] TARGET_INSTR must be -1 (all matching HMMA)." >&2; exit 1; }
[[ "$VI_VII_B_DURATION_RANDOM" -eq 1 ]] || { echo "[VII-B][ERROR] VII-B expects random duration 1..$VI_VII_B_DURATION_MAX." >&2; exit 1; }
[[ "$VI_VII_B_DATA_REPEAT" -ge 1 ]] || { echo "[VII-B][ERROR] VI_VII_B_DATA_REPEAT must be >= 1." >&2; exit 1; }
[[ "$VI_VII_BATCH_SIZE" -eq 256 && "$VI_VII_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[VII-B][ERROR] Expected paper 60M batch structure: batch_size=256, total_batch_size=512." >&2; exit 1; }
command -v python >/dev/null || { echo "[VII-B][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${VI_VII_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$VI_VII_KERNEL_MANIFEST"

expected_header=$'location\tkernel_label\tfunction_id_first_seen\tlaunch_count\thmma_instr_count\thmma_instr_indexes\ttarget_instr\tkernel_name\ttarget_func_contains'
actual_header="$(head -n 1 "$VI_VII_KERNEL_MANIFEST")"
[[ "$actual_header" == "$expected_header" ]] || {
  echo "[VII-B][ERROR] Unexpected kernel manifest header. Regenerate it with profile_iv_a_kernels.sh." >&2
  exit 1
}

# Candidate function IDs: BP8 and BP9 (strongest finite backward kernels in
# local IV-B at bit 13; replaces the paper-style random kernel choice).
target_ids=()
for wanted_label in BP8 BP9; do
  function_id="$(awk -F '\t' -v label="$wanted_label" '$2 == label {print $3; exit}' "$VI_VII_KERNEL_MANIFEST")"
  [[ "$function_id" =~ ^[0-9]+$ ]] || { echo "[VII-B][ERROR] Function ID for $wanted_label not found in manifest." >&2; exit 1; }
  location="$(awk -F '\t' -v label="$wanted_label" '$2 == label {print $1; exit}' "$VI_VII_KERNEL_MANIFEST")"
  [[ "$location" == "backward" ]] || { echo "[VII-B][ERROR] $wanted_label is not a backward kernel (location=$location)." >&2; exit 1; }
  target_ids+=("$function_id")
done
bitmask=$(python -c "print(1 << $VI_VII_B_BIT)")
manifest_sha256="$(python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$VI_VII_KERNEL_MANIFEST")"

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
require_file "$NVBIT_SO"
if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[VII-B][ERROR] NVBit source is newer than the .so. Rebuild it before running this campaign." >&2
  exit 1
fi

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export TOOL_VERBOSE="${VI_VII_TOOL_VERBOSE:-0}"
export TARGET_REGISTER="$VI_VII_TARGET_REGISTER"
export TARGET_OP="$VI_VII_TARGET_OP"
export TARGET_SMID="$VI_VII_TARGET_SMID"
export TARGET_LANEID="$VI_VII_TARGET_LANEID"
export TARGET_BITMASK="$bitmask"
export TARGET_FUNC=-1
export PYTHONUNBUFFERED=1

baseline_dir="$VI_VII_B_CHECKPOINT_ROOT/baseline/seed_${VI_VII_SEED}"
mkdir -p "$VI_VII_B_CHECKPOINT_ROOT" "$VI_VII_B_LOG_ROOT"

campaign_manifest="$VI_VII_B_CHECKPOINT_ROOT/campaign_manifest.txt"
expected_manifest=$(cat <<EOF
campaign=vii_b_scale_60m
seed=$VI_VII_SEED
bit=$VI_VII_B_BIT
bitmask=$bitmask
kernel_labels=BP8,BP9
target_func_ids=${target_ids[*]}
target_register=$VI_VII_TARGET_REGISTER
target_smid=$VI_VII_TARGET_SMID
target_laneid=$VI_VII_TARGET_LANEID
target_op=$VI_VII_TARGET_OP
target_instr=-1_all_matching_hmma
rate=$VI_VII_B_TRIGGER_RATE
duration=random_1_${VI_VII_B_DURATION_MAX}
alpha=$VI_VII_ALPHA
exit_after=$VI_VII_B_EXIT_AFTER
eval_every=$VI_VII_B_EVAL_EVERY
data_repeat=$VI_VII_B_DATA_REPEAT
manifest=$VI_VII_KERNEL_MANIFEST
manifest_sha256=$manifest_sha256
EOF
)
if [[ -f "$campaign_manifest" ]]; then
  recorded_manifest="$(cat "$campaign_manifest")"
  [[ "$recorded_manifest" == "$expected_manifest" ]] || {
    echo "[VII-B][ERROR] Existing campaign manifest does not match current configuration: $campaign_manifest" >&2
    exit 1
  }
else
  printf '%s\n' "$expected_manifest" > "$campaign_manifest"
fi

echo "============================================================"
echo "VII-B scale experiment (60M column of paper Table I)"
echo "Seed: $VI_VII_SEED | Steps: $VI_VII_B_EXIT_AFTER | eval every $VI_VII_B_EVAL_EVERY"
echo "Faults: bit=$VI_VII_B_BIT (bitmask=$bitmask) rate=1/$VI_VII_B_TRIGGER_RATE duration=random_1_${VI_VII_B_DURATION_MAX}"
echo "Kernels: BP8/BP9 func IDs: ${target_ids[*]}"
echo "Alpha: $VI_VII_ALPHA"
echo "Checkpoint root: $VI_VII_B_CHECKPOINT_ROOT"
echo "Log root: $VI_VII_B_LOG_ROOT"
echo "============================================================"

run_conditions=(baseline fi fi_recompute)
total_runs=${#run_conditions[@]}
run_index=0
for condition in "${run_conditions[@]}"; do
  run_index=$((run_index + 1))
  run_name="vii_b_${condition}_seed_${VI_VII_SEED}"
  save_dir="$VI_VII_B_CHECKPOINT_ROOT/${condition}/seed_${VI_VII_SEED}"
  log_file="$VI_VII_B_LOG_ROOT/${condition}_seed_${VI_VII_SEED}.log"
  CURRENT_RUN="$run_name"

  if is_complete "$save_dir"; then
    echo "[VII-B][$run_index/$total_runs] SKIP completed: $condition"
    continue
  fi
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "[VII-B][ERROR] Partial output already exists for $condition:" >&2
    echo "  $save_dir" >&2
    echo "  $log_file" >&2
    echo "Move or rename the partial output before resuming; it will not be overwritten." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$save_dir")" "$(dirname "$log_file")"

  fi_prefix=()
  fi_args=()
  if [[ "$condition" != "baseline" ]]; then
    # NOTE: "env" must lead the array: words from array expansion are not re-parsed
    # as variable assignments, so a bare VAR=value element would run as a command.
    fi_prefix=(env TARGET_FUNC_CONTAINS="$VI_VII_TARGET_FUNC_CONTAINS" TARGET_INSTR=-1 LD_PRELOAD="$NVBIT_SO")
    fi_args=(--fi_nvbit_enable --fi_nvbit_location "$VI_VII_LOCATION" \
      --fi_nvbit_trigger_rate "$VI_VII_B_TRIGGER_RATE" \
      --fi_nvbit_target_funcs "${target_ids[@]}" \
      --fi_nvbit_duration "$VI_VII_B_DURATION_MAX" \
      --fi_nvbit_duration_random \
      --fi_nvbit_steps -1)
    if [[ "$condition" == "fi_recompute" ]]; then
      fi_args+=(--fi_nvbit_recompute --fi_nvbit_alpha "$VI_VII_ALPHA")
    fi
  fi

  echo
  echo "[VII-B][$run_index/$total_runs] START condition=$condition"
  echo "Checkpoint: $save_dir/model_${VI_VII_B_EXIT_AFTER}"
  echo "Log: $log_file"

  common_args=(
    --single_gpu
    --seed "$VI_VII_SEED"
    --model_config "configs/${VI_VII_MODEL}.json"
    --max_length "$VI_VII_MAX_LENGTH"
    --lr "$VI_VII_LR"
    --scheduler cosine
    --batch_size "$VI_VII_BATCH_SIZE"
    --total_batch_size "$VI_VII_TOTAL_BATCH_SIZE"
    --num_training_steps "$VI_VII_TRAINING_SCHEDULE_STEPS"
    --warmup_steps "$VI_VII_WARMUP_STEPS"
    --eval_every "$VI_VII_B_EVAL_EVERY"
    --exit_after "$VI_VII_B_EXIT_AFTER"
    --save_every "$VI_VII_B_SAVE_EVERY"
    --grad_clipping 1.0
    --weight_decay 0.01
    --dtype bfloat16
    --optimizer adamw
    --workers "$VI_VII_WORKERS"
    --record_attn_metrics
    --tokenizer_path "$TOKENIZER_PATH"
    --train_data_path "$TRAIN_DATA_PATH"
    --train_data_repeat "$VI_VII_B_DATA_REPEAT"
    --val_data_path "$VAL_DATA_PATH"
    --save_dir "$save_dir"
    --name "$run_name"
  )
  if [[ "$condition" != "baseline" ]]; then
    common_args+=(--base_model_path "$baseline_dir" --compare_every "$VI_VII_B_COMPARE_EVERY")
  fi

  "${fi_prefix[@]}" python torchrun_main.py \
    "${common_args[@]}" \
    "${fi_args[@]}" \
    2>&1 | tee "$log_file"

  is_complete "$save_dir" || { echo "[VII-B][ERROR] $run_name ended without a complete checkpoint." >&2; exit 1; }
  if [[ "$VI_VII_KEEP_OPTIMIZER_PT" == "0" ]]; then
    rm -f -- "$save_dir/model_${VI_VII_B_EXIT_AFTER}/optimizer.pt"
    echo "[VII-B][$run_index/$total_runs] Removed optimizer.pt (not needed for offline analysis)."
  fi
  echo "[VII-B][$run_index/$total_runs] DONE condition=$condition"
done

echo
echo "[VII-B] Campaign complete. Outputs: $VI_VII_B_CHECKPOINT_ROOT"

CURRENT_RUN="VII-B summary"
python scripts/experiments/summarize_vii_b.py \
  --checkpoint-root "$VI_VII_B_CHECKPOINT_ROOT" \
  --baseline-dir "$baseline_dir" \
  --output "$VI_VII_B_LOG_ROOT/vii_b_summary.csv" \
  --expected-step "$VI_VII_B_EXIT_AFTER" \
  --eval-every "$VI_VII_B_EVAL_EVERY" \
  --alpha "$VI_VII_ALPHA" \
  --bit "$VI_VII_B_BIT"

echo "[VII-B] Summary: $VI_VII_B_LOG_ROOT/vii_b_summary.csv"
