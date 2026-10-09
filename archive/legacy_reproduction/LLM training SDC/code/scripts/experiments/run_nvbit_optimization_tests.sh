#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/iv_f.sh

MODE="${1:-all}"
if [[ "$MODE" != "pretrigger" && "$MODE" != "injection" && "$MODE" != "all" ]]; then
  echo "Usage: $0 [pretrigger|injection|all]" >&2
  exit 2
fi

NVBIT_SO="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so"
NVBIT_SOURCE="$REPO_ROOT/nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.cu"
RUN_TAG="${NVBIT_TEST_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_ROOT="checkpoints/nvbit_optimization/$RUN_TAG"
LOG_ROOT="logs/nvbit_optimization/$RUN_TAG"
TEST_SEED="${NVBIT_TEST_SEED:-351344}"
MAX_SLOWDOWN="${NVBIT_MAX_FASTPATH_SLOWDOWN:-${NVBIT_MAX_PRETRIGGER_SLOWDOWN:-3.0}}"

require_file() {
  [[ -f "$1" ]] || { echo "[NVBit test][ERROR] Required file not found: $1" >&2; exit 1; }
}

require_file "$NVBIT_SO"
require_file "configs/${IVF_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"

if [[ "$NVBIT_SOURCE" -nt "$NVBIT_SO" ]]; then
  echo "[NVBit test][ERROR] fault_injection.cu is newer than fault_injection.so." >&2
  echo "Rebuild the tool before running these tests." >&2
  exit 1
fi

if [[ -e "$CHECKPOINT_ROOT" || -e "$LOG_ROOT" ]]; then
  echo "[NVBit test][ERROR] RUN_TAG already exists: $RUN_TAG" >&2
  echo "Choose a new NVBIT_TEST_RUN_TAG; existing results will not be overwritten." >&2
  exit 1
fi

unset LD_PRELOAD
export TARGET_REGISTER="$IVF_TARGET_REGISTER"
export TARGET_OP="$IVF_TARGET_OP"
export TARGET_SMID="$IVF_TARGET_SMID"
export TARGET_LANEID="$IVF_TARGET_LANEID"
export TARGET_BITMASK="$IVF_TARGET_BITMASK"
export TARGET_FUNC="$IVF_TARGET_FUNC"
export TARGET_INSTR="$IVF_TARGET_INSTR"
export TARGET_FUNC_CONTAINS="$IVF_TARGET_FUNC_CONTAINS"
export PYTHONUNBUFFERED=1

mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT"

common_args=(
  --single_gpu
  --seed "$TEST_SEED"
  --model_config "configs/${IVF_MODEL}.json"
  --max_length "$IVF_MAX_LENGTH"
  --lr "$IVF_LR"
  --scheduler cosine
  --batch_size "$IVF_BATCH_SIZE"
  --total_batch_size "$IVF_TOTAL_BATCH_SIZE"
  --num_training_steps "$IVF_TRAINING_SCHEDULE_STEPS"
  --warmup_steps "$IVF_WARMUP_STEPS"
  --eval_every "$IVF_EVAL_EVERY"
  --grad_clipping 1.0
  --weight_decay 0.01
  --dtype bfloat16
  --optimizer adamw
  --workers "$IVF_WORKERS"
  --record_attn_metrics
  --disable_final_evaluation
  --tokenizer_path "$TOKENIZER_PATH"
  --train_data_path "$TRAIN_DATA_PATH"
  --val_data_path "$VAL_DATA_PATH"
)

run_plain() {
  local run_name="$1"
  local steps="$2"
  local save_dir="$CHECKPOINT_ROOT/$run_name"
  local log_file="$LOG_ROOT/$run_name.log"

  echo "[NVBit test] START $run_name"
  python torchrun_main.py \
    "${common_args[@]}" \
    --exit_after "$steps" \
    --save_every "$steps" \
    --save_dir "$save_dir" \
    --name "$run_name" \
    2>&1 | tee "$log_file"
  echo "[NVBit test] DONE $run_name"
}

run_fi() {
  local run_name="$1"
  local steps="$2"
  local trigger_step="$3"
  local verbose="$4"
  local save_dir="$CHECKPOINT_ROOT/$run_name"
  local log_file="$LOG_ROOT/$run_name.log"

  echo "[NVBit test] START $run_name (trigger step: $trigger_step)"
  LD_PRELOAD="$NVBIT_SO" TOOL_VERBOSE="$verbose" python torchrun_main.py \
    "${common_args[@]}" \
    --exit_after "$steps" \
    --save_every "$steps" \
    --save_dir "$save_dir" \
    --name "$run_name" \
    --fi_nvbit_enable \
    --fi_nvbit_location backward \
    --fi_nvbit_target_funcs -1 \
    --fi_nvbit_duration 1 \
    --fi_nvbit_steps "$trigger_step" \
    2>&1 | tee "$log_file"
  echo "[NVBit test] DONE $run_name"
}

run_pretrigger_test() {
  local steps=30
  echo "============================================================"
  echo "Test 1: pre-trigger overhead and trajectory equivalence"
  echo "No injection should occur because trigger Step 500 is outside 30 steps."
  echo "============================================================"
  run_plain pretrigger_baseline "$steps"
  run_fi pretrigger_fi "$steps" 500 0
  python scripts/experiments/analyze_nvbit_optimization.py pretrigger \
    --baseline-dir "$CHECKPOINT_ROOT/pretrigger_baseline" \
    --fi-dir "$CHECKPOINT_ROOT/pretrigger_fi" \
    --expected-steps "$steps" \
    --max-slowdown "$MAX_SLOWDOWN" \
    2>&1 | tee "$LOG_ROOT/pretrigger_analysis.log"
}

run_injection_test() {
  local steps=15
  local trigger_step=12
  echo "============================================================"
  echo "Test 2: injection schedule and checkpoint regression"
  echo "Exactly one backward injection should occur at Step 12."
  echo "============================================================"
  run_plain injection_control "$steps"
  run_fi injection_fi "$steps" "$trigger_step" 1
  python scripts/experiments/analyze_nvbit_optimization.py injection \
    --control-dir "$CHECKPOINT_ROOT/injection_control" \
    --fi-dir "$CHECKPOINT_ROOT/injection_fi" \
    --expected-steps "$steps" \
    --trigger-step "$trigger_step" \
    --max-posttrigger-slowdown "$MAX_SLOWDOWN" \
    2>&1 | tee "$LOG_ROOT/injection_analysis.log"

  grep -nE "Enable FI at update step|instrumentation enabled|idx: 312" \
    "$LOG_ROOT/injection_fi.log" \
    | tee "$LOG_ROOT/injection_evidence.log"
}

case "$MODE" in
  pretrigger)
    run_pretrigger_test
    ;;
  injection)
    run_injection_test
    ;;
  all)
    run_pretrigger_test
    run_injection_test
    ;;
esac

echo
echo "[NVBit test] PASS: $MODE"
echo "Checkpoints: $CHECKPOINT_ROOT"
echo "Logs: $LOG_ROOT"
