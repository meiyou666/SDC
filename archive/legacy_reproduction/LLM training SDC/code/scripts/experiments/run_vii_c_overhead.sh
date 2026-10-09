#!/usr/bin/env bash
set -Eeuo pipefail

# Paper Section VII-C: runtime overhead of the detection path.
# Detection-on sample : first 2,000 steps of the saved VII-B fault-free
#                       baseline run (torchrun_main.py).
# Detection-off sample: matching 2,000-step fault-free run with
#                       torchrun_main_no_detection.py.
# Overhead is reported as the relative difference of median per-update-step
# time for steps 101..2000.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source scripts/configs/common.sh
source scripts/configs/vi_vii.sh

CURRENT_RUN="preflight"
trap 'status=$?; echo "[VII-C][ERROR] ${CURRENT_RUN} failed with exit code ${status}." >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/experiments/run_vii_c_overhead.sh

Reuses the saved VII-B fault-free baseline as the detection-on sample,
launches only the matching 2,000-step detection-off sample if needed, then
compares steady-state s/it over steps 101..2000.
EOF
}

is_complete() {
  local run_dir="$1"
  local checkpoint="$run_dir/model_${VI_VII_C_EXIT_AFTER}"
  [[ -f "$checkpoint/model.safetensors" \
    && -f "$checkpoint/training_state.json" \
    && -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

detection_on_complete() {
  local run_dir="$1"
  [[ -f "$run_dir/metrics.jsonl" \
    && -f "$run_dir/summary.json" \
    && -f "$run_dir/run_config.json" ]]
}

require_file() {
  [[ -f "$1" ]] || { echo "[VII-C][ERROR] Required file not found: $1" >&2; exit 1; }
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ "$VI_VII_CONFIG_REVISION" -ge 1 ]] || { echo "[VII-C][ERROR] vi_vii config revision 1 or newer is required." >&2; exit 1; }
[[ "$VI_VII_BATCH_SIZE" -eq 256 && "$VI_VII_TOTAL_BATCH_SIZE" -eq 512 ]] || { echo "[VII-C][ERROR] Expected paper 60M batch structure: batch_size=256, total_batch_size=512." >&2; exit 1; }
command -v python >/dev/null || { echo "[VII-C][ERROR] Activate the project Python environment first." >&2; exit 1; }
require_file "configs/${VI_VII_MODEL}.json"
require_file "$TOKENIZER_PATH/tokenizer_config.json"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"

detection_on_dir="$VI_VII_C_DETECTION_ON_DIR"
no_detection_dir="$VI_VII_C_NO_DETECTION_DIR"
log_file="$VI_VII_C_LOG_ROOT/no_detection_match_vii_b_seed_${VI_VII_SEED}.log"
mkdir -p "$VI_VII_C_LOG_ROOT"

unset LD_PRELOAD INSTR_BEGIN INSTR_END TARGET_EVERY TARGET_INSTR_MAP_FILE
export PYTHONUNBUFFERED=1

CURRENT_RUN="vii_c_detection_on_reuse_check"
detection_on_complete "$detection_on_dir" || {
  echo "[VII-C][ERROR] Detection-on reuse source is incomplete: $detection_on_dir" >&2
  echo "Expected metrics.jsonl, summary.json, and run_config.json from the saved VII-B baseline segment." >&2
  exit 1
}

echo "============================================================"
echo "VII-C runtime overhead: reuse VII-B detection-on vs matched detection-off"
echo "Steps compared: 1-$VI_VII_C_EXIT_AFTER (median after first $VI_VII_C_WARMUP_SAMPLE_STEPS steps)"
echo "Detection on : $detection_on_dir (reused; no new run)"
echo "Detection off: $no_detection_dir"
echo "No-detection config: eval_every=$VI_VII_C_EVAL_EVERY save_every=$VI_VII_C_SAVE_EVERY repeat=$VI_VII_C_DATA_REPEAT"
echo "============================================================"

CURRENT_RUN="vii_c_no_detection"
if is_complete "$no_detection_dir"; then
  echo "[VII-C] SKIP completed no-detection run."
else
  if [[ -e "$no_detection_dir" || -e "$log_file" ]]; then
    echo "[VII-C][ERROR] Partial no-detection output already exists; move it away before resuming." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$log_file")"
  data_repeat_args=()
  if [[ "$VI_VII_C_DATA_REPEAT" -ne 1 ]]; then
    data_repeat_args=(--train_data_repeat "$VI_VII_C_DATA_REPEAT")
  fi
  echo "[VII-C] START no-detection run"
  python torchrun_main_no_detection.py \
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
    --eval_every "$VI_VII_C_EVAL_EVERY" \
    --exit_after "$VI_VII_C_EXIT_AFTER" \
    --save_every "$VI_VII_C_SAVE_EVERY" \
    --grad_clipping 1.0 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --workers "$VI_VII_WORKERS" \
    --record_attn_metrics \
    --tokenizer_path "$TOKENIZER_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    "${data_repeat_args[@]}" \
    --val_data_path "$VAL_DATA_PATH" \
    --save_dir "$no_detection_dir" \
    --name "vii_c_no_detection_seed_${VI_VII_SEED}" \
    2>&1 | tee "$log_file"
  is_complete "$no_detection_dir" || { echo "[VII-C][ERROR] No-detection run ended without a complete checkpoint." >&2; exit 1; }
  echo "[VII-C] DONE no-detection run."
fi

if [[ "$VI_VII_KEEP_OPTIMIZER_PT" == "0" ]]; then
  rm -f -- "$no_detection_dir/model_${VI_VII_C_EXIT_AFTER}/optimizer.pt"
fi

CURRENT_RUN="VII-C summary"
mkdir -p "$VI_VII_C_LOG_ROOT"
DETECTION_ON_DIR="$detection_on_dir" \
DETECTION_OFF_DIR="$no_detection_dir" \
OUTPUT_CSV="$VI_VII_C_LOG_ROOT/overhead_summary.csv" \
WARMUP_STEPS="$VI_VII_C_WARMUP_SAMPLE_STEPS" \
MAX_STEP="$VI_VII_C_EXIT_AFTER" \
python - <<'EOF'
import csv
import json
import math
import os
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def median_steady_state_s_per_it(metrics_path, warmup_steps, max_step):
    records = load_jsonl(metrics_path)
    core = [r for r in records if "loss" in r and "_time" in r]
    times = [
        (int(r["_step"]), float(r["_time"]))
        for r in core
        if 1 <= int(r["_step"]) <= max_step
    ]
    present_steps = {step for step, _ in times}
    missing = sorted(set(range(1, max_step + 1)) - present_steps)
    if missing:
        raise SystemExit(
            f"{metrics_path} does not contain a complete 1..{max_step} step range; "
            f"missing first examples: {missing[:10]}"
        )
    deltas = []
    for (step_a, time_a), (step_b, time_b) in zip(times, times[1:]):
        if step_b <= warmup_steps:
            continue
        dt = time_b - time_a
        if dt > 0 and math.isfinite(dt):
            deltas.append(dt)
    deltas.sort()
    return deltas[len(deltas) // 2] if deltas else float("nan"), len(deltas), len(times)


on_dir = Path(os.environ["DETECTION_ON_DIR"])
off_dir = Path(os.environ["DETECTION_OFF_DIR"])
out_path = Path(os.environ["OUTPUT_CSV"])
warmup = int(os.environ["WARMUP_STEPS"])
max_step = int(os.environ["MAX_STEP"])

on_sit, on_n, on_total = median_steady_state_s_per_it(on_dir / "metrics.jsonl", warmup, max_step)
off_sit, off_n, off_total = median_steady_state_s_per_it(off_dir / "metrics.jsonl", warmup, max_step)
overhead_pct = (on_sit - off_sit) / off_sit * 100 if off_sit else float("nan")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["run", "median_s_per_it_steady_state", "steps_sampled", "steps_total", "step_range", "warmup_steps"])
    writer.writerow(["detection_on_reused_vii_b", on_sit, on_n, on_total, f"1-{max_step}", warmup])
    writer.writerow(["detection_off", off_sit, off_n, off_total, f"1-{max_step}", warmup])
    writer.writerow(["overhead_pct", overhead_pct, "", "", "", ""])

print(f"median s/it (detection on reused) : {on_sit:.4f}  ({on_n} deltas after warmup {warmup}, steps 1-{max_step})")
print(f"median s/it (detection off)       : {off_sit:.4f}  ({off_n} deltas after warmup {warmup}, steps 1-{max_step})")
print(f"overhead: {overhead_pct:.2f}%")
print(f"Wrote {out_path}")
EOF

echo "[VII-C] Summary: $VI_VII_C_LOG_ROOT/overhead_summary.csv"
