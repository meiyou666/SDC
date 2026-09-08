#!/usr/bin/env bash

# IV-F rate rerun using the IV-B injection semantics: within one selected
# kernel, instrument all matching HMMA instructions and flip the first input
# register. Source this after iv_a.sh.
IVF_RATE_CONFIG_REVISION=1

IVF_RATE_MODEL="$IV_A_MODEL"
IVF_RATE_MAX_LENGTH="$IV_A_MAX_LENGTH"
IVF_RATE_BATCH_SIZE="$IV_A_BATCH_SIZE"
IVF_RATE_TOTAL_BATCH_SIZE="$IV_A_TOTAL_BATCH_SIZE"
IVF_RATE_LR="$IV_A_LR"
IVF_RATE_TRAINING_SCHEDULE_STEPS="$IV_A_TRAINING_SCHEDULE_STEPS"
IVF_RATE_WARMUP_STEPS="$IV_A_WARMUP_STEPS"
IVF_RATE_EXIT_AFTER="$IV_A_EXIT_AFTER"
IVF_RATE_EVAL_EVERY="$IV_A_EVAL_EVERY"
IVF_RATE_SAVE_EVERY="$IV_A_SAVE_EVERY"
IVF_RATE_WORKERS="$IV_A_WORKERS"

# Single-seed rerun. Keep this tied to the paired mb256 baseline seed.
IVF_RATE_SEED="$IV_A_SEED"

# Paper IV-F rate axis with the revised local injection target.
IVF_RATE_BIT=13
IVF_RATE_TARGET_BITMASK=$((1 << IVF_RATE_BIT))
IVF_RATE_TARGET_REGISTER="$IV_A_TARGET_REGISTER"
IVF_RATE_TARGET_SMID="$IV_A_TARGET_SMID"
IVF_RATE_TARGET_LANEID="$IV_A_TARGET_LANEID"
IVF_RATE_TARGET_OP="$IV_A_TARGET_OP"
IVF_RATE_TARGET_FUNC="$IV_A_TARGET_FUNC"
IVF_RATE_TARGET_INSTR=-1
IVF_RATE_DURATION=1

# Run modes used by run_iv_f_rate_all_hmma.sh.
IVF_RATE_PROBE_RATES=(1)
# Main IV-F rate rerun: run the three new rates in one serial pass. The 1/10
# condition is reused from IV-B when the seed, bit, kernel, and all-HMMA target
# match exactly.
IVF_RATE_SERIAL_RATES=(1 100 1000)
IVF_RATE_FULL_RATES=(1 10 100 1000)

IVF_RATE_KERNEL_MANIFEST="$IV_A_KERNEL_MANIFEST"
IVF_RATE_BASELINE_DIR="$IV_A_BASELINE_DIR"
IVF_RATE_CHECKPOINT_ROOT=checkpoints/iv_f_rate_all_hmma_mb256
IVF_RATE_LOG_ROOT=logs/iv_f_rate_all_hmma_mb256

# Set to 1 only if completed runs must remain resumable.
IVF_RATE_KEEP_OPTIMIZER_PT=0
