#!/usr/bin/env bash

# Paper Section IV-E: spatial effects (lane sweep and SM sweep), reduced to
# 16 runs per part. Source this after scripts/configs/iv_a.sh and
# scripts/configs/iv_b.sh so IV-E inherits the validated mb256 training shape
# and paired baseline path.
IV_E_CONFIG_REVISION=1

IV_E_MODEL="$IV_A_MODEL"
IV_E_MAX_LENGTH="$IV_A_MAX_LENGTH"
IV_E_BATCH_SIZE="$IV_A_BATCH_SIZE"
IV_E_TOTAL_BATCH_SIZE="$IV_A_TOTAL_BATCH_SIZE"
IV_E_LR="$IV_A_LR"
IV_E_TRAINING_SCHEDULE_STEPS="$IV_A_TRAINING_SCHEDULE_STEPS"
IV_E_WARMUP_STEPS="$IV_A_WARMUP_STEPS"
IV_E_EXIT_AFTER="$IV_A_EXIT_AFTER"
IV_E_EVAL_EVERY="$IV_A_EVAL_EVERY"
IV_E_SAVE_EVERY="$IV_A_SAVE_EVERY"
IV_E_WORKERS="$IV_A_WORKERS"
IV_E_SEED="$IV_A_SEED"
IV_E_KEEP_OPTIMIZER_PT="$IV_A_KEEP_OPTIMIZER_PT"

# Fixed paper IV-E target: kernel BP9 from the local kernel manifest, bit 13,
# backward injection covering all HMMA instructions in the kernel.
IV_E_BIT=13
IV_E_TARGET_BITMASK=8192
IV_E_KERNEL_LABEL=BP9
IV_E_LOCATION=backward
IV_E_TARGET_INSTR=-1

# Spatial sampling. The local RTX 4090 has 128 SMs (confirmed via
# multi_processor_count). Each part is reduced to 16 runs with even stride:
#   lane part: 16 lanes, stride 2 over 0..31  -> 0,2,...,30,   SM fixed at 0
#   SM part:   16 SMs,   stride 8 over 0..127 -> 0,8,...,120, lane fixed at 0
IV_E_RUNS_PER_PART=16
IV_E_LANE_FIXED_SMID=0
IV_E_LANE_START=0
IV_E_LANE_STRIDE=2
IV_E_LANE_END=31
IV_E_SM_FIXED_LANEID=0
IV_E_SM_START=0
IV_E_SM_STRIDE=8
IV_E_SM_END=127
IV_E_SM_COUNT=128

IV_E_TRIGGER_RATE="$IV_A_TRIGGER_RATE"
IV_E_DURATION="$IV_A_DURATION"
IV_E_TARGET_REGISTER="$IV_A_TARGET_REGISTER"
IV_E_TARGET_OP="$IV_A_TARGET_OP"
IV_E_TARGET_FUNC="$IV_A_TARGET_FUNC"

IV_E_KERNEL_MANIFEST="$IV_A_KERNEL_MANIFEST"
IV_E_BASELINE_DIR="$IV_A_BASELINE_DIR"
IV_E_CHECKPOINT_ROOT=checkpoints/iv_e_mb256
IV_E_LOG_ROOT=logs/iv_e_mb256
