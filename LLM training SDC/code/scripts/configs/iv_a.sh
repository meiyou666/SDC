#!/usr/bin/env bash

# Paper Section IV-A: bit-position sensitivity. Bits 13 and 14 are handled by
# separate sanity runs and skipped by the two half-register sweeps.
IV_A_CONFIG_REVISION=6

IV_A_MODEL=llama_60m
IV_A_MAX_LENGTH=256
IV_A_BATCH_SIZE=256
IV_A_TOTAL_BATCH_SIZE=512
IV_A_LR=1e-3
IV_A_TRAINING_SCHEDULE_STEPS=100000
IV_A_WARMUP_STEPS=1000
IV_A_EXIT_AFTER=1000
IV_A_EVAL_EVERY=100000
IV_A_SAVE_EVERY=1000
IV_A_WORKERS=0

# Single seed used by IV-A and the paired fault-free baseline. Change this one
# value if later experiments should use a different fixed seed.
IV_A_SEED=42

# The paper analysis needs the final model and metrics, not many Adam optimizer
# states. Set to 1 only if every completed run must remain resumable.
IV_A_KEEP_OPTIMIZER_PT=0

IV_A_BIT=13
IV_A_TARGET_BITMASK=8192
IV_A_TRIGGER_RATE=10
IV_A_DURATION=1
IV_A_LOCATIONS=(forward backward)

IV_A_TARGET_REGISTER=1
IV_A_TARGET_SMID=0
IV_A_TARGET_LANEID=0
IV_A_TARGET_OP=HMMA
IV_A_TARGET_FUNC=-1

# Stable local target used by the IV-A bit-position sweep. The paper does not
# publish concrete SASS instruction indexes, so keep the previously validated
# local target instead of inferring a first-five selection rule.
IV_A_TARGET_FUNC_CONTAINS=ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn
IV_A_BIT_SWEEP_LOCATION=backward
IV_A_BIT_SWEEP_TARGET_INSTR=312
IV_A_SKIP_BITS=(13 14)

# Profiling uses TARGET_INSTR=-1 only to enumerate matching HMMA instructions.
# The generated manifest is only evidence for optional kernel-level probing.
IV_A_TARGET_INSTR=-1
IV_A_INSTR_SELECTION_SEED=42

IV_A_PROFILE_EXIT_AFTER=12
IV_A_PROFILE_TRIGGER_STEP=12
IV_A_PROFILE_FUNC_CONTAINS=gemm

IV_A_CHECKPOINT_ROOT=checkpoints/iv_a_mb256
IV_A_LOG_ROOT=logs/iv_a_mb256
IV_A_KERNEL_MANIFEST="$IV_A_LOG_ROOT/kernel_manifest.tsv"
IV_A_BASELINE_DIR="checkpoints/iv_f_mb256/baseline/seed_${IV_A_SEED}"
