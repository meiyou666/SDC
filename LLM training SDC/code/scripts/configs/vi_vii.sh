#!/usr/bin/env bash

# Paper Sections VI/VII: detection + recompute reproduction (single seed).
# Training shape is identical to the IV-A mb256 mainline so that kernel
# function IDs from logs/iv_a_mb256/kernel_manifest.tsv stay valid.
VI_VII_CONFIG_REVISION=1

VI_VII_MODEL=llama_60m
VI_VII_MAX_LENGTH=256
VI_VII_BATCH_SIZE=256
VI_VII_TOTAL_BATCH_SIZE=512
VI_VII_LR=1e-3
VI_VII_TRAINING_SCHEDULE_STEPS=100000
VI_VII_WARMUP_STEPS=1000
VI_VII_SEED=42
VI_VII_KEEP_OPTIMIZER_PT=0
VI_VII_WORKERS=0

# Detector sensitivity: fixed to the paper-selected value (paper Section VII-A).
VI_VII_ALPHA=0.05

# Shared injection site (paper Section VII uses first input register, one SM,
# one lane, all HMMA instructions of the chosen kernel).
VI_VII_TARGET_REGISTER=1
VI_VII_TARGET_OP=HMMA
VI_VII_TARGET_INSTR=-1
VI_VII_TARGET_FUNC_CONTAINS=gemm
VI_VII_TARGET_SMID=0
VI_VII_TARGET_LANEID=0
VI_VII_LOCATION=backward

# VII-A: detection + recompute at alpha=0.05, bits 9-12, rate 1/10, duration 1.
# SKIPPED in final plan: recovery effectiveness is covered by the VII-B paired
# fi/fi_recompute runs; VII-C reuses the saved VII-B fault-free baseline sample.
VI_VII_A_BITS=(9 10 11 12)
VI_VII_A_TRIGGER_RATE=10
VI_VII_A_DURATION=1
VI_VII_A_EXIT_AFTER=2000
VI_VII_A_EVAL_EVERY=100000
VI_VII_A_SAVE_EVERY=2000
VI_VII_A_CHECKPOINT_ROOT=checkpoints/vii_a_mb256
VI_VII_A_LOG_ROOT=logs/vii_a_mb256

# VII-B: 60M column of paper Table I, 10,000 steps, eval every 1,000 steps.
# Faults: BP8/BP9 chosen at random per trigger, rate 1/100, duration random 1-5,
# bit 13 (local deviation from paper bit 12, see plan doc).
VI_VII_B_BIT=13
VI_VII_B_TRIGGER_RATE=100
VI_VII_B_DURATION_MAX=5
VI_VII_B_DURATION_RANDOM=1
VI_VII_B_EXIT_AFTER=10000
VI_VII_B_EVAL_EVERY=1000
VI_VII_B_SAVE_EVERY=10000
VI_VII_B_COMPARE_EVERY=10000
# Local C4 subset holds ~2M sequences = one epoch = ~3,906 update steps at
# batch 512. Repeat the data so 10,000 steps fit (~2.56 passes, identical
# batch order each pass); the paper trains on effectively fresh C4 data, so
# this repetition is a documented deviation.
VI_VII_B_DATA_REPEAT=3
VI_VII_B_CHECKPOINT_ROOT=checkpoints/vii_b_mb256
VI_VII_B_LOG_ROOT=logs/vii_b_mb256

# VII-C: runtime overhead of the detection path. Reuse the first 2,000 steps
# from the saved VII-B fault-free baseline as the detection-on sample, then
# launch only a matching torchrun_main_no_detection.py detection-off run.
VI_VII_C_EXIT_AFTER=2000
VI_VII_C_EVAL_EVERY="$VI_VII_B_EVAL_EVERY"
VI_VII_C_SAVE_EVERY="$VI_VII_B_SAVE_EVERY"
# The saved VII-B 3906-step baseline segment was produced before the explicit
# repeat setting was added, so it used the default repeat=1. That is sufficient
# for the 2,000-step VII-C timing window and is the strictest match here.
VI_VII_C_DATA_REPEAT=1
VI_VII_C_WARMUP_SAMPLE_STEPS=100
VI_VII_C_DETECTION_ON_DIR=checkpoints/vii_b_mb256_3906/baseline/seed_${VI_VII_SEED}
VI_VII_C_NO_DETECTION_DIR=checkpoints/vii_c_mb256/no_detection_match_vii_b/seed_${VI_VII_SEED}
VI_VII_C_LOG_ROOT=logs/vii_c_mb256

# Kernel manifest from IV-A profiling: kernel label -> function ID mapping.
# Function IDs are only valid for the identical model/seed/batch structure.
VI_VII_KERNEL_MANIFEST=logs/iv_a_mb256/kernel_manifest.tsv
