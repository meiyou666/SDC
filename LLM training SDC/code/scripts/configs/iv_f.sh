#!/usr/bin/env bash

# Simplified paper Section IV-F campaign: 3 seeds, 18 total runs.
IVF_CONFIG_REVISION=3

# The paper does not publish its ten concrete seed values. Keep the existing
# repository seed and add two fixed seeds; edit this single list if needed.
IVF_SEEDS=(351344 42 1337)

IVF_MODEL=llama_60m
IVF_MAX_LENGTH=256
IVF_BATCH_SIZE=256
IVF_TOTAL_BATCH_SIZE=512
IVF_LR=1e-3
IVF_TRAINING_SCHEDULE_STEPS=100000
IVF_WARMUP_STEPS=1000
IVF_EXIT_AFTER=1000
# No periodic validation is needed inside this 1,000-step campaign. The
# trainer performs one final validation after all 1,000 updates; using 1000
# here would evaluate once in the loop and then repeat the same evaluation.
IVF_EVAL_EVERY=100000
IVF_SAVE_EVERY=1000
IVF_WORKERS=0

# Paper IV-F: bit position 13, first input register, one SM and one lane.
IVF_TARGET_BITMASK=8192
IVF_TARGET_REGISTER=1
IVF_TARGET_SMID=0
IVF_TARGET_LANEID=0
IVF_TARGET_OP=HMMA

# The paper labels the fixed sensitive backward kernel BP9 but does not publish
# its SASS kernel name. Function IDs are dynamic and changed across our runs.
# This is the backward GEMM kernel verified by the local v3 NVBit smoke test;
# using its name keeps the target stable across the 18 runs on this environment.
IVF_TARGET_FUNC_CONTAINS=ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn
IVF_TARGET_FUNC=-1

# Section IV-F does not specify an instruction index or state that all HMMA
# instructions are targeted. Revision 2 therefore uses the instruction
# verified by the local v3 smoke test instead of inferring TARGET_INSTR=-1
# from Section IV-B.
IVF_TARGET_INSTR=312

IVF_CHECKPOINT_ROOT=checkpoints/iv_f_mb256
IVF_LOG_ROOT=logs/iv_f_mb256
