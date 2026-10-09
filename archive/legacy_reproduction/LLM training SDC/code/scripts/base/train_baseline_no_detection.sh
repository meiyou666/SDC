#!/bin/bash

source scripts/configs/common.sh

python torchrun_main_no_detection.py \
  --single_gpu \
  --seed "$SEED" \
  --model_config "configs/${MODEL}.json" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --total_batch_size 512 \
  --num_training_steps 100000 \
  --eval_every "$EVAL_EVERY" \
  --exit_after "$EXIT_AFTER" \
  --grad_clipping 1.0 \
  --warmup_steps 1000 \
  --weight_decay 0.01 \
  --dtype bfloat16 \
  --optimizer adamw \
  --tokenizer_path "$TOKENIZER_PATH" \
  --train_data_path "$TRAIN_DATA_PATH" \
  --val_data_path "$VAL_DATA_PATH" \
  --save_dir "checkpoints/base_runs/${MODEL}_${SEED}_${LR}" \
  --save_every 1000 \
  --name "${MODEL}_${SEED}_base_no_detection"
