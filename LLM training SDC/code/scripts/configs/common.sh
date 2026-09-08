#!/bin/bash

export SEED=351344
export MODEL=llama_60m
export BATCH_SIZE=256
export LR=1e-3

export EVAL_EVERY=1000
export EXIT_AFTER=10001

# Offline paths: place the downloaded T5 tokenizer and local C4 files here.
export TOKENIZER_PATH="./tokenizer/t5-base"
export TRAIN_DATA_PATH="./data/c4_subset_2M.json.gz"
export VAL_DATA_PATH="./data/c4_validation.jsonl"

# Enforce offline behavior even if another Transformers/Hugging Face call is
# added later. All required assets must already exist at the paths above.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
