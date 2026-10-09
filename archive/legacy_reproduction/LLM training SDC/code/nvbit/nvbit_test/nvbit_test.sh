LD_PRELOAD=../1.7.6_nvbit_release/tools/fault_injection/fault_injection.so \
    TOOL_VERBOSE=0 \
    TARGET_OP=FMUL \
    python ../../torchrun_main.py \
    --single_gpu \
    --model_config ../../configs/llama_60m.json \
    --lr 9e-4 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --optimizer adamw \
    --tokenizer_path ../../tokenizer/t5-base \
    --train_data_path ../../data/c4_subset_2M.json.gz \
    --val_data_path ../../data/c4_validation.jsonl \
    --save_dir ../../checkpoints/nvbit_test \
    --save_every 1000 \
    --exit_after 100 > logs/out.txt
