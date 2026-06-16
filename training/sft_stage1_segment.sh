#!/bin/bash
# Stage 1: Segment-level SFT training
# Recommended: 6 x 80GB A100 GPUs

OUTPUT_DIR="./output/segment-sft"

export MAX_PIXELS=1003520
export NPROC_PER_NODE=6
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export FORCE_QWENVL_VIDEO_READER=decord
# Optimize data loading to reduce GPU idle time
export OMP_NUM_THREADS=16
export NUMEXPR_MAX_THREADS=32

swift sft \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --dataset ./data/train_segments.jsonl \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file false \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --padding_free true \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --max_length 8192 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataset_num_proc 16 \
    --deepspeed zero3 \
    --dataloader_num_workers 8 \
    --lazy_tokenize true
