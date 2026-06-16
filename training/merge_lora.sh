#!/bin/bash
# Merge LoRA adapter weights into the base model

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters YOUR_LORA_CHECKPOINT_PATH \
    --merge_lora true \
    --output-dir ./output/merged_model
