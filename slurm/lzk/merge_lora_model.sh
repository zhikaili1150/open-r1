#!/bin/bash

# 用法:
# bash merge_lora.sh <base_model_path> <lora_path> <merged_model_path>

source openr1/bin/activate

BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
LORA_PATH=data/DeepSeek-R1-Distill-Qwen-1.5B-GRPO/2accuracy1format_ga=1
MERGED_PATH=data/merged_model/DeepSeek-R1-Distill-Qwen-1.5B-GRPO/2accuracy1format_ga=1

mkdir -p "$MERGED_PATH"

echo ">>> Running merge_lora_model.py ..."
python merge_lora_model.py "$BASE_MODEL" "$LORA_PATH" "$MERGED_PATH"
