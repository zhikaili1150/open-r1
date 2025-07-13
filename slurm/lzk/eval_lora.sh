#!/bin/bash

# 用法:
# bash eval_lora.sh <base_model_path> <lora_paths...>

BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# 父目录
PARENT_DIR="data/DeepSeek-R1-Distill-Qwen-1.5B-GRPO/exp5_lr"

# 清空数组
LORA_PATHS=()

# 遍历 PARENT_DIR 下所有一级子目录（仅目录）
for dir in "$PARENT_DIR"/*/ ; do
    # 去掉结尾的斜杠，存进数组
    LORA_PATHS+=("${dir%/}")
done

printf '%s\n' "${LORA_PATHS[@]}"

source openr1/bin/activate

# 遍历每个LoRA路径
for LORA_PATH in "${LORA_PATHS[@]}"; do
    echo "=========================================="
    echo ">>> Evaluating LoRA: $LORA_PATH"
    echo "=========================================="
    
    MERGED_PATH="$LORA_PATH/merged_model"
    
    echo ">>> Merging LoRA into base model with merge_lora_model.py ..."
    mkdir -p "$MERGED_PATH"
    
    python merge_lora_model.py "$BASE_MODEL" "$LORA_PATH" "$MERGED_PATH"
    
    echo ">>> Starting evaluation with lighteval..."
    
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    
    MODEL="$MERGED_PATH"
    MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:2048,temperature:0.6,top_p:0.95}"
    OUTPUT_DIR=.
    
    TASK=gsm8k
    
    lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR
    
    # 清理合并的模型
    rm -rf $MERGED_PATH
    
    echo "=========================================="
    echo ">>> Finished: $LORA_PATH"
    echo "=========================================="
    echo ""
done

echo "🎉 All LoRA evaluations completed!"
