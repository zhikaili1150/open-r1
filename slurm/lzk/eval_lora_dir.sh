#!/bin/bash

#SBATCH --job-name=lzk_exp4
#SBATCH --output=logs/%j_exp4_eval_merged_policy.out
#SBATCH --error=logs/%j_exp4_eval_merged_policy.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB

source ~/.bashrc
source openr1/bin/activate

# 用法:
# bash eval_lora.sh <base_model_path> <lora_paths...> <output_dir> <task>

# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# PARENT_DIR="data/merged_policy"
# TASK="gsm8k|8|0"

BASE_MODEL=$1
PARENT_DIR=$2
OUTPUT_DIR=$3
TASK=$4


LORA_PATHS=()
# 遍历 PARENT_DIR 下所有一级子目录（仅目录）
for dir in "$PARENT_DIR"/*/ ; do
    # 去掉结尾的斜杠，存进数组
    LORA_PATHS+=("${dir%/}")
done

printf '%s\n' "${LORA_PATHS[@]}"

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
    
    bash slurm/lzk/eval.sh "$MERGED_PATH" "$OUTPUT_DIR" "$TASK" 
    
    # 清理合并的模型
    rm -rf $MERGED_PATH
    
    echo "=========================================="
    echo ">>> Finished: $LORA_PATH"
    echo "=========================================="
    echo ""
done

echo "🎉 All LoRA evaluations completed!"
