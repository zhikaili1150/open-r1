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
# bash eval_lora.sh <base_model_path> <lora_paths...>

BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
LORA_PATH=/local/scratch/zli2255/workspace/open-r1/data/Llama-3.2-1B-Instruct-GRPO/exp6.1/llama-1e-04
MERGED_PATH="$LORA_PATH/merged_model"
TASK="gsm8k|8|0"

echo ">>> Merging LoRA into base model with merge_lora_model.py ..."
mkdir -p "$MERGED_PATH"

python merge_lora_model.py "$BASE_MODEL" "$LORA_PATH" "$MERGED_PATH"

echo ">>> Starting evaluation with lighteval..."

bash slurm/lzk/eval.sh "$MERGED_PATH" "$TASK"

# 清理合并的模型
rm -rf $MERGED_PATH
