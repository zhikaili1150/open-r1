#!/bin/sh

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# conda
source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh
conda activate openr1

BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_PATH="data/OpenRS-GRPO/210/checkpoint-100"

echo "=========================================="
echo ">>> Evaluating LoRA: $LORA_PATH"
echo "=========================================="

MERGED_PATH="$LORA_PATH/merged_model"
MERGED_PATH="knoveleng/Open-RS2"
echo ">>> Merging LoRA into base model with merge_lora_model.py ..."
# mkdir -p "$MERGED_PATH"

# python scripts/lzk/merge_lora_model.py "$BASE_MODEL" "$LORA_PATH" "$MERGED_PATH"

# Base model configuration
BASE_MODEL_ARGS="model_name=$MERGED_PATH,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# Define evaluation tasks
# TASKS="aime24 math_500 amc23 minerva olympiadbench"
TASKS="aime24"

output_dir="logs/evals/OpenRS-210"

# Set model args with the specific revision
model_args="$BASE_MODEL_ARGS"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Run evaluations for each task
for task in $TASKS; do
    echo "Evaluating task: $task"
    lighteval vllm "$model_args" "custom|$task|0|0" \
        --custom-tasks src/open_r1/evaluate.py \
        --use-chat-template \
        --output-dir "$output_dir" 
done