#!/bin/bash

#SBATCH --job-name=exp_lr_eval
#SBATCH --output=logs/%j_exp_lr_eval.out
#SBATCH --error=logs/%j_exp_lr_eval.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB


export VLLM_WORKER_MULTIPROC_METHOD=spawn

# conda
# source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh
# conda activate openr1

# uv
source openr1/bin/activate

BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXP_NAME="exp_lr_1e-07"

# =========================
# Mode selection
# =========================
# Mode 1: specify a directory containing LoRA checkpoints
LORA_DIR=""   # e.g., "data/OpenRS-GRPO/210"

# Mode 2: manually specify LoRA checkpoint list
LORA_LIST=(
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-22"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-44"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-66"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-88"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-110"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-132"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-154"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-176"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-198"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-220"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-242"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-264"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-286"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-308"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-330"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-352"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-374"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-396"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110/checkpoint-418"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_lr/ckpt/110"

    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-40"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-80"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-120"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-160"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-200"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-240"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-280"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-320"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-360"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/mix_reward/211/checkpoint-400"

    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-40"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-80"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-120"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-160"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-200"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-240"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-280"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-320"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-360"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/merge_policy/211-110_101_1_1/checkpoint-400"

    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-40"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-80"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-120"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-160"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-200"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-240"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-280"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-320"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-360"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/101/checkpoint-400"

    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-40"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-80"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-120"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-160"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-200"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-240"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-280"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-320"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-360"
    # "experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs/110/checkpoint-400"
)


# Build final LoRA path list
if [ -n "$LORA_DIR" ]; then
    # Find all directories (assume each directory is a LoRA checkpoint)
    LORA_PATHS=($(find "$LORA_DIR" -mindepth 1 -maxdepth 1 -type d))
else
    LORA_PATHS=("${LORA_LIST[@]}")
fi

# =========================
# Evaluation
# =========================

# TASKS=("aime24" "math_500" "amc23" "minerva" "olympiadbench")
# TASKS=("aime24" "math_500" "amc23" "minerva" "olympiadbench")
# TASKS=("aime24" "amc23" "math_500")
TASKS=("math_500")

# CSV file
RESULTS_DIR="results/$EXP_NAME"
mkdir -p "$RESULTS_DIR"
TIMING_CSV_FILE="$RESULTS_DIR/eval_times.csv"

if [ ! -f "$TIMING_CSV_FILE" ]; then
    echo "model,task,start_time,end_time,duration" > "$TIMING_CSV_FILE"
fi

for LORA_PATH in "${LORA_PATHS[@]}"; do
    echo "=========================================="
    echo ">>> Evaluating LoRA: $LORA_PATH"
    echo "=========================================="

    MERGED_PATH="$LORA_PATH/merged_model"
    echo ">>> Merging LoRA into base model with merge_lora_model.py ..."
    mkdir -p "$MERGED_PATH"
    python scripts/lzk/merge_lora_model.py "$BASE_MODEL" "$LORA_PATH" "$MERGED_PATH"

    # Model configuration
    MODEL_ARGS="model_name=$MERGED_PATH,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"


    for task in "${TASKS[@]}"; do
        start_time=$(date +"%Y-%m-%d %H:%M:%S")
        start_sec=$(date +%s)

        # Run evaluations for each task
        echo "Evaluating task: $task"

        output_dir="$RESULTS_DIR/$task"
        mkdir -p "$output_dir"

        lighteval vllm "$MODEL_ARGS" "custom|$task|0|0" \
            --custom-tasks src/open_r1/evaluate.py \
            --use-chat-template \
            --output-dir "$output_dir"

        end_time=$(date +"%Y-%m-%d %H:%M:%S")
        end_sec=$(date +%s)
        duration=$((end_sec - start_sec))

        echo "$LORA_PATH,$task,$start_time,$end_time,$duration" >> "$TIMING_CSV_FILE"
    done

    rm -rf $MERGED_PATH
done
