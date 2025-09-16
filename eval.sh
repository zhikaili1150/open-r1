#!/bin/bash

#SBATCH --job-name=eval_baseline_reproduce
#SBATCH --output=logs/%j_eval_baseline_reproduce.out
#SBATCH --error=logs/%j_eval_baseline_reproduce.err
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
EXP_NAME="baseline_reproduce"

RESULTS_DIR="/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/results/$EXP_NAME"
mkdir -p "$RESULTS_DIR"

# =========================
# Mode selection
# =========================
# Mode 1: specify a directory containing LoRA checkpoints
LORA_DIR_LIST=(
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward_v2


    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/001"
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/010"
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/100"
    )

# Mode 2: manually specify LoRA checkpoint list
LORA_LIST=(

    # knoveleng/Open-RS2
    # knoveleng/Open-RS2
    # knoveleng/Open-RS2

    knoveleng/Open-RS3
    knoveleng/Open-RS3
    knoveleng/Open-RS3

    SpiceRL/DRA-DR.GRPO
    SpiceRL/DRA-DR.GRPO
    SpiceRL/DRA-DR.GRPO

    # SpiceRL/DRA-GRPO
    # SpiceRL/DRA-GRPO
    # SpiceRL/DRA-GRPO

    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/0001/checkpoint-50
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/0010/checkpoint-50
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/0100/checkpoint-50
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/1000/checkpoint-50


    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/011
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/101
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/110
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/111


    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/2010
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/2100
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/2110
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3010
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3100
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3110

    # mv /local/scratch/zli2255/workspace/open-r1/data/1.5B_1e-6/merge_policy /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3100
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3110
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3140

    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/310
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/310
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/310

    # Nickyang/FastCuRL-1.5B-V3

    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3100
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3100
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3100

    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3110
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3110
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3110

    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3140
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3140
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2/3140




    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/0001/checkpoint-50"
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward_v2/310/checkpoint-50"

    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/011
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/101
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/110
    # /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/111
)


# Build final LoRA path list
if [ ${#LORA_LIST[@]} -gt 0 ]; then
    # If LORA_LIST is not empty, use it
    LORA_PATHS=("${LORA_LIST[@]}")
else
    # Otherwise, find all subdirectories under each LORA_DIR
    LORA_PATHS=()
    for dir in "${LORA_DIR_LIST[@]}"; do
        # Collect all subdirectories
        for sub in "$dir"/*/; do
            [ -d "$sub" ] && LORA_PATHS+=("${sub%/}")  # 去掉结尾的斜杠
        done
    done
fi

# =========================
# Evaluation
# =========================

TASKS=("aime24" "math_500" "amc23" "minerva" "olympiadbench")

# LIGHT_EVAL_TASKS=$(printf "custom|%s|0|0, " "${TASKS[@]}")
# LIGHT_EVAL_TASKS=${LIGHT_EVAL_TASKS%, }
# LIGHT_EVAL_TASKS="custom|aime24|0|0, custom|math_500|0|0, custom|amc23|0|0, custom|minerva|0|0, custom|olympiadbench|0|0"
# echo $LIGHT_EVAL_TASKS


# CSV file
TIMING_CSV_FILE="$RESULTS_DIR/eval_times.csv"

if [ ! -f "$TIMING_CSV_FILE" ]; then
    echo "model,task,start_time,end_time,duration" > "$TIMING_CSV_FILE"
fi

for LORA_PATH in "${LORA_PATHS[@]}"; do
    # echo "=========================================="
    # echo ">>> Evaluating LoRA: $LORA_PATH"
    # echo "=========================================="

    # MERGED_PATH="$LORA_PATH/merged_model"
    # echo ">>> Merging LoRA into base model with merge_lora_model.py ..."
    # mkdir -p "$MERGED_PATH"
    # python scripts/lzk/merge_lora_model.py "$BASE_MODEL" "$LORA_PATH" "$MERGED_PATH"

    # For base model, we don't need to merge
    BASE_MODEL=$LORA_PATH

    # Model configuration
    # MODEL_ARGS="model_name=$MERGED_PATH,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
    MODEL_ARGS="model_name=$BASE_MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:1.0}"


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

    # rm -rf $MERGED_PATH
done
