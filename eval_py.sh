#!/bin/bash

#SBATCH --job-name=val_v3
#SBATCH --output=logs/%j_val_v3.out
#SBATCH --error=logs/%j_val_v3.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB


export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="Find_LR"

# conda
# conda activate openr1

# uv
source openr1/bin/activate

RESULTS_DIR="experiments/exp_grpo_fft/results/RE_val_v3"
mkdir -p "$RESULTS_DIR"

# =========================
# Mode selection
# =========================
# Mode 1: specify a directory containing checkpoints
CKPT_DIR_LIST=(
    experiments/exp_grpo_fft/ckpt/reward_expert/0100_1e-05
    experiments/exp_grpo_fft/ckpt/reward_expert/0100_1e-06
    experiments/exp_grpo_fft/ckpt/reward_expert/0100_9e-07
    experiments/exp_grpo_fft/ckpt/reward_expert/0100_8e-07
    experiments/exp_grpo_fft/ckpt/reward_expert/0100_5e-07
    experiments/exp_grpo_fft/ckpt/reward_expert/0100_1e-07

    experiments/exp_grpo_fft/ckpt/reward_expert/0001_1e-05
    experiments/exp_grpo_fft/ckpt/reward_expert/0001_1e-06
    experiments/exp_grpo_fft/ckpt/reward_expert/0001_9e-07
    experiments/exp_grpo_fft/ckpt/reward_expert/0001_8e-07
    experiments/exp_grpo_fft/ckpt/reward_expert/0001_5e-07
    experiments/exp_grpo_fft/ckpt/reward_expert/0001_1e-07
    )
CKPT_LIST=()
for dir in "${CKPT_DIR_LIST[@]}"; do
    for sub in "$dir"/*/; do
        [ -d "$sub" ] && CKPT_LIST+=("${sub%/}")  
    done
done

# Mode 2: manually specify CKPT checkpoints
# CKPT_LIST=(
#     deepseek-ai/Deepseek-R1-Distill-Qwen-1.5B
# )

# =========================
# Evaluation
# =========================
# TASKS=("aime24" "math_500" "amc23" "minerva" "olympiadbench")
# TASKS=("aime24" "math_500" "amc23")
# TASKS=("math_500")
# TASKS=("aime25")
TASKS=("math_validation")

# CSV file
TIMING_CSV_FILE="$RESULTS_DIR/eval_times.csv"
if [ ! -f "$TIMING_CSV_FILE" ]; then
    echo "model,task,start_time,end_time,duration" > "$TIMING_CSV_FILE"
fi

for CKPT in "${CKPT_LIST[@]}"; do

    # if "adapter.safetensors" exists, then evaluate LoRA models
    # if "model.safetensors" exists, then evaluate FFT models
    IS_LORA=False
    if [ -f "$CKPT/adapter.safetensors" ]; then
        IS_LORA=True
    fi

    if [ $IS_LORA == True ]; then
        echo ">>> Evaluating LoRA: $CKPT"
        echo "=========================================="
        MODEL="$CKPT/merged_model"

        echo ">>> Merging LoRA into base model with merge_lora_model.py ..."
        mkdir -p "$MODEL"
        python scripts/lzk/merge_lora_model.py "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "$CKPT" "$MODEL"
    fi

    # if "model.safetensors" exists, then evaluate FFT models
    if [ $IS_LORA == False ]; then
        echo ">>> Evaluating FFT: $CKPT"
        echo "=========================================="
        MODEL=$CKPT
    fi

    for task in "${TASKS[@]}"; do
        start_time=$(date +"%Y-%m-%d %H:%M:%S")
        start_sec=$(date +%s)

        # Run evaluations for each task
        echo "Evaluating task: $task"

        output_dir="$RESULTS_DIR/$task"
        mkdir -p "$output_dir"

        python eval.py \
            --result_dir $output_dir \
            --model_name $MODEL \
            --task "custom|math_validation|0|0"

        end_time=$(date +"%Y-%m-%d %H:%M:%S")
        end_sec=$(date +%s)
        duration=$((end_sec - start_sec))

        echo "$CKPT,$task,$start_time,$end_time,$duration" >> "$TIMING_CSV_FILE"
    done

    if [ $IS_LORA == True ]; then
        rm -rf $MODEL
    fi
done

# python scripts/lzk/collect_result.py \
#     --root_dir $RESULTS_DIR \
#     --metrics extractive_match mean_token_length \
#     --tasks math_validation \
#     --output result.csv