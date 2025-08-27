#!/bin/bash

#SBATCH --job-name=lzk_grpo
#SBATCH --output=logs/%j_grpo_training.out
#SBATCH --error=logs/%j_grpo_training.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB

source ~/.bashrc
source openr1/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    src/open_r1/grpo.py \
    --config recipes/meta-llama/Llama-3.2-1B-Instruct/grpo/exp7_assistantthink/config_demo.yaml \
    --vllm_mode colocate

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"