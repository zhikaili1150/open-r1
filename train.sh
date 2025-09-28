#!/bin/bash

#SBATCH --job-name=exp_drgrpo
#SBATCH --output=logs/%j_exp_drgrpo_training.out
#SBATCH --error=logs/%j_exp_drgrpo_training.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB

# uv
source openr1/bin/activate

# conda
# conda activate openr1


START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Mode 1: specify directory, automatically collect yaml files
CONFIG_DIR=""
CONFIG_FILES=($(find "$CONFIG_DIR" -mindepth 1 -maxdepth 1 -type f -name "*.yaml"))

# Mode 2: manually specify config files
CONFIG_FILES=(
)

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    echo "🚀 Launching with config: $CONFIG_FILE"
    accelerate launch \
        --config_file recipes/accelerate_configs/zero2.yaml \
        src/open_r1/grpo.py \
        --config "$CONFIG_FILE" \
        --vllm_mode colocate
done


END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"