#!/bin/bash

#SBATCH --job-name=exp3
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high

#SBATCH -e logs/%j-exp3.err              # File to redirect stderr
#SBATCH -o logs/%j-exp3.out              # File to redirect stdout
#SBATCH --mem=10GB                   # Memory per processor
#SBATCH --time=24:00:00              # The walltime
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks-per-node=1          # number of MP tasks
#SBATCH --cpus-per-task=12           # CPUs per task
#SBATCH --gres=gpu:1                 # Number of GPUs

# source env.sh

# conda
source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh
conda activate openr1


START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Mode 1: specify directory, automatically collect yaml files
# CONFIG_DIR="experiments/03_dsqwen1b_openrs/config"
# CONFIG_FILES=($(find "$CONFIG_PARENT" -mindepth 1 -maxdepth 1 -type f -name "*.yaml"))


# Mode 2: manually specify yaml file list
CONFIG_FILES=(
"experiments/03_dsqwen1b_openrs/config/config_reward_411.yaml"
"experiments/03_dsqwen1b_openrs/config/config_reward_412.yaml"
"experiments/03_dsqwen1b_openrs/config/config_reward_421.yaml"
"experiments/03_dsqwen1b_openrs/config/config_reward_413.yaml"
"experiments/03_dsqwen1b_openrs/config/config_reward_431.yaml"
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