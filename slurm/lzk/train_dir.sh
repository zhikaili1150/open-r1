

source ~/.bashrc
source openr1/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

export ACCELERATE_LOG_LEVEL=info

CONFIG_DIR=$1

for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
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