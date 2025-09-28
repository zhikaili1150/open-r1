#!/bin/bash

#SBATCH --job-name=eval_fft
#SBATCH --output=logs/%j_eval_fft.out
#SBATCH --error=logs/%j_eval_fft.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB


export VLLM_WORKER_MULTIPROC_METHOD=spawn

# conda
# conda activate openr1

# uv
source openr1/bin/activate

BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
RESULTS_DIR="/local/scratch/zli2255/workspace/open-r1/experiments/exp_fft/results/step0"
mkdir -p "$RESULTS_DIR"

# =========================
# Mode selection
# =========================
# Mode 1: specify a directory containing checkpoints
CKPT_DIR_LIST=(
    )
CKPT_LIST=()
for dir in "${CKPT_DIR_LIST[@]}"; do
    for sub in "$dir"/*/; do
        [ -d "$sub" ] && CKPT_LIST+=("${sub%/}")  
    done
done

# Mode 2: manually specify CKPT checkpoints
CKPT_LIST=(
)

# =========================
# Evaluation
# =========================
TASKS=("aime24" "math_500" "amc23" "minerva" "olympiadbench")

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
    elif [ -f "$CKPT/model.safetensors" ]; then
        IS_LORA=False
    else
        echo ">>> Error: \"adapter.safetensors\" or \"model.safetensors\" not found in $CKPT"
        exit 1
    fi

    if [ $IS_LORA == True ]; then
        echo ">>> Evaluating LoRA: $CKPT"
        echo "=========================================="
        MODEL="$CKPT/merged_model"

        echo ">>> Merging LoRA into base model with merge_lora_model.py ..."
        mkdir -p "$MODEL"
        python scripts/lzk/merge_lora_model.py "$BASE_MODEL" "$CKPT" "$MODEL"
    fi

    # if "model.safetensors" exists, then evaluate FFT models
    if [ $IS_LORA == False ]; then
        echo ">>> Evaluating FFT: $CKPT"
        echo "=========================================="
        MODEL=$CKPT
    fi

    # Model configuration
    MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" # seed不可复现
    # MODEL_ARGS="model_name=$MODEL,seed=42,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={seed:42, max_new_tokens:32768,temperature:0.6,top_p:0.95}" # seed可复现
    # MODEL_ARGS="model_name=$MODEL,seed=42,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0,top_p:1}" #确定性采样，可以复现，但是acc降低很多


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

        echo "$CKPT,$task,$start_time,$end_time,$duration" >> "$TIMING_CSV_FILE"
    done

    if [ $IS_LORA == True ]; then
        rm -rf $MODEL
    fi
done