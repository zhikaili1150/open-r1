#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# conda
source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh
conda activate openr1

BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 父目录
CKPT_PARENT="data/OpenRS-GRPO"

# 找到所有子目录 (例如 210, 211)
CKPT_DIRS=($(find "$CKPT_PARENT" -mindepth 1 -maxdepth 1 -type d))

# 自定义CKPT_DIRS
# CKPT_DIRS=( "data/OpenRS-GRPO/211" )

# 定义 evaluation tasks
TASKS="aime24 math_500 amc23 minerva olympiadbench"

# 总表 CSV
GLOBAL_CSV="logs/evals/all_eval_times.csv"
mkdir -p "logs/evals"
if [ ! -f "$GLOBAL_CSV" ]; then
    echo "model,step,task,start_time,end_time,duration_sec" > "$GLOBAL_CSV"
fi

for ckpt_dir in "${CKPT_DIRS[@]}"; do
    echo "=========================================="
    echo ">>> Processing ckpt_dir: $ckpt_dir"
    echo "=========================================="

    model_name=$(basename "$ckpt_dir")   # e.g. 210
    output_root="logs/evals/${model_name}"
    mkdir -p "$output_root"

    STEP_DIRS=($(find "$ckpt_dir" -mindepth 1 -maxdepth 1 -type d))
    # STEP_DIRS=( "$ckpt_dir"/checkpoint-100 )
    for step_ckpt in "${STEP_DIRS[@]}"; do
        [ -d "$step_ckpt" ] || continue
        step_name=$(basename "$step_ckpt")  # e.g. checkpoint-10
        step_num=${step_name#checkpoint-}   # e.g. 10
        echo ">>> Evaluating checkpoint: $step_name"

        merged_path="$step_ckpt/merged_model"
        if [ ! -d "$merged_path" ]; then
            echo ">>> Merging LoRA into base model: $merged_path"
            mkdir -p "$merged_path"
            python scripts/lzk/merge_lora_model.py "$BASE_MODEL" "$step_ckpt" "$merged_path"
        else
            echo ">>> Already merged: $merged_path"
        fi

        BASE_MODEL_ARGS="model_name=$merged_path,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

        outdir="$output_root/$step_name"
        mkdir -p "$outdir"

        for task in $TASKS; do
            echo "Evaluating task: $task"

            start_time=$(date +"%Y-%m-%d %H:%M:%S")
            start_sec=$(date +%s)

            lighteval vllm "$BASE_MODEL_ARGS" "custom|$task|0|0" \
                --custom-tasks src/open_r1/evaluate.py \
                --use-chat-template \
                --output-dir "$outdir"

            end_time=$(date +"%Y-%m-%d %H:%M:%S")
            end_sec=$(date +%s)
            duration=$((end_sec - start_sec))

            # 写到总表
            echo "$model_name,$step_num,$task,$start_time,$end_time,$duration" >> "$GLOBAL_CSV"
        done

        rm -rf "$merged_path"
    done
done