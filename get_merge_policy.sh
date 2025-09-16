#!/bin/bash
set -e

source openr1/bin/activate

# LoRA 目录
lora_dirs=(
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/1000/checkpoint-50"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/0100/checkpoint-50"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/0010/checkpoint-50"
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/0001/checkpoint-50"
)

# 权重组合
weights_list=(
    "2 0 1 0"
    "2 1 0 0"
    "2 1 1 0"
    "3 0 1 0"
    "3 1 0 0"
    "3 1 1 0"
)

# 输出根目录
output_dir="/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy_v2"

mkdir -p "$output_dir"

# 循环运行
for weights in "${weights_list[@]}"; do
    read -r w1 w2 w3 w4 <<< "$weights"
    output_path="${output_dir}/${w1}${w2}${w3}${w4}"

    echo "🚀 Merging with weights: $w1 $w2 $w3 $w4"
    echo "   Output path: $output_path"

    # 调用 merge_safetensors.py (假设你把 merge_lora 的逻辑存成脚本)
    python scripts/lzk/weight_averaging.py \
        "${lora_dirs[0]}" "$w1" \
        "${lora_dirs[1]}" "$w2" \
        "${lora_dirs[2]}" "$w3" \
        "${lora_dirs[3]}" "$w4" \
        "$output_path"
done

echo "🎉 All merges done. Saved in $output_dir"