#!/bin/bash

# Generate merge policy step for each merge policy
# The output directory is the same as the merge policy directory
# The tree structure could be

# CKPT_DIR/
# ├── merge_policy
# │   ├── 211-110_101_1_1
# │   │   ├── checkpoint-10
# │   │   ├── checkpoint-20
# │   │   ├── ...
# │   │   ├── checkpoint-100
# │   ├── 311-110_201_1_1
# │   │   ├── checkpoint-10
# │   │   ├── checkpoint-20
# │   │   ├── ...
# │   │   ├── checkpoint-100
# │   ├── ...

source openr1/bin/activate

CKPT_DIR="experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs"
# CKPT_DIR=$1

MERGE_POLICY_DIR="$CKPT_DIR/merge_policy"
MERGE_SCRIPT="scripts/lzk/weight_averaging.py"

mkdir -p "$MERGE_POLICY_DIR"

# 定义 merge 组合
MERGE_PAIRS=(
  "$CKPT_DIR/110 1.0 $CKPT_DIR/101 1.0 $MERGE_POLICY_DIR/211-110_101_1_1"
  "$CKPT_DIR/110 1.0 $CKPT_DIR/201 1.0 $MERGE_POLICY_DIR/311-110_201_1_1"
  "$CKPT_DIR/101 1.0 $CKPT_DIR/210 1.0 $MERGE_POLICY_DIR/311-101_210_1_1"

  "$CKPT_DIR/110 2.0 $CKPT_DIR/101 1.0 $MERGE_POLICY_DIR/321-110_101_2_1"
  "$CKPT_DIR/110 1.0 $CKPT_DIR/101 2.0 $MERGE_POLICY_DIR/312-110_101_1_2"

  "$CKPT_DIR/210 1.0 $CKPT_DIR/201 1.0 $MERGE_POLICY_DIR/411-210_201_1_1"

  "$CKPT_DIR/110 1.0 $CKPT_DIR/101 1.0 $CKPT_DIR/210 1.0 $MERGE_POLICY_DIR/421-110_101_210_1_1_1"
  "$CKPT_DIR/110 1.0 $CKPT_DIR/101 1.0 $CKPT_DIR/201 1.0 $MERGE_POLICY_DIR/412-110_101_201_1_1_1"

  "$CKPT_DIR/110 3.0 $CKPT_DIR/101 1.0 $MERGE_POLICY_DIR/431-110_101_3_1"
  "$CKPT_DIR/110 1.0 $CKPT_DIR/101 3.0 $MERGE_POLICY_DIR/413-110_101_1_3"
)

# 遍历 MERGE_PAIRS
for pair in "${MERGE_PAIRS[@]}"; do
    # 拆分每个元素（ckpt_path + weight）和最后的输出目录
    args=($pair)
    output_dir="${args[-1]}"  # 最后一个是输出目录
    mkdir -p "$output_dir"

    # 剩下的就是 ckpt + weight 对
    ckpt_args=("${args[@]:0:${#args[@]}-1}")

    # 找出所有 checkpoint step，按第一个 ckpt 的子目录遍历
    first_ckpt_path="${ckpt_args[0]}"  # 第一个 ckpt 路径
    for step_dir in "$first_ckpt_path"/*/; do
        step_name=$(basename "$step_dir")
        step_output="$output_dir/$step_name"
        mkdir -p "$step_output"

        # 构造 python merge 命令
        merge_cmd="python $MERGE_SCRIPT"
        for ckpt_idx in $(seq 0 2 $((${#ckpt_args[@]}-1))); do
            ckpt_path="${ckpt_args[$ckpt_idx]}/$step_name"
            weight="${ckpt_args[$((ckpt_idx+1))]}"
            merge_cmd="$merge_cmd $ckpt_path $weight"
        done
        merge_cmd="$merge_cmd $step_output"

        echo "Running: $merge_cmd"
        $merge_cmd
    done
done