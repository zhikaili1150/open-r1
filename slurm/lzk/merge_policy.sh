#!/bin/bash

# Python 脚本路径
MERGE_SCRIPT="scripts/lzk/weight_averaging.py"
BASE_DIR=$1
OUTPUT_DIR=$2

# 定义要合并的多个组合（每一组包含两个LoRA路径、两个权重、输出目录）
# 格式: "lora1_path lora2_path weight1 weight2 output_dir"

# 110, 101, 1, 1
# 110, 201, 1, 1 / 101, 210, 1, 1
# 2 * 110 + 101
# 110 + 2 * 101
# 210 + 201
# 110 + 101 + 210
# 110+101+201
# 3*110+101
# 110+3*101

MERGE_PAIRS=(
  "$BASE_DIR/110 1.0 $BASE_DIR/101 1.0 $OUTPUT_DIR/211-110_101_1_1"
  # "$BASE_DIR/110 1.0 $BASE_DIR/201 1.0 $OUTPUT_DIR/311-110_201_1_1"
  # "$BASE_DIR/101 1.0 $BASE_DIR/210 1.0 $OUTPUT_DIR/311-101_210_1_1"

  # "$BASE_DIR/110 2.0 $BASE_DIR/101 1.0 $OUTPUT_DIR/321-110_101_2_1"
  # "$BASE_DIR/110 1.0 $BASE_DIR/101 2.0 $OUTPUT_DIR/312-110_101_1_2"

  # "$BASE_DIR/210 1.0 $BASE_DIR/201 1.0 $OUTPUT_DIR/411-210_201_1_1"

  # "$BASE_DIR/110 1.0 $BASE_DIR/101 1.0 $BASE_DIR/210 1.0 $OUTPUT_DIR/421-110_101_210_1_1_1"
  # "$BASE_DIR/110 1.0 $BASE_DIR/101 1.0 $BASE_DIR/201 1.0 $OUTPUT_DIR/412-110_101_201_1_1_1"

  # "$BASE_DIR/110 3.0 $BASE_DIR/101 1.0 $OUTPUT_DIR/431-110_101_3_1"
  # "$BASE_DIR/110 1.0 $BASE_DIR/101 3.0 $OUTPUT_DIR/413-110_101_1_3"
)


# source openr1/bin/activate
source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh
conda activate openr1

# 循环执行每组合并
for pair in "${MERGE_PAIRS[@]}"; do
  echo "🚀 正在处理: $pair"
  python $MERGE_SCRIPT $pair
  echo ""
done
