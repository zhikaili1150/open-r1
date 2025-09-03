#!/bin/bash

#SBATCH --job-name=eval1_dsllama8b_openrs
#SBATCH --output=logs/%j_eval1_dsllama8b_openrs.out
#SBATCH --error=logs/%j_eval1_dsllama8b_openrs.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24GB

source openr1/bin/activate

CKPT_DIR="experiments/exp14_dsqwen7b_openrs/ckpt/dsqwen7b_openrs"

##################################
# Mix Reward Dir
##################################
MIX_REWARD_DIR="$CKPT_DIR/mix_reward"
mkdir -p "$MIX_REWARD_DIR"

# check if 211 exists
if [ -d "$CKPT_DIR/211" ]; then
    echo "Directory 211 found in $CKPT_DIR, proceeding to move other dirs..."

    # keep directories
    KEEP_DIRS=("101" "110" "201" "210")

    # loop through all subdirectories in CKPT_DIR
    for dir in "$CKPT_DIR"/*/; do
        dirname=$(basename "$dir")
        
        # if not in keep list, move to mix_reward
        if [[ ! " ${KEEP_DIRS[@]} " =~ " $dirname " ]]; then
            echo "Moving $dirname to mix_reward"
            mv "$CKPT_DIR/$dirname" "$MIX_REWARD_DIR/"
        fi
    done
else
    echo "Directory 211 does not exist in $CKPT_DIR, skipping move."
fi


##################################
# Merge Policy Dir
##################################
MERGE_POLICY_DIR="$CKPT_DIR/merge_policy"
mkdir -p "$MERGE_POLICY_DIR"

MERGE_SCRIPT="scripts/lzk/weight_averaging.py"
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


# for pair in "${MERGE_PAIRS[@]}"; do
#   echo "🚀 Processing: $pair"
#   python $MERGE_SCRIPT $pair
#   echo ""
# done


##################################
# Eval Mix Reward and Merge Policy
##################################

export VLLM_WORKER_MULTIPROC_METHOD=spawn

BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# LoRA directory list
LORA_DIRS=("$MIX_REWARD_DIR" "$MERGE_POLICY_DIR")
METHODS=("mix_reward" "merge_policy")

# Evaluation tasks
TASKS=("aime24")
output_dir="results/exp14_dsqwen7b_openrs"
mkdir -p "$output_dir"

# CSV file
CSV_FILE="$output_dir/eval_times.csv"
echo "method,lora_path,task,running_time_sec,start_time,end_time" > "$CSV_FILE"

# loop through mix_reward and merge_policy two directories
for idx in "${!LORA_DIRS[@]}"; do
    DIR="${LORA_DIRS[$idx]}"
    METHOD="${METHODS[$idx]}"

    [ -d "$DIR" ] || continue
    echo "Processing directory: $DIR"

    # loop through LoRA subdirectories
    LORA_PATHS=($(find "$DIR" -mindepth 1 -maxdepth 1 -type d))
    for LORA_PATH in "${LORA_PATHS[@]}"; do
        echo "=========================================="
        echo ">>> Evaluating LoRA: $LORA_PATH"
        echo "=========================================="

        MERGED_PATH="$LORA_PATH/merged_model"
        mkdir -p "$MERGED_PATH"
        python scripts/lzk/merge_lora_model.py "$BASE_MODEL" "$LORA_PATH" "$MERGED_PATH"

        # Base model configuration
        BASE_MODEL_ARGS="model_name=$MERGED_PATH,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

        # loop through tasks
        for task in "${TASKS[@]}"; do
            echo "Evaluating task: $task"

            # record start time
            START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
            START_TS=$(date +%s)

            lighteval vllm "$BASE_MODEL_ARGS" "custom|$task|0|0" \
                --custom-tasks src/open_r1/evaluate.py \
                --use-chat-template \
                --output-dir "$output_dir"

            # record end time
            END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
            END_TS=$(date +%s)
            RUNNING_TIME=$((END_TS - START_TS))

            # write to CSV
            echo "$METHOD,$LORA_PATH,$task,$RUNNING_TIME,$START_TIME,$END_TIME" >> "$CSV_FILE"
        done

        # clean merged model
        rm -rf "$MERGED_PATH"
    done
done