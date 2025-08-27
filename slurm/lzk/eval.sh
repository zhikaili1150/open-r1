# leaderboard|gsm8k
source openr1/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
# MODEL=meta-llama/Llama-3.2-1B
# MODEL=meta-llama/Llama-3.2-1B-Instruct
# MODEL=Qwen/Qwen2.5-Math-1.5B
# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL=$1
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=$2

# gsm8k
TASK=$3
lighteval vllm $MODEL_ARGS "lighteval|$TASK" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# lighteval vllm $MODEL_ARGS "lighteval|$TASK" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# lighteval vllm $MODEL_ARGS ./mmlu_stem.txt \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR