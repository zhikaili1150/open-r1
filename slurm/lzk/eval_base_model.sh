#!/bin/bash

#SBATCH --job-name=eval_base_model_7B
#SBATCH --output=logs/%j_eval_base_model_7B.out
#SBATCH --error=logs/%j_eval_base_model_7B.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=16GB

# MODEL=meta-llama/Llama-3.2-1B
# MODEL=meta-llama/Llama-3.2-1B-Instruct
# MODEL=Qwen/Qwen2.5-1.5B
# MODEL=Qwen/Qwen2.5-1.5B-Instruct
# MODEL=Qwen/Qwen2.5-Math-1.5B
# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# bash slurm/lzk/eval.sh meta-llama/Llama-3.2-1B "gsm8k|8|0" no chat template
# bash slurm/lzk/eval.sh meta-llama/Llama-3.2-1B-Instruct "gsm8k|8|0"
# bash slurm/lzk/eval.sh Qwen/Qwen2.5-1.5B "gsm8k|8|0"
# bash slurm/lzk/eval.sh Qwen/Qwen2.5-1.5B-Instruct "gsm8k|8|0"
# bash slurm/lzk/eval.sh Qwen/Qwen2.5-Math-1.5B "gsm8k|8|0"
# bash slurm/lzk/eval.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B "gsm8k|8|0"

# bash slurm/lzk/eval.sh meta-llama/Llama-3.1-8B-Instruct "gsm8k|0|0"
# bash slurm/lzk/eval.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B "gsm8k|0|0"
# bash slurm/lzk/eval.sh Qwen/Qwen2.5-7B "gsm8k|0|0"
# bash slurm/lzk/eval.sh Qwen/Qwen2.5-7B-Instruct "gsm8k|0|0"
# bash slurm/lzk/eval.sh Qwen/Qwen2.5-Math-7B "gsm8k|0|0"
bash slurm/lzk/eval.sh Qwen/Qwen2.5-Math-7B-Instruct "gsm8k|0|0"
# bash slurm/lzk/eval.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B "gsm8k|0|0"

