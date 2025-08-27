srun --pty --gres=gpu:1 --mem=32GB -p h100 bash



bash scripts/training/post_train_grpo.sh

bash scripts/slurm/eval.sh

# Train via YAML config
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --vllm_mode colocate \
    --eos_token '<|im_end|>'