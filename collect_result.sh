source openr1/bin/activate

python scripts/lzk/collect_result.py \
    --root_dir /local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/results/fastcurl_v3 \
    --metrics extractive_match mean_token_length \
    --output result.csv