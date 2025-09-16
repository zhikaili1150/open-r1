import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

folder_path_list = [
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/011/checkpoint-25",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/011/checkpoint-50",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/101/checkpoint-25",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/101/checkpoint-50",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/110/checkpoint-25",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/110/checkpoint-50",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/111/checkpoint-25",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/merge_policy/111/checkpoint-50",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/011/checkpoint-25",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/011/checkpoint-50",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/101/checkpoint-25",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/101/checkpoint-50",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/110/checkpoint-25",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/110/checkpoint-50",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/111/checkpoint-25",
    # "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/mix_reward/111/checkpoint-50",

    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/001/checkpoint-50",
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/010/checkpoint-50",
    "/local/scratch/zli2255/workspace/open-r1/experiments/exp_single_reward/ckpt/100/checkpoint-50",
]
path_in_repo_list = [
    "/".join(folder_path.split("/")[-2:])  # 取最后 3 级目录
    for folder_path in folder_path_list
]

for folder_path, path_in_repo in zip(folder_path_list, path_in_repo_list):
    api.upload_folder(
        folder_path=folder_path,
        repo_id="Zachary1150/1.5B_1e-6",
        repo_type="model",
        path_in_repo=path_in_repo,
    )