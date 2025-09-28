# 1 Installation

  

To run the code in this project, first, create a Python virtual environment using e.g. `uv`.

To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

  

```shell

uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip

```

  

> [!TIP]
> For Hugging Face cluster users, add `export UV_LINK_MODE=copy` to your `.bashrc` to suppress cache warnings from `uv`

  

Next, install vLLM and FlashAttention:

  

```shell

uv pip install vllm==0.8.5.post1

uv pip install setuptools && uv pip install flash-attn==2.7.4.post1 --no-build-isolation

```

  

This will also install PyTorch `v2.6.0` and it is **very important** to use this version since the vLLM binaries are compiled for it. You can then install the remaining dependencies for your specific use case via `pip install -e .[LIST OF MODES]`. For most contributors, we recommend:

  

```shell

GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"

```

  

Next, log into your Hugging Face and Weights and Biases accounts as follows:

  

```shell

huggingface-cli login

wandb login

```

  

Finally, check whether your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

  

```shell

git-lfs --version

```

  

If it isn't installed, run:

  

```shell

sudo apt-get install git-lfs

```

  

# 2 Training models

  

We provide a training script `train.sh` to simplify the process.

  

## 2.1 Modify training script

  

You can launch training based on config files in two ways:

  

1. **Folder Mode**

Specify a folder, and all `.yaml` files inside will be automatically discovered.

  

2. **File Mode**

Directly specify one or more `.yaml` files.

  

> [!TIP]
> Please choose **only one** of the above methods.
> Comment out the code for the other method.

  

```bash

# Mode 1: specify directory, automatically collect yaml files

CONFIG_DIR="experiments/exp_fft/config/mix_reward/2acc1fmt"

CONFIG_FILES=($(find "$CONFIG_DIR" -mindepth 1 -maxdepth 1 -type f -name "*.yaml"))

  

# Mode 2: manually specify config files

# CONFIG_FILES=(

# experiments/exp_fft/config/mix_reward/2acc1fmt/config_lr_1e-05.yaml

# experiments/exp_fft/config/mix_reward/2acc1fmt/config_lr_1e-06.yaml

# )

```

  

## 2.2 Execute the script


```bash
bash train.sh
```

The model will be saved in the path specificed by `output_dir` in `.yaml` config.

# 3 Evaluate Models

We provide an evaluation script `eval.sh` to simplify the process.

## 3.1 Specify `RESULTS_DIR`

Specify the path to save evaluation results by setting `RESULTS_DIR`:

```bash
RESULTS_DIR="experiments/exp_fft/results/<eval_name>"
````

## 3.2 Specify Checkpoints

Checkpoints can be either **LoRA** or **FFT models**, depending on whether the folder contains `adapter.safetensors` or `model.safetensors`.

There are two ways to specify checkpoints (inside the checkpoint directory):

1. **Directory List Mode**  
	Specify a list of directories, each containing multiple checkpoints at different training step. This is suitable for evaluating all training steps of multiple models.
    
2. **Manual Checkpoint List Mode**  
    Specify a list of checkpoints directly. This is suitable for evaluating specific training steps.
    

```bash
# Mode 1: specify a directory containing checkpoints
CKPT_DIR_LIST=(
	experiments/exp_fft/ckpt/mix_reward/1e-06
	experiments/exp_fft/ckpt/mix_reward/1e-07
)

CKPT_LIST=()

for dir in "${CKPT_DIR_LIST[@]}"; do
    for sub in "$dir"/*/; do
        [ -d "$sub" ] && CKPT_LIST+=("${sub%/}")
    done
done

# Mode 2: manually specify CKPT checkpoints
# CKPT_LIST=(
# 	experiments/exp_fft/ckpt/mix_reward/1e-06/checkpoint-50
#  	experiments/exp_fft/ckpt/mix_reward/1e-07/checkpoint-50
#)
```

> [!TIP]
> Please choose **only one** of the above methods.
> Comment out the code for the other method.


## 3.3 Specify Tasks

```bash
TASKS=("aime24" "math_500" "amc23" "minerva" "olympiadbench")
```

## 3.4 Run the Script

```bash
bash eval.sh
```

The results will be saved in `$RESULTS_DIR/result.csv`.