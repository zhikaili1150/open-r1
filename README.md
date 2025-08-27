# Open R1

*A fully open reproduction of DeepSeek-R1. This repo is a work in progress, let's build it together!*

**Table of Contents**  
1. [Overview](#overview)  
2. [Plan of attack](#plan-of-attack)  
3. [Installation](#installation)  
4. [Training models](#training-models)  
   - [SFT](#sft)  
   - [GRPO](#grpo)  
5. [Evaluating models](#evaluating-models)  
6. [Reproducing Deepseek's evaluation results](#reproducing-deepseeks-evaluation-results)  
7. [Data generation](#data-generation)  
   - [Generate data from a smol distilled R1 model](#generate-data-from-a-smol-distilled-r1-model)  
   - [Generate data from DeepSeek-R1](#generate-data-from-deepseek-r1)  
8. [Contributing](#contributing)

## Overview

The goal of this repo is to build the missing pieces of the R1 pipeline such that everybody can reproduce and build on top of it. The project is simple by design and mostly consists of:


- `src/open_r1`: contains the scripts to train models as well as generate synthetic data:
    - `grpo.py`: trains a model with GRPO on a given dataset.
    - `sft.py`: performs a simple SFT of a model on a dataset.
    - `generate.py`: generates synthetic data from a model using [Distilabel](https://github.com/argilla-io/distilabel).
- `Makefile`: contains easy-to-run commands for each step in the R1 pipeline leveraging the scripts above.

### Plan of attack

We will use the DeepSeek-R1 [tech report](https://github.com/deepseek-ai/DeepSeek-R1) as a guide, which can roughly be broken down into three main steps:

* Step 1: replicate the R1-Distill models by distilling a high-quality corpus from DeepSeek-R1.
* Step 2: replicate the pure RL pipeline that DeepSeek used to create R1-Zero. This will likely involve curating new, large-scale datasets for math, reasoning, and code.
* Step 3: show we can go from base model to RL-tuned via multi-stage training.

<center>
    <img src="assets/plan-of-attack.png" width="500">
</center>

## News 🗞️

* **🧑‍🍳 [2025/05/26] (Step 1 completed!)** We release [**Mixture-of-Thoughts**](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts)--a curated reasoning dataset of 350k verified traces distilled from R1. The dataset spans tasks in mathematics, coding, and science, and is designed to teach language models to reason step-by-step. We also provide a recipe to train [OpenR1-Distill-7B](https://huggingface.co/open-r1/OpenR1-Distill-7B), which replicates the reasoning capabilities of [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) and marks the completion of step 1 in the Open R1 project.
* **⚡️ [2025/03/11] [(update #3)](https://huggingface.co/blog/open-r1/update-3):** We release the [**CodeForces-CoTs**](https://huggingface.co/datasets/open-r1/codeforces-cots) dataset of 10k competitive programming problems and 100k solutions distilled from R1. We also release IOI24: a new benchmark of _very_ hard problems from international olympiads. A 7B Qwen model trained on CodeForces-CoTs can outperform Claude 3.7 Sonnet on IOI24, while a 32B model can outperform R1 itself.
* **∞ [2025/02/10] [(update #2)](https://huggingface.co/blog/open-r1/update-2):** We release the [**OpenR1-Math-220k**](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) dataset of 220k traces distilled from R1 on a new version of NuminaMath. Models trained on this dataset match the performance of DeepSeek's distilled ones.
* **🔥 [2025/02/02] [(update #1)](https://huggingface.co/blog/open-r1/update-1):** We implement the first parts of the [training](https://github.com/huggingface/open-r1?tab=readme-ov-file#training-models), [inference](https://github.com/huggingface/open-r1?tab=readme-ov-file#data-generation), and [evaluation](https://github.com/huggingface/open-r1?tab=readme-ov-file#reproducing-deepseeks-evaluation-results) pipelines. Let's go!  

## Installation

> [!CAUTION]
> Libraries rely on CUDA 12.4. If you see errors related to segmentation faults, double check the version your system is running with `nvcc --version`.

To run the code in this project, first, create a Python virtual environment using e.g. `uv`.
To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).


> [!NOTE]
> As a shortcut, run `make install` to setup development libraries (spelled out below). Afterwards, if everything is setup correctly you can try out the Open-R1 models.


```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
```

> [!TIP]
> For Hugging Face cluster users, add `export UV_LINK_MODE=copy` to your `.bashrc` to suppress cache warnings from `uv`

Next, install vLLM and FlashAttention:

```shell
uv pip install vllm==0.8.5.post1
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
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

## Training models

> [!NOTE]
> The training commands below are configured for a node of 8 x H100s (80GB). For different hardware and topologies, you may need to tune the batch size and number of gradient accumulation steps.

We support training models with either DDP or DeepSpeed (ZeRO-2 and ZeRO-3). For example, to perform SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts), run:

```shell
# Train via command line
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path open-r1/Qwen2.5-Math-7B-RoPE-300k \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --eos_token '<|im_end|>' \
    --learning_rate 4.0e-5 \
    --num_train_epochs 5 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/OpenR1-Distill-7B

# Train via YAML config
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml
```

Currently, the following tasks are supported:

* Supervised Fine-Tuning `sft`
* Group Relative Policy Optimization `grpo`

> [!TIP]
> If you scale up/down the number of GPUs, we recommend also scaling up the per-device batch size or number of gradient accumulation steps to keep the global batch size constant.

By default, these scripts will push each model to your Hugging Face Hub username, i.e. `{username}/{model_name}-{task}`. You can override the parameters in each YAML config by appending them to the command as follows: 

```shell
# Change the base model to a smaller variant
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --hub_model_id OpenR1-Distill-0.6B \
    --output_dir data/OpenR1-Distill-0.6B
```

If you also wish to override the Weights and Biases default settings, you can do so as follows:

```shell
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml
    --wandb_entity huggingface --wandb_project open-r1 --run_name Qwen2.5-1.5B-GRPO
```

**🚨 WARNING 🚨**

Most base models like `meta-llama/Llama-3.2-1B` do not have a chat template, so we set ChatML as the default during training. However, for Qwen base models like `Qwen/Qwen2.5-1.5B`, a chat template is pre-defined in the tokenizer, so the EOS token must be set accordingly, e.g.

```diff
# Align EOS token with chat template for Qwen base models
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
+   --eos_token '<|im_end|>'
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --learning_rate 4.0e-5 \
    --num_train_epochs 1 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
```

If you wish to use a custom chat template (e.g. Llama or Gemma), then the chat template and associated EOS token must be provided:

```diff
# Align EOS token with custom chat template
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
+   --chat_template "$(cat llama_chat_template.jinja)" \
+   --eos_token '<|eot_id|>' \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --learning_rate 4.0e-5 \
    --num_train_epochs 1 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/Llama-3.2-1B-Open-R1-Distill
```

### SFT distillation

We provide a recipe to reproduce the reasoning capabilities of [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), starting from the same base model. To do so, run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml
```

The result will be a model like [open-r1/OpenR1-Distill-7B](https://huggingface.co/open-r1/OpenR1-Distill-7B), with the following downstream performance:

| Model                       | AIME 2024 | MATH-500 | GPQA Diamond | LiveCodeBench v5 |
|-----------------------------|-----------|----------|--------------|------------------|
| OpenR1-Distill-7B           | 52.7      | 89.0     | 52.8         | 39.4             |
| DeepSeek-R1-Distill-Qwen-7B | 51.3      | 93.5     | 52.4         | 37.4             |

You can adjust the YAML config to train on a different base model or dataset.

### GRPO

We use TRL's [vLLM backend](https://huggingface.co/docs/trl/speeding_up_training?vllm+examples=GRPO#vllm-for-fast-generation-in-online-methods) to scale training to large models across multiple nodes. For single-node training of smol models across 8 GPUs, use `vllm_mode="colocate"` to run vLLM in the same process as the training script:

```shell
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --vllm_mode colocate
```

> [!WARNING]
> The chat template used in the distilled DeepSeek models omits the contents of the reasoning block within the `<think>` and `</think>` tags. It also prefills the assistant response with `<think>` which interferes with the format reward function. To handle that, it is important to override the chat template as done in e.g.  [recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml](./recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml).

For multi-node training on N+1 nodes, with 1 node running the vLLM server and N nodes running training, we provide an example Slurm script. For example, to run the above example on 1+1 nodes with data parallelism, run:

```shell
sbatch --nodes=2 slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config demo --accelerator zero2 --dp 8 --tp 1
```

See the [Launching jobs on a Slurm cluster](#launching-jobs-on-a-slurm-cluster) section for more details.

#### GRPO dataset filtering

We provide support to filter datasets by generating and computing pass rate on veriable tasks, see this [README](scripts/pass_rate_filtering/README.md)

#### 👨‍💻 Training with a code interpreter

We provide a `code` reward function for executing code generated by the policy during training. Currently, this reward function targets code contests like [Codeforces](https://codeforces.com), where solutions are executed against a set of test cases and the overall success rate is returned as the final reward. To ensure safe execution, we support multiple sandbox providers:

1. [E2B](https://e2b.dev) - Fast, cloud-based sandboxes with focus on Python execution
2. [Morph](https://cloud.morph.so/web/) - Cloud-based sandboxes with broader language support - Python/JS/C++/Rust

To use the code reward function, first install the necessary dependencies:

```shell
uv pip install -e '.[code]'
```

##### E2B Provider

To use E2B sandboxes, create a `.env` file and add your E2B API token:

```
E2B_API_KEY="e2b_xxx"
```

##### Morph Provider

To use Morph, first install the morphcloud package:

```shell
pip install morphcloud
```

Then add your Morph API token to the `.env` file:

```
MORPH_API_KEY="YOUR_MORPH_API_KEY"
```

To specify which provider to use, add the `provider_type` parameter in your configuration:

```yaml
# For E2B
provider_type: e2b

# For Morph
provider_type: morph
```

##### Dataset Requirements

Make sure your dataset contains a `verification_info` column with the following schema (adopted from PrimeIntellect's excellent [datasets](https://huggingface.co/collections/PrimeIntellect/synthetic-1-67a2c399cfdd6c9f7fae0c37) of verifiable problems):

```python
{
    "language": "python",  # Morph supports more languages including C++, Java, etc.
    "test_cases": [
        {
            "input": "4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n",
            "output": "1\n3 \n-1\n0\n\n2\n1 2 \n",
            "type": "stdin_stdout",
        }
    ],
}
```

For example, to train a smol model on Python problems, start the vLLM server:

```shell
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct
```

Then run training with:

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 \
    src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml
```

##### Using Router Services

It is possible to be rate limited when too many scripts are executed on sandbox services. For both providers, we offer router scripts that can be launched on a CPU node:

For E2B:
```shell
sbatch slurm/e2b_router.slurm
```

For Morph:
```shell
sbatch slurm/morph_router.slurm
```

Then add the router URL in your training YAML config:
```yaml
# For E2B
e2b_router_url: 1.2.3.4:8000

# For Morph
morph_router_url: 1.2.3.4:8000
```

The port should match the one used when launching the router.
All training jobs can share the same router IP which will ensure parallel executions are properly managed.

#### Competitive Programming problems: IOI & CodeForces

We provide `ioi_code_reward` and `cf_code_reward` reward functions for executing problems from [IOI](https://hf.co/datasets/open-r1/ioi) and [CodeForces](https://huggingface.co/datasets/open-r1/codeforces), respectively. You can use either [piston](https://github.com/engineer-man/piston) or Morph (currently IOI only) as your execution provider.

##### Piston 

To use Piston:
1. Get piston workers running, see [slurm/piston/README.md](./slurm/piston/README.md)
2. Set your environment variable `PISTON_ENDPOINTS` to `slurm` or to a list of piston worker endpoints

For IOI:

3. In your configuration, use `ioi_provider: "piston"`

For CodeForces:

3. Download the generated (hard) test cases:
```
# change PATH_TO_SAVE_TESTCASES. Increase --max-workers according to your machine's capacity
huggingface-cli download open-r1/codeforces --repo-type=dataset --include='generated_tests/*.parquet' --max-workers=8 --local-dir PATH_TO_SAVE_TESTCASES 
```
4. Save the path in .env:
```
CF_TESTS_FOLDER=PATH_TO_SAVE_TESTCASES
```

##### Morph 

Morph is a cloud-based solution that provides sandboxed environments for running code. To use it:
1. Install the Morph client: `pip install morphcloud`
2. Add your Morph API key to the `.env` file: `MORPH_API_KEY="your_key_here"`
3. In your configuration, use `ioi_provider: "morph"`

##### Example recipes
For IOI:

See the [example recipe](./recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code_ioi.yaml) for how to use the IOI reward function:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code_ioi.yaml
```

For CodeForces:

```shell
sbatch --job-name=cf-grpo --nodes=2 slurm/train.slurm --model Qwen2.5-Coder-7B-Instruct --task grpo --config codeforces --accelerator zero3 --dp 8 --tp 1
```

### Launching jobs on a Slurm cluster

If you have access to a Slurm cluster, we provide a `slurm/train.slurm` script that will automatically queue training jobs for you. Here's how you can use it:

```shell
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm --model {model_name} --task {task} --config {config_suffix} --accelerator {accelerator}
```

Here `{model_name}` and `{task}` are defined as above, while `{config_suffix}` refers to the specific config and `{accelerator}` refers to the choice of 🤗 Accelerate config in `recipes/accelerate_configs`. If you wish to override the default config parameters, you can provide them by appending a space-separated string like `'--arg1=value1 --arg2=value2'`. Here's a concrete example to run SFT on 1 node of 8 GPUs:

```shell
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm --model OpenR1-Distill-7B --task sft --config distill --accelerator zero3
```

You can scale the number of nodes by increasing the `--nodes` flag.

For GRPO, we use 1 node for the vLLM server and N nodes for training. For example, to run GRPO on 1+1 nodes with mixed data and tensor parallelism, run:

```shell
sbatch --job-name=open_r1 --nodes=2 slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config demo --accelerator zero2 --dp 4 --tp 2
```

> [!NOTE]
> The configuration in `slurm/train.slurm` is optimised for the Hugging Face Compute Cluster and may require tweaking to be adapted to your own compute nodes.

### Customising the dataset mixture

To combine multiple datasets as a single training mixture, you can specify the `dataset_mixture` parameter in the YAML config file. Here's a template for how to do this:

```yaml
dataset_mixture:
  datasets:                     # List of datasets to include in the mixture
    - id: dataset_1             # Hub dataset ID
      config: config_name_1     # Name of the dataset config
      split: split_1            # Split to use from the dataset
      columns:                  # Columns to keep
        - column_1              
        - column_2    
      weight: 0.25              # Fraction of dataset to use
    - id: dataset_2
      config: config_name_2
      split: split_2
      columns:                  
        - column_1              
        - column_2   
      weight: 0.5
  seed: 42                      # Seed for shuffling the combined dataset
  test_split_size: 0.1          # Fraction of mixture to use for a test split
```

## Evaluating models

We use `lighteval` to evaluate models. For models which fit on a single GPU, run:

```shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

To increase throughput across multiple GPUs, use _data parallel_ as follows:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

For large models which require sharding across GPUs, use _tensor parallel_ and run:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

You can also launch an evaluation with `make evaluate`, specifying the model, task, and optionally the parallelism technique and number of GPUs.

To evaluate on a single GPU:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

To use Data Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

To use Tensor Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## Reproducing Deepseek's evaluation results

The DeepSeek-R1 paper uses sampling with 4-64 responses per query to estimate `pass@1` accuracy, but does not specify the specific number of responses per benchmark. In the tables below, we estimate `pass@1` accuracy with the following number of responses per query:

|   Benchmark   | Number of responses per query |
|:-------------:|:-----------------------------:|
|   AIME 2024   |              64               |
|   MATH-500    |               4               |
| GPQA Diamond  |               8               |
| LiveCodeBench |              16               |


Note that for benchmarks like AIME24, it is important to sample many responses as there are only 30 problems and this can introduce high variance across repeated runs. The choice of how many responses to sample per prompt likely explains the small differences between our evaluation results and those reported by DeepSeek.

### AIME 2024

We are able to reproduce Deepseek's reported results on the AIME 2024 benchmark within ~1-3 standard deviations:

| Model                         | AIME 2024 (🤗 LightEval) | AIME 2024 (DeepSeek Reported) |
|:------------------------------|:------------------------:|:-----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |           30.7           |             28.9              |
| DeepSeek-R1-Distill-Qwen-7B   |           50.8           |             55.5              |
| DeepSeek-R1-Distill-Qwen-14B  |           65.9           |             69.7              |
| DeepSeek-R1-Distill-Qwen-32B  |           69.7           |             72.6              |
| DeepSeek-R1-Distill-Llama-8B  |           43.9           |             41.7              |
| DeepSeek-R1-Distill-Llama-70B |           63.0           |             70.0              |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks aime24
```

### MATH-500

We are able to reproduce Deepseek's reported results on the MATH-500 benchmark within ~1-3 standard deviations:

| Model                         | MATH-500 (🤗 LightEval) | MATH-500 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          83.1           |             83.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          94.5           |             92.8             |
| DeepSeek-R1-Distill-Qwen-14B  |          94.1           |             93.9             |
| DeepSeek-R1-Distill-Qwen-32B  |          95.6           |             94.3             |
| DeepSeek-R1-Distill-Llama-8B  |          88.6           |             89.1             |
| DeepSeek-R1-Distill-Llama-70B |          95.1           |             94.5             |

To reproduce these results use the following command:

```shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "lighteval|math_500|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks math_500
```

### GPQA Diamond

We are able to reproduce Deepseek's reported results on the GPQA Diamond benchmark within ~1-3 standard deviations:

| Model                         | GPQA Diamond (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:---------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |            35.8             |               33.8               |
| DeepSeek-R1-Distill-Qwen-7B   |            50.5             |               49.1               |
| DeepSeek-R1-Distill-Qwen-14B  |            61.5             |               59.1               |
| DeepSeek-R1-Distill-Qwen-32B  |            63.1             |               62.1               |
| DeepSeek-R1-Distill-Llama-8B  |            46.7             |               49.0               |
| DeepSeek-R1-Distill-Llama-70B |            67.4             |               65.2               |

To reproduce these results use the following command:

```shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "lighteval|gpqa:diamond|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks gpqa
```

### LiveCodeBench

We are able to reproduce Deepseek's reported results on the LiveCodeBench code generation benchmark within ~1-3 standard deviations:

| Model                         | LiveCodeBench (🤗 LightEval) | LiveCodeBench (DeepSeek Reported) |
|:------------------------------|:----------------------------:|:---------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |             16.1             |               16.9                |
| DeepSeek-R1-Distill-Qwen-7B   |             37.4             |               37.6                |
| DeepSeek-R1-Distill-Qwen-14B  |             51.3             |               53.1                |
| DeepSeek-R1-Distill-Qwen-32B  |             56.0             |               57.2                |
| DeepSeek-R1-Distill-Llama-8B  |             37.4             |               39.6                |
| DeepSeek-R1-Distill-Llama-70B |             55.9             |               57.5                |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models, or data_parallel_size=8 with the smaller models for speed
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks lcb
```

## Data generation

### Generate data from a smol distilled R1 model

The following example can be run in 1xH100. 
First install the following dependencies:

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

Now save the following snippet into a file named `pipeline.py` and run it with `python pipeline.py`. It will generate 4 outputs for each of the 10 examples (change the username for the repository to your org/user name):

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

Take a look at the sample dataset at [HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b).


### Generate data from DeepSeek-R1

To run the bigger DeepSeek-R1, we used 2 nodes, each with 8×H100 GPUs using the slurm file present in this repo at `slurm/generate.slurm`. First, install the dependencies:

(for now we need to install the vllm dev wheel that [fixes the R1 cuda graph capture](https://github.com/vllm-project/vllm/commits/221d388cc5a836fa189305785ed7e887cea8b510/csrc/moe/moe_align_sum_kernels.cu))
```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

And then run the following command:

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]  
> While the job is running, you can setup an SSH tunnel through the cluster login node to access the Ray dashboard from your computer running `ssh -L 8265:ray_ip_head_node:8265 <login_node>`, then browsing `http://localhost:8265`


### Data decontamination

Following [s1: Simple test-time scaling](https://huggingface.co/papers/2501.19393) the data can be decontaminated using the script at: [scripts/decontaminate.py](./scripts/decontaminate.py), which decontaminates a dataset using 8-grams and deduplicate the data. Sample run:

```shell
python scripts/decontaminate.py \
    --dataset "open-r1/verifiable-coding-problems-python" \
    --problem_column problem \
    --cleanup
```

It will decontaminate against the benchmark datasets, and remove the contaminated samples afterwards. If no argument `--new_dataset_name` is provided, the same dataset will be reused, adding a `_decontaminated`. It runs against the prompt, which for this dataset is the column `problem`, but a different one can be provided.

Arguments for the script:

```shell
usage: decontaminate.py [-h] --dataset DATASET [--split SPLIT] [--ngram_size NGRAM_SIZE] [--problem_column PROBLEM_COLUMN] [--cleanup] [--new_dataset_name NEW_DATASET_NAME]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of the dataset to check for contamination.
  --split SPLIT         Split to check for contamination, defaults to `train`.
  --ngram_size NGRAM_SIZE
                        Size of n-grams to build, defaults to 8.
  --problem_column PROBLEM_COLUMN
                        Name of the column containing the problem (prompt).
  --cleanup           Whether to remove the contaminated rows before pushing the dataset.
  --new_dataset_name NEW_DATASET_NAME
                        New name for the dataset. If not provided, will reuse the name and add a `_decontaminated` to the name.
```

## Contributing

Contributions are welcome. Please refer to https://github.com/huggingface/open-r1/issues/23.

## Acknowledgements

This project is built with the collective efforts of many groups and individuals in the open AI community. We are especially grateful to the vLLM and SGLang teams for creating high-performance tooling to scale the rollouts of GRPO. We also thank the teams at [OpenThoughts](https://www.open-thoughts.ai), [Prime Intellect](https://www.primeintellect.ai), and [General Reasoning](https://gr.inc) for creating and sharing high-quality datasets for reasoning.

## Citation

If you find this project is useful in your own work, please consider citing as follows:

```
@misc{openr1,
    title = {Open R1: A fully open reproduction of DeepSeek-R1},
    url = {https://github.com/huggingface/open-r1},
    author = {{Hugging Face}},
    month = {January},
    year = {2025}
}
```
