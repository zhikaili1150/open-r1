import argparse
import os
from pathlib import Path
from datetime import timedelta

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))]
    )
else:
    accelerator = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with lighteval")
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name to evaluate, e.g. 'custom|math_validation|0'",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    wandb_project = "_".join(Path(args.result_dir).parts[-2:])
    os.environ["WANDB_PROJECT"] = wandb_project
    wandb_project = "_".join(Path(args.model_name).parts[-2:])
    os.environ["WANDB_NAME"] = wandb_project

    evaluation_tracker = EvaluationTracker(
        output_dir=args.result_dir,
        save_details=True,
        push_to_hub=False,
        hub_results_org="Zachary1150",
        wandb=True,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="src/open_r1/evaluate.py",
    )

    model_config = VLLMModelConfig(
        model_name=args.model_name,
        dtype="bfloat16",
        use_chat_template=True,
        max_model_length=32768,
        gpu_memory_utilization=0.8,
        generation_parameters={
            "max_new_tokens": 32768,
            "temperature": 0.6,
            "top_p": 0.95,
        },
    )

    task = args.task

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    main()
