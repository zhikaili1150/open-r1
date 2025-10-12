import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
# from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available

import os
os.environ["WANDB_PROJECT"]="VAL"

if is_accelerate_available():
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def main():
    evaluation_tracker = EvaluationTracker(
        output_dir="./my_results/test",
        save_details=True,
        push_to_hub=False,
        hub_results_org="Zachary1150",
        wandb=True,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="src/open_r1/evaluate.py", # if using a custom task
        # Remove the 2 parameters below once your configuration is tested
        # override_batch_size=1,
        # max_samples=10
    )

    model_config = VLLMModelConfig(
            model_name="deepseek-ai/Deepseek-R1-Distill-Qwen-1.5B",
            dtype="bfloat16",
            use_chat_template=True,
            max_model_length=32768,
            gpu_memory_utilization=0.8,
            generation_parameters={
                "max_new_tokens":32768,
                "temperature":0.6,
                "top_p":0.95}
    )

    task = "custom|math_validation|0|0"

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