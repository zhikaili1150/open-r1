import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

if len(sys.argv) != 4:
    print("Usage: python merge_lora_model.py <base_model_path> <lora_path> <merged_output_path>")
    sys.exit(1)

base_model_path = sys.argv[1]
lora_path = sys.argv[2]
save_path = sys.argv[3]

print(f"Base model: {base_model_path}")
print(f"LoRA path:  {lora_path}")
print(f"Save path:  {save_path}")

# 1. 加载 base model（全量参数）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 2. 加载 LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_path)

# 3. 合并 LoRA 和 base model（并释放 LoRA adapter 结构）
model = model.merge_and_unload()

# 4. 保存合并后的模型
model.save_pretrained(save_path, safe_serialization=True)

# 5. 保存 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

print(f"✅ Merged model saved to: {save_path}")
