import sys
import os
import shutil
from safetensors.torch import load_file, save_file
import torch

def main():
    # 至少需要 3 个参数：<lora_dir1> <weight1> ... <output_dir>
    if len(sys.argv) < 5 or (len(sys.argv) - 2) % 2 != 0:
        print("❌ 用法: python merge_safetensors.py <lora_dir1> <weight1> <lora_dir2> <weight2> ... <output_dir>")
        sys.exit(1)

    args = sys.argv[1:]
    output_dir = args[-1]
    pair_args = args[:-1]

    # 拆分路径和权重
    lora_dirs = pair_args[::2]
    weights = list(map(float, pair_args[1::2]))

    # 过滤掉权重为 0 的 lora
    filtered = [(d, w) for d, w in zip(lora_dirs, weights) if w > 0]
    if not filtered:
        print("❌ 所有权重均为 0，无法合并")
        sys.exit(1)

    # 拆分过滤后的路径和权重
    lora_dirs, weights = zip(*filtered)
    lora_dirs, weights = list(lora_dirs), list(weights)

    assert len(lora_dirs) == len(weights), "目录数与权重数不匹配"
    assert len(lora_dirs) >= 1, "请至少提供一个有效的 LoRA 模型"

    print(f"📦 开始合并 {len(lora_dirs)} 个 LoRA 模型 (已自动忽略 weight=0 的模型)")
    print("🔢 权重列表:", weights)

    # 加载每个 LoRA 的 adapter_model.safetensors
    safetensor_paths = []
    for d in lora_dirs:
        model_path = os.path.join(d, "adapter_model.safetensors")
        assert os.path.exists(model_path), f"❌ 找不到: {model_path}"
        safetensor_paths.append(model_path)

    # 加载所有 state_dict
    all_state_dicts = [load_file(path) for path in safetensor_paths]
    param_keys = all_state_dicts[0].keys()

    # 归一化权重
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    # 加权合并
    avg_state_dict = {}
    for key in param_keys:
        weighted_sum = sum(w * sd[key].float() for w, sd in zip(norm_weights, all_state_dicts))
        avg_state_dict[key] = weighted_sum

    # 保存合并模型
    os.makedirs(output_dir, exist_ok=True)
    merged_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(avg_state_dict, merged_path)
    print(f"✅ 合并模型已保存到: {merged_path}")

    # 拷贝辅助文件（从第一个目录）
    files_to_copy = [
        "adapter_config.json",
        "chat_template.jinja",
        "config.json", 
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    for fname in files_to_copy:
        src = os.path.join(lora_dirs[0], fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"📁 拷贝辅助文件: {fname}")

    print(f"🎉 所有内容已保存在: {output_dir}")


if __name__ == "__main__":
    main()
