import sys
import os
import shutil
from safetensors.torch import load_file, save_file
import torch


def main():
    if len(sys.argv) < 5 or (len(sys.argv) - 2) % 2 != 0:
        print(
            "❌ 用法: python merge_safetensors.py <dir1> <weight1> <dir2> <weight2> ... <output_dir>"
        )
        sys.exit(1)

    args = sys.argv[1:]
    output_dir = args[-1]
    pair_args = args[:-1]

    model_dirs = pair_args[::2]
    weights = list(map(float, pair_args[1::2]))

    filtered = [(d, w) for d, w in zip(model_dirs, weights) if w > 0]
    if not filtered:
        print("❌ 所有权重均为 0，无法合并")
        sys.exit(1)

    model_dirs, weights = zip(*filtered)
    model_dirs, weights = list(model_dirs), list(weights)

    print(f"📦 开始合并 {len(model_dirs)} 个模型")
    print("🔢 权重列表:", weights)

    safetensor_paths = []
    for d in model_dirs:
        path = None
        if os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            path = os.path.join(d, "adapter_model.safetensors")
        elif os.path.exists(os.path.join(d, "model.safetensors")):
            path = os.path.join(d, "model.safetensors")
        else:
            raise FileNotFoundError(
                f"❌ 在 {d} 找不到 adapter_model.safetensors 或 model.safetensors"
            )
        safetensor_paths.append(path)

    all_state_dicts = [load_file(p) for p in safetensor_paths]
    param_keys = all_state_dicts[0].keys()

    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    avg_state_dict = {}
    for key in param_keys:
        # 保存原始 dtype
        dtype = all_state_dicts[0][key].dtype
        # 加权计算使用 float32，提高精度
        weighted_sum = sum(
            w * sd[key].to(torch.float32)
            for w, sd in zip(norm_weights, all_state_dicts)
        )
        # 转回原 dtype
        avg_state_dict[key] = weighted_sum.to(dtype)

    os.makedirs(output_dir, exist_ok=True)

    files_to_copy = [
        "chat_template.jinja",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]

    # 根据第一个模型的类型决定输出文件名
    if "adapter_model.safetensors" in safetensor_paths[0]:
        merged_filename = "adapter_model.safetensors"
        files_to_copy.append("adapter_config.json")
    else:
        merged_filename = "model.safetensors"
        files_to_copy.append("generation_config.json")

    merged_path = os.path.join(output_dir, merged_filename)
    save_file(avg_state_dict, merged_path)
    print(f"✅ 合并模型已保存到: {merged_path}")

    for fname in files_to_copy:
        src = os.path.join(model_dirs[0], fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"📁 拷贝辅助文件: {fname}")

    print(f"🎉 所有内容已保存在: {output_dir}")


if __name__ == "__main__":
    main()
