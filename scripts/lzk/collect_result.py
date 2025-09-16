import os
import json
import pandas as pd
import re
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="汇总任务结果 JSON -> CSV")
    parser.add_argument("--root_dir", required=True,
                        help="起始搜索路径，例如 /path/to/experiments")
    parser.add_argument("--tasks", nargs="+",
                        default=["aime24", "math_500", "amc23", "minerva", "olympiadbench"],
                        help="要处理的任务列表 (默认: aime24 math_500 amc23 minerva olympiadbench)")
    parser.add_argument("--metrics", nargs="+",
                        default=["extractive_match"],
                        help="要提取的指标 key 列表 (默认: extractive_match)")
    parser.add_argument("--output", default="results_summary.csv",
                        help="输出 CSV 文件名 (默认: results_summary.csv)")
    args = parser.parse_args()

    records = {}

    for task in args.tasks:
        task_dir = os.path.join(args.root_dir, task, "results")
        if not os.path.isdir(task_dir):
            continue

        for model_dir in os.listdir(task_dir):
            model_path = os.path.join(task_dir, model_dir)
            if not os.path.isdir(model_path):
                continue

            for fname in os.listdir(model_path):
                if not fname.endswith(".json"):
                    continue

                fpath = os.path.join(model_path, fname)
                with open(fpath, "r") as f:
                    data = json.load(f)

                for metric in args.metrics:
                    try:
                        value = data["results"][f"custom|{task}|0"][metric]
                    except KeyError:
                        print(f"⚠️ 跳过 {fpath}, 没有 {task}/{metric}")
                        continue

                    # 用正则提取 model 名称，例如 merge_policy_v2_201
                    m = re.search(r"(merge_policy_v\d+_\d+)", model_dir)
                    if m:
                        model = m.group(1)
                    else:
                        model = model_dir  # fallback

                    if model not in records:
                        records[model] = {}
                    if f"{task}_{metric}" not in records[model]:
                        records[model][f"{task}_{metric}"] = []
                    records[model][f"{task}_{metric}"].append(value)

    # === 把 list 转换成平均值 ===
    for model, vals in records.items():
        for key in vals:
            if isinstance(vals[key], list):
                records[model][key] = float(np.mean(vals[key]))

    # 转换为 DataFrame
    rows = []
    for model, vals in records.items():
        row = {"model": model}
        for task in args.tasks:
            for metric in args.metrics:
                row[f"{task}_{metric}"] = vals.get(f"{task}_{metric}", None)
        rows.append(row)

    df = pd.DataFrame(rows)

    # === 每个 metric 分别算平均值 ===
    for metric in args.metrics:
        metric_cols = [f"{task}_{metric}" for task in args.tasks if f"{task}_{metric}" in df.columns]
        if metric_cols:
            df[f"avg_{metric}"] = df[metric_cols].mean(axis=1, skipna=True)

    # === 对 extractive_match 及其平均值 ×100 并保留 1 位小数 ===
    for col in df.columns:
        if "extractive_match" in col:
            df[col] = df[col].apply(lambda x: round(x * 100, 1) if pd.notnull(x) else x)

    # 保存
    csv_path = os.path.join(args.root_dir, args.output)
    df.to_csv(csv_path, index=False)

    print(f"✅ 已保存 {csv_path}")

if __name__ == "__main__":
    main()
