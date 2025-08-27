import os
import json
import csv
import argparse
from pathlib import Path

import pandas as pd

'''
python result2csv.py \
  --result_dir experiments/exp9_ds1b_reward_ratio/results \
  --task gsm8k|8 \
  --metric extractive_match \
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Parse evaluation results and visualize accuracy.")
    parser.add_argument("--result_dir", type=str, required=True, help="Input directory with result .json files")
    parser.add_argument("--task", type=str, default="gsm8k|0", help="Evaluation task name")
    parser.add_argument("--metric", type=str, default="extractive_match", help="Metric to extract from result files")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save the output CSV file")
    return parser.parse_args()

def main():
    args = parse_args()

    result_dir = args.result_dir
    task = args.task
    metric = args.metric
    output_csv = args.output_csv or f"{result_dir}/result.csv"
    
    rows = []

    for root, _, files in os.walk(result_dir):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    acc = data["results"][f"lighteval|{task}"][f"{metric}"]
                    relative_path = os.path.relpath(filepath, result_dir)
                    rows.append((relative_path, acc))

                except Exception as e:
                    print(f"❌ 跳过 {filepath}: {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy"])
        writer.writerows(rows)

    print(f"✅ 已写入 {output_csv}，共 {len(rows)} 条记录")

    # 平均计算
    df = pd.read_csv(output_csv)
    df["Model"] = df["Model"].apply(lambda x: str(Path(x).parent))
    
    if "Method" in df.columns:
        df["Method"] = df["Method"].str.strip()
        df_avg = df.groupby(["Method", "Model"], as_index=False)["Accuracy"].mean()
    else:
        df_avg = df.groupby(["Model"], as_index=False)["Accuracy"].mean()

    averaged_results_csv = f"{Path(output_csv).parent}/averaged_results.csv"
    df_avg.to_csv(averaged_results_csv, index=False)
    print(f"✅ 已保存平均结果到 {averaged_results_csv}")


if __name__ == "__main__":
    main()
