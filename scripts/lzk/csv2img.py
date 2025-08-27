import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

'''
Usage:
python plot_accuracy.py --csv_path results/xxx/averaged_results.csv
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Plot model accuracy from averaged_results.csv")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to averaged_results.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    csv_path = args.csv_path
    df = pd.read_csv(csv_path)

    if "Model" not in df.columns or "Accuracy" not in df.columns:
        raise ValueError("CSV must contain 'Model' and 'Accuracy' columns.")

    models = df["Model"].tolist()
    accuracies = df["Accuracy"].tolist()

    plt.figure(figsize=(len(models) * 2, 10))
    bars = plt.bar(models, accuracies, color='skyblue')

    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(min(accuracies) * 0.98, max(accuracies) * 1.02)
    plt.grid(axis='y')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}",
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = f"{Path(csv_path).parent}/comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ 图像已保存到 {output_path}")

if __name__ == "__main__":
    main()
