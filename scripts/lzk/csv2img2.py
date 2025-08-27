import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

'''
Usage:
python plot_accuracy.py --csv_path results/xxx/averaged_results.csv
'''

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, required=True)
args = parser.parse_args()

averaged_results_csv = args.csv_path

def main():

    # 读取 CSV
    df = pd.read_csv(averaged_results_csv)

    # 确保数据类型正确
    df["Model"] = df["Model"].astype(str)

    # 按 Method 分类
    methods = df["Method"].unique()

    # 设置颜色和样式（可选）
    colors = {
        "Mix_Reward": "blue",
        "Merge_Policy": "orange"
    }

    # 开始画图
    plt.figure(figsize=(10, 8))

    # for method in methods:
    #     subset = df[df["Method"] == method]
        
    #     # 按 Model 排序以避免折线乱跳
    #     subset = subset.sort_values("Model")
        
    #     plt.plot(
    #         subset["Model"],
    #         subset["Accuracy"],
    #         label=method,
    #         marker="o",
    #         color=colors.get(method, None)
    #     )
    for method in methods:
        subset = df[df["Method"] == method].copy()  # 加 .copy() 以避免 SettingWithCopyWarning

        # 将 Model 转换成数字和
        subset["ScoreSum"] = subset["Model"].apply(lambda x: sum(int(d) for d in str(x)))

        # 按 ScoreSum 再按 Model 排序，避免线乱跳
        subset = subset.sort_values(by=["ScoreSum", "Model"])

        plt.plot(
            subset["Model"],
            subset["Accuracy"],
            label=method,
            marker="o",
            color=colors.get(method, None)
        )

    # 美化图像
    plt.xlabel("Reward Ratio - Accuracy:Format:Length")
    plt.ylabel("Accuracy")
    plt.title("Method Comparison on GSM8K|0shot")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    input_dir = Path(averaged_results_csv).parent
    # 保存或展示
    plt.savefig(f"{input_dir}/comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()