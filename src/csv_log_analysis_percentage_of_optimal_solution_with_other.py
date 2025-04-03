import csv
import os

import japanize_matplotlib  # 日本語フォント対応（必要に応じて）
import matplotlib.pyplot as plt


def process_csv_data(csv_file_name):
    """
    Process CSV data and calculate the percentage of simulation runs
    in which 100 Mbps was recorded in each generation.

    Assumptions:
      - Each row represents a single simulation run.
      - Each column corresponds to a generation.
      - Each value represents a recorded bandwidth (e.g., 0, 10, 20, ..., 100).

    This function counts occurrences of different bandwidths (0-100 Mbps in 10-unit steps)
    in each generation and calculates the percentage of occurrences of 100 Mbps.
    """
    with open(csv_file_name, "r") as f:
        reader = csv.reader(f)
        data = [list(map(float, row)) for row in reader]

    # 各列（世代）の数を取得（シミュレーション回数）
    num_generations = len(data[0])
    # 各世代ごとに、0～100Mbps（10刻み）をカウントするためのリストを用意
    counts = [[0] * 11 for _ in range(num_generations)]

    # 各シミュレーション実行（行）について、各世代（列）の値をカウント
    for row in data:
        for gen_index, width in enumerate(row):
            bin_index = int(width) // 10  # 例えば100なら 100//10=10
            counts[gen_index][bin_index] += 1

    # 各世代の総実行回数（通常は100回）を計算
    totals = [sum(counts[gen_index]) for gen_index in range(num_generations)]
    # 各世代で、各帯域幅区間の割合（％）を算出
    ratios = []
    for gen_index in range(num_generations):
        total = totals[gen_index]
        if total > 0:
            gen_ratios = [count * 100 / total for count in counts[gen_index]]
        else:
            gen_ratios = [0] * 11
        ratios.append(gen_ratios)

    # x_values：世代番号のリスト（例：0,1,2,...）
    x_values = list(range(num_generations))
    return x_values, ratios


def plot_results(
    x_values,
    y_values_list,
    labels,
    colors,
    output_file,
    xlabel="Generation",
    ylabel="Optimal Path Selection Ratio [%]",
):
    """
    Plot multiple log results on the same graph.

    Parameters:
      - x_values: List of generations
      - y_values_list: List of datasets representing the ratio of 100 Mbps occurrences per generation
      - labels: List of legend labels
      - colors: List of colors for the plots
      - output_file: File name for saving the graph (eps format recommended)
      - xlabel, ylabel: Labels for axes
    """
    plt.figure(figsize=(10, 6))
    for y_values, label, color in zip(y_values_list, labels, colors):
        plt.scatter(x_values, y_values, color=color, s=10, label=label)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim(0, 100)
    plt.legend(fontsize=15, loc="lower right")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(output_file, format="eps")
    plt.show()


# Set file paths for different conditions
file_paths = {
    "50_3": "./simulation_result/log_ant_50_3_random_optional_10-90-base.csv",
    "50_6": "./simulation_result/log_ant_50_6_random_optional_10-90-base.csv",
    "100_3": "./simulation_result/log_ant_100_3_random_optional_10-90-base.csv",
    "100_6": "./simulation_result/log_ant_100_6_random_optional_10-90-base.csv",
    "150_3": "./simulation_result/log_ant_150_3_random_optional_10-90-base.csv",
    "150_6": "./simulation_result/log_ant_150_6_random_optional_10-90-base.csv",
}

# Read data from files
data_results = {}
for key, path in file_paths.items():
    if os.path.exists(path):
        x_vals, ratios = process_csv_data(path)
        data_results[key] = (x_vals, ratios)
    else:
        print(f"File not found: {path}")

# Generate plot for 50 nodes
if "50_3" in data_results and "50_6" in data_results:
    x_vals = data_results["50_3"][0]  # 世代番号は全て同じと仮定
    # 各世代での100Mbpsの割合は、各 ratios のインデックス10
    y_vals_50_3 = [gen_ratio[10] for gen_ratio in data_results["50_3"][1]]
    y_vals_50_6 = [gen_ratio[10] for gen_ratio in data_results["50_6"][1]]
    plot_results(
        x_values=x_vals,
        y_values_list=[y_vals_50_3, y_vals_50_6],
        labels=["Nodes = 50, Edges per node = 3", "Nodes = 50, Edges per node = 6"],
        colors=["black", "dimgray"],
        output_file="./simulation_result/log_ant_analysis_50.eps",
        xlabel="Generation",
        ylabel="Optimal Path Selection Ratio [%]",
    )

# Generate plot for 100 nodes
if "100_3" in data_results and "100_6" in data_results:
    x_vals = data_results["100_3"][0]
    y_vals_100_3 = [gen_ratio[10] for gen_ratio in data_results["100_3"][1]]
    y_vals_100_6 = [gen_ratio[10] for gen_ratio in data_results["100_6"][1]]
    plot_results(
        x_values=x_vals,
        y_values_list=[y_vals_100_3, y_vals_100_6],
        labels=["Nodes = 100, Edges per node = 3", "Nodes = 100, Edges per node = 6"],
        colors=["black", "dimgray"],
        output_file="./simulation_result/log_ant_analysis_100.eps",
        xlabel="Generation",
        ylabel="Optimal Path Selection Ratio [%]",
    )

# Generate plot for 150 nodes
if "150_3" in data_results and "150_6" in data_results:
    x_vals = data_results["150_3"][0]
    y_vals_150_3 = [gen_ratio[10] for gen_ratio in data_results["150_3"][1]]
    y_vals_150_6 = [gen_ratio[10] for gen_ratio in data_results["150_6"][1]]
    plot_results(
        x_values=x_vals,
        y_values_list=[y_vals_150_3, y_vals_150_6],
        labels=["Nodes = 150, Edges per node = 3", "Nodes = 150, Edges per node = 6"],
        colors=["black", "dimgray"],
        output_file="./simulation_result/log_ant_analysis_150.eps",
        xlabel="Generation",
        ylabel="Optimal Path Selection Ratio [%]",
    )
