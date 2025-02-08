import csv
import os

import japanize_matplotlib  # 日本語フォント対応（必要に応じて）
import matplotlib.pyplot as plt


def process_csv_data(csv_file_name):
    """
    CSVデータを処理し、各世代において
    シミュレーション100回中に100Mbpsが記録された割合を計算する。

    前提：
      - CSVの各行は1回のシミュレーション実行を表し、
        各列は世代に対応している。
      - 各値は記録された帯域幅（例: 0, 10, 20, …, 100）である。

    各世代について、0～100Mbpsを10刻み（インデックス0～10）にカウントし、
    そのうちインデックス10（100Mbps）の割合を算出する。
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
    xlabel="世代",
    ylabel="最適経路の割合 [%]",
):
    """
    複数のログの結果を同じグラフ上に描画する関数。

    Parameters:
      - x_values: 世代番号のリスト
      - y_values_list: 各ログの「100Mbpsの割合」（各世代ごとの値）のリスト
      - labels: 各ログの凡例ラベル（文字列のリスト）
      - colors: プロットに使用する色のリスト
      - output_file: グラフの保存先ファイル名（SVG形式推奨）
      - xlabel, ylabel: 軸ラベル
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
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_file, format="svg")
    plt.show()


# CSVファイルのパス設定（各条件のログファイル）
# ノード数50と100の場合に対応するよう、ファイル名およびキーを変更
file_paths = {
    "50_3": "./simulation_result/log_ant_50_3_.csv",
    "50_6": "./simulation_result/log_ant_50_6_.csv",
    "100_3": "./simulation_result/log_ant_100_3_.csv",
    "100_6": "./simulation_result/log_ant_100_6_.csv",
}

# 各ログのデータを読み込む
data_results = {}
for key, path in file_paths.items():
    if os.path.exists(path):
        x_vals, ratios = process_csv_data(path)
        data_results[key] = (x_vals, ratios)
    else:
        print(f"ファイルが存在しません: {path}")

# ノード数50の場合のグラフを作成
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
        output_file="./simulation_result/log_ant_analysis_50.svg",
        xlabel="世代",
        ylabel="最適経路の割合 [%]",
    )

# ノード数100の場合のグラフを作成
if "100_3" in data_results and "100_6" in data_results:
    x_vals = data_results["100_3"][0]
    y_vals_100_3 = [gen_ratio[10] for gen_ratio in data_results["100_3"][1]]
    y_vals_100_6 = [gen_ratio[10] for gen_ratio in data_results["100_6"][1]]
    plot_results(
        x_values=x_vals,
        y_values_list=[y_vals_100_3, y_vals_100_6],
        labels=["Nodes = 100, Edges per node = 3", "Nodes = 100, Edges per node = 6"],
        colors=["black", "dimgray"],
        output_file="./simulation_result/log_ant_analysis_100.svg",
        xlabel="世代",
        ylabel="最適経路の割合 [%]",
    )
