import csv
import os

import japanize_matplotlib
import matplotlib.pyplot as plt

ANT_NUM = 10


def process_csv_data(file_path, ant_num):
    data = []
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    data.append([int(val) for val in row])
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return []
    if not data:
        print(f"警告: CSVファイル '{file_path}' が空です。")
        return []
    num_simulations = len(data)
    total_log_entries = len(data[0])
    num_generations = total_log_entries // ant_num
    optimal_percentages = []
    for gen_idx in range(num_generations):
        generation_success_count = 0
        for sim_row in data:
            start_index = gen_idx * ant_num
            end_index = start_index + ant_num
            generation_chunk = sim_row[start_index:end_index]
            if 1 in generation_chunk:
                generation_success_count += 1
        percentage = (generation_success_count / num_simulations) * 100
        optimal_percentages.append(percentage)
    return optimal_percentages


def plot_compare_node_sizes(
    file_small,
    file_large,
    ant_num,
    label_small,
    label_large,
    color_small,
    color_large,
    output_file,
    xlabel,
    ylabel,
):
    y_small = process_csv_data(file_small, ant_num)
    y_large = process_csv_data(file_large, ant_num)
    x_small = list(range(len(y_small)))
    x_large = list(range(len(y_large)))
    plt.figure(figsize=(10, 6))
    plt.plot(
        x_small,
        y_small,
        marker="o",
        linestyle="-",
        color=color_small,
        label=label_small,
    )
    plt.plot(
        x_large,
        y_large,
        marker="o",
        linestyle="-",
        color=color_large,
        label=label_large,
    )
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.legend(fontsize=15, loc="lower right")
    plt.tight_layout()
    # PDF保存に変更
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"グラフを {output_file} にPDF形式で保存しました。")
    plt.show()


# ===== ファイル名の定義（12個） =====
# 格子状ネットワーク
grid_static_49 = "./simulation_result/log_static_grid_49.csv"
grid_static_100 = "./simulation_result/log_static_grid_100.csv"
grid_dynamic_49 = "./simulation_result/log_dynamic_grid_49.csv"
grid_dynamic_100 = "./simulation_result/log_dynamic_grid_100.csv"
# ランダムネットワーク
random_static_50 = "./simulation_result/log_static_er_50.csv"
random_static_100 = "./simulation_result/log_static_er_100.csv"
random_dynamic_50 = "./simulation_result/log_dynamic_er_50.csv"
random_dynamic_100 = "./simulation_result/log_dynamic_er_100.csv"
# スケールフリーネットワーク
ba_static_50 = "./simulation_result/log_static_ba_50.csv"
ba_static_100 = "./simulation_result/log_static_ba_100.csv"
ba_dynamic_50 = "./simulation_result/log_dynamic_ba_50.csv"
ba_dynamic_100 = "./simulation_result/log_dynamic_ba_100.csv"

# ===== 空ファイル作成コマンド（12個） =====
# touch ./simulation_result/log_static_grid_49.csv ./simulation_result/log_static_grid_100.csv ./simulation_result/log_dynamic_grid_49.csv ./simulation_result/log_dynamic_grid_100.csv ./simulation_result/log_static_er_50.csv ./simulation_result/log_static_er_100.csv ./simulation_result/log_dynamic_er_50.csv ./simulation_result/log_dynamic_er_100.csv ./simulation_result/log_static_ba_50.csv ./simulation_result/log_static_ba_100.csv ./simulation_result/log_dynamic_ba_50.csv ./simulation_result/log_dynamic_ba_100.csv

# ===== グラフ出力（6個） =====
# グリッド: 固定・動的
plot_compare_node_sizes(
    grid_static_49,
    grid_static_100,
    ANT_NUM,
    "Grid: Nodes=49",
    "Grid: Nodes=100",
    "black",
    "dimgray",
    "./simulation_result/static_grid.pdf",
    "Generation",
    "Optimal Path Selection Ratio [%]",
)
plot_compare_node_sizes(
    grid_dynamic_49,
    grid_dynamic_100,
    ANT_NUM,
    "Grid: Nodes=49",
    "Grid: Nodes=100",
    "black",
    "dimgray",
    "./simulation_result/dynamic_grid.pdf",
    "Generation",
    "Optimal Path Selection Ratio [%]",
)
# ランダム: 固定・動的
plot_compare_node_sizes(
    random_static_50,
    random_static_100,
    ANT_NUM,
    "ER: Nodes=50",
    "ER: Nodes=100",
    "black",
    "dimgray",
    "./simulation_result/static_er.pdf",
    "Generation",
    "Optimal Path Selection Ratio [%]",
)
plot_compare_node_sizes(
    random_dynamic_50,
    random_dynamic_100,
    ANT_NUM,
    "ER: Nodes=50",
    "ER: Nodes=100",
    "black",
    "dimgray",
    "./simulation_result/dynamic_er.pdf",
    "Generation",
    "Optimal Path Selection Ratio [%]",
)
# BA: 固定・動的
plot_compare_node_sizes(
    ba_static_50,
    ba_static_100,
    ANT_NUM,
    "BA: Nodes=50",
    "BA: Nodes=100",
    "black",
    "dimgray",
    "./simulation_result/static_ba.pdf",
    "Generation",
    "Optimal Path Selection Ratio [%]",
)
plot_compare_node_sizes(
    ba_dynamic_50,
    ba_dynamic_100,
    ANT_NUM,
    "BA: Nodes=50",
    "BA: Nodes=100",
    "black",
    "dimgray",
    "./simulation_result/dynamic_ba.pdf",
    "Generation",
    "Optimal Path Selection Ratio [%]",
)
