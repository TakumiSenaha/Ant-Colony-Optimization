import csv
import os

import japanize_matplotlib
import matplotlib.pyplot as plt

# ===== 解析設定 =====
ANT_NUM = 10  # シミュレーションで設定したアリの数
# ===================


def process_csv_data(file_path, ant_num):
    """
    CSVデータを読み込み、世代ごとの最適解発見率を計算する。
    ant_numの値に応じて処理を切り替える。
    """
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
    optimal_percentages = []

    if ant_num == 1:
        if not data[0]:
            return []
        num_generations = len(data[0])
        for gen_idx in range(num_generations):
            count_optimal = sum(row[gen_idx] == 1 for row in data)
            percentage = (count_optimal / num_simulations) * 100
            optimal_percentages.append(percentage)
    else:
        if not data[0]:
            return []
        total_log_entries = len(data[0])
        num_generations = total_log_entries // ant_num
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


def plot_multiple_results(
    x_values, y_values_list, labels, colors, output_file, xlabel, ylabel
):
    plt.figure(figsize=(10, 6))
    for y_values, label, color in zip(y_values_list, labels, colors):
        plt.plot(
            x_values, y_values, marker="o", linestyle="-", color=color, label=label
        )
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.legend(fontsize=15, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    ax = plt.gca()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.5)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_frame_on(True)
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(output_file, format="svg")
    print(f"グラフを {output_file} に保存しました。")
    plt.show()


def plot_compare_node_sizes(
    file_50,
    file_100,
    ant_num,
    label_50,
    label_100,
    color_50,
    color_100,
    output_file,
    xlabel,
    ylabel,
):
    y_50 = process_csv_data(file_50, ant_num)
    y_100 = process_csv_data(file_100, ant_num)
    x_50 = list(range(len(y_50)))
    x_100 = list(range(len(y_100)))
    plt.figure(figsize=(10, 6))
    plt.plot(x_50, y_50, marker="o", linestyle="-", color=color_50, label=label_50)
    plt.plot(x_100, y_100, marker="o", linestyle="-", color=color_100, label=label_100)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.legend(fontsize=15, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    ax = plt.gca()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.5)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_frame_on(True)
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(output_file, format="svg")
    print(f"グラフを {output_file} に保存しました。")
    plt.show()


def plot_two_csvs(
    file_1,
    file_2,
    ant_num,
    label_1,
    label_2,
    color_1,
    color_2,
    output_file_1,
    output_file_2,
    xlabel,
    ylabel,
):
    y_1 = process_csv_data(file_1, ant_num)
    x_1 = list(range(len(y_1)))
    plt.figure(figsize=(10, 6))
    plt.plot(x_1, y_1, marker="o", linestyle="-", color=color_1, label=label_1)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.legend(fontsize=15, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file_1, format="svg")
    print(f"グラフを {output_file_1} に保存しました。")
    plt.show()

    y_2 = process_csv_data(file_2, ant_num)
    x_2 = list(range(len(y_2)))
    plt.figure(figsize=(10, 6))
    plt.plot(x_2, y_2, marker="o", linestyle="-", color=color_2, label=label_2)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.legend(fontsize=15, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file_2, format="svg")
    print(f"グラフを {output_file_2} に保存しました。")
    plt.show()


# ===== ファイル名・凡例・色の設定例 =====
# 静的環境
static_files = {
    "50_3": "./simulation_result/log_static_50_3.csv",
    "50_6": "./simulation_result/log_static_50_6.csv",
    "100_3": "./simulation_result/log_static_100_3.csv",
    "100_6": "./simulation_result/log_static_100_6.csv",
}
# 動的環境
dynamic_files = {
    "50_3": "./simulation_result/log_dynamic_50_3.csv",
    "50_6": "./simulation_result/log_dynamic_50_6.csv",
    "100_3": "./simulation_result/log_dynamic_100_3.csv",
    "100_6": "./simulation_result/log_dynamic_100_6.csv",
}

# グラフ出力設定
output_settings = [
    # グラフ1: 静的環境・ノード数50
    {
        "files": [static_files["50_3"], static_files["50_6"]],
        "labels": ["Static: Nodes=50, Edges=3", "Static: Nodes=50, Edges=6"],
        "colors": ["black", "dimgray"],
        "output_file": "./simulation_result/result_static_50.svg",
    },
    # グラフ2: 静的環境・ノード数100
    {
        "files": [static_files["100_3"], static_files["100_6"]],
        "labels": ["Static: Nodes=100, Edges=3", "Static: Nodes=100, Edges=6"],
        "colors": ["black", "dimgray"],
        "output_file": "./simulation_result/result_static_100.svg",
    },
    # グラフ3: 動的環境・ノード数50
    {
        "files": [dynamic_files["50_3"], dynamic_files["50_6"]],
        "labels": ["Dynamic: Nodes=50, Edges=3", "Dynamic: Nodes=50, Edges=6"],
        "colors": ["black", "dimgray"],
        "output_file": "./simulation_result/result_dynamic_50.svg",
    },
    # グラフ4: 動的環境・ノード数100
    {
        "files": [dynamic_files["100_3"], dynamic_files["100_6"]],
        "labels": ["Dynamic: Nodes=100, Edges=3", "Dynamic: Nodes=100, Edges=6"],
        "colors": ["black", "dimgray"],
        "output_file": "./simulation_result/result_dynamic_100.svg",
    },
]

# ===== グラフ生成処理 =====
for setting in output_settings:
    y_values_list = []
    x_values = None
    for file_path in setting["files"]:
        if os.path.exists(file_path):
            y_values = process_csv_data(file_path, ANT_NUM)
            y_values_list.append(y_values)
            if x_values is None:
                x_values = list(range(len(y_values)))
        else:
            print(f"File not found: {file_path}")
    if y_values_list and x_values:
        plot_multiple_results(
            x_values=x_values,
            y_values_list=y_values_list,
            labels=setting["labels"],
            colors=setting["colors"],
            output_file=setting["output_file"],
            xlabel="Generation",
            ylabel="Optimal Path Selection Ratio [%]",
        )

# ===== 使用例 =====
plot_two_csvs(
    file_1="./simulation_result/log_static_ba_50.csv",
    file_2="./simulation_result/log_static_ba_100.csv",
    ant_num=ANT_NUM,
    label_1="BA: Nodes=50",
    label_2="BA: Nodes=100",
    color_1="black",
    color_2="dimgray",
    output_file_1="./simulation_result/result_static_ba_50.svg",
    output_file_2="./simulation_result/result_static_ba_100.svg",
    xlabel="Generation",
    ylabel="Optimal Path Selection Ratio [%]",
)
# ERモデル
plot_compare_node_sizes(
    file_50="./simulation_result/log_static_er_50.csv",
    file_100="./simulation_result/log_static_er_100.csv",
    ant_num=ANT_NUM,
    label_50="ER: Nodes=50",
    label_100="ER: Nodes=100",
    color_50="blue",
    color_100="deepskyblue",
    output_file="./simulation_result/result_static_er_compare.svg",
    xlabel="Generation",
    ylabel="Optimal Path Selection Ratio [%]",
)
# グリッドモデル
plot_compare_node_sizes(
    file_50="./simulation_result/log_static_grid_49.csv",
    file_100="./simulation_result/log_static_grid_100.csv",
    ant_num=ANT_NUM,
    label_50="Grid: Nodes=49",
    label_100="Grid: Nodes=100",
    color_50="green",
    color_100="lime",
    output_file="./simulation_result/result_static_grid_compare.svg",
    xlabel="Generation",
    ylabel="Optimal Path Selection Ratio [%]",
)
