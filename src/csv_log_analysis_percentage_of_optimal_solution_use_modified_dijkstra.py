import csv
import os

import japanize_matplotlib
import matplotlib.pyplot as plt

# CSVファイル名
csv_file_name = "./simulation_result/log_ant.csv"
export_image_name = "./simulation_result/result_optimal_percentage.svg"


def process_csv_data(csv_file_name):
    with open(csv_file_name, "r") as f:
        reader = csv.reader(f)
        data = [list(map(int, row)) for row in reader]

    # 各世代ごとに最適解(1)の割合を計算
    num_generations = len(data[0])
    num_trials = len(data)
    optimal_percentages = []
    for gen in range(num_generations):
        count_optimal = sum(row[gen] == 1 for row in data)
        percentage = (count_optimal / num_trials) * 100
        optimal_percentages.append(percentage)
    return optimal_percentages


# データ読み込みと処理
optimal_percentages = process_csv_data(csv_file_name)
x_values = list(range(len(optimal_percentages)))
y_values = optimal_percentages

# グラフ描画
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color="black", s=5, label="最適解割合")

plt.ylim((0, 100))
plt.xlabel("世代", fontsize=20)
plt.ylabel("最適解割合 [%]", fontsize=20)
plt.legend(fontsize=15, loc="lower right")

# 軸と枠線の強調
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
plt.savefig(export_image_name, format="svg")
plt.show()
