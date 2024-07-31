import csv

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np


# CSVファイルを読み込む
def read_log(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        data = [list(map(int, row)) for row in reader]
    return data


# 各世代のボトルネック帯域の平均値を計算
def calculate_average_bandwidth_per_generation(log_data):
    # 各行の長さをチェックし、最も短い行の長さを世代数とする
    generations = min(len(row) for row in log_data)
    avg_bandwidth = []

    for gen in range(generations):
        gen_values = [row[gen] for row in log_data]
        avg_bandwidth.append(np.mean(gen_values))

    return avg_bandwidth


# ログデータを読み込む
interest_log_data = read_log("./simulation_result/log_interest.csv")
rand_log_data = read_log("./simulation_result/log_rand.csv")
# ant_log_data = read_log("./simulation_result/log_ant.csv")

# 平均値を計算
interest_avg_bandwidth_per_generation = calculate_average_bandwidth_per_generation(
    interest_log_data
)
rand_avg_bandwidth_per_generation = calculate_average_bandwidth_per_generation(
    rand_log_data
)
# ant_avg_bandwidth_per_generation = calculate_average_bandwidth_per_generation(
#     ant_log_data
# )

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.scatter(
    range(len(interest_avg_bandwidth_per_generation)),
    interest_avg_bandwidth_per_generation,
    color="blue",
    s=5,
)
plt.scatter(
    range(len(rand_avg_bandwidth_per_generation)),
    rand_avg_bandwidth_per_generation,
    color="orange",
    s=5,
)
# plt.scatter(range(len(ant_avg_bandwidth_per_generation)), ant_avg_bandwidth_per_generation, color="green", s=5)

plt.xlabel("世代", fontsize=20)
plt.ylabel("平均ボトルネック帯域", fontsize=20)

# 縦軸と横軸の囲いを表示
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)

# フォントサイズを2倍に設定
ax.tick_params(axis="both", which="major", labelsize=20)

plt.grid(True)
plt.show()
