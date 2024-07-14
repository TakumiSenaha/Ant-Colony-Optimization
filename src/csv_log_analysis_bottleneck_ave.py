import csv

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
plt.plot(interest_avg_bandwidth_per_generation, label="Interest")
plt.plot(rand_avg_bandwidth_per_generation, label="Rand")
# plt.plot(ant_avg_bandwidth_per_generation, label="Ant (Pheromone)")
plt.xlabel("Generation")
plt.ylabel("Average Bottleneck Bandwidth")
plt.title("Average Bottleneck Bandwidth per Generation")
plt.legend()
plt.grid(True)
plt.show()
plt.show()
plt.show()
