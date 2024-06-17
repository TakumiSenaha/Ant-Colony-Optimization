import csv

import matplotlib.pyplot as plt
import numpy as np

import numpy as np


# CSVファイルを読み込む
def read_log(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        data = [list(map(int, row)) for row in reader]
    return data


# 世代ごとのボトルネック帯域の平均値を計算
def calculate_average_bandwidth(log_data):
    return np.mean(log_data, axis=0)


# ログデータを読み込む
interest_log_data = read_log("./simulation_result/log_interest.csv")
rand_log_data = read_log("./simulation_result/log_rand.csv")

# 平均値を計算
interest_avg_bandwidth = calculate_average_bandwidth(interest_log_data)
rand_avg_bandwidth = calculate_average_bandwidth(rand_log_data)

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(interest_avg_bandwidth, label="Interest")
plt.plot(rand_avg_bandwidth, label="Rand")
plt.xlabel("Generation")
plt.ylabel("Average Bottleneck Bandwidth")
plt.title("Average Bottleneck Bandwidth per Generation")
plt.legend()
plt.grid(True)
plt.show()
