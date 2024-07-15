import csv

import matplotlib.pyplot as plt


# CSVファイルを読み込む
def read_log(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        data = [list(map(int, row)) for row in reader]
    return data


# 各世代の失敗率（0の割合）を計算
def calculate_failure_rate_per_generation(log_data):
    # 各行の長さをチェックし、最も短い行の長さを世代数とする
    generations = min(len(row) for row in log_data)
    failure_rates = []

    for gen in range(generations):
        gen_values = [row[gen] for row in log_data]
        failure_rate = gen_values.count(0) / len(gen_values)
        failure_rates.append(failure_rate)

    return failure_rates


# ログデータを読み込む
interest_log_data = read_log("./simulation_result/log_interest.csv")
rand_log_data = read_log("./simulation_result/log_rand.csv")

# 失敗率を計算
interest_failure_rate_per_generation = calculate_failure_rate_per_generation(
    interest_log_data
)
rand_failure_rate_per_generation = calculate_failure_rate_per_generation(rand_log_data)

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(interest_failure_rate_per_generation, label="Interest")
plt.plot(rand_failure_rate_per_generation, label="Rand")
plt.xlabel("Generation")
plt.ylabel("Failure Rate")
plt.title("Failure Rate per Generation")
plt.legend()
plt.grid(True)
plt.show()
plt.show()
plt.show()
