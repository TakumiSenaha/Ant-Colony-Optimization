import csv

import japanize_matplotlib
import matplotlib.pyplot as plt


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
        failure_rate = (
            gen_values.count(0) / len(gen_values)
        ) * 100  # パーセンテージに変換
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
plt.plot(
    range(len(interest_failure_rate_per_generation)),
    interest_failure_rate_per_generation,
    label="Interest",
    color="blue",
)
plt.plot(
    range(len(rand_failure_rate_per_generation)),
    rand_failure_rate_per_generation,
    label="Rand",
    color="orange",
)

plt.xlabel("世代", fontsize=20)
plt.ylabel("パケットロス率（探索失敗率）[%]", fontsize=20)

# 縦軸と横軸の囲いを表示
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)

# フォントサイズを2倍に設定
ax.tick_params(axis="both", which="major", labelsize=20)

plt.grid(False)
plt.show()
