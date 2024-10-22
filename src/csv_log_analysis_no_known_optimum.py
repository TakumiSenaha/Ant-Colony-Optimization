import csv

import matplotlib.pyplot as plt

csv_file_name = "./simulation_result/log_ant.csv"

# ! 結果の集計---------------------------------------------------------------

# CSVファイルを読み込む
with open(csv_file_name, "r") as f:
    reader = csv.reader(f)
    data = []
    for row in reader:
        try:
            data.append(list(map(float, row)))  # float型も受け付けるように変更
        except ValueError:
            print(f"Error converting row to float: {row}")

# 縦列がボトルネック帯域の値・横行(0,10,20...100)がその帯域における回数
counts = [[0] * 11 for _ in range(1000)]  # 1000世代分

# 各ボトルネック帯域における出現回数を集計
for row in data:
    for search_count, width in enumerate(row):
        try:
            index = int(width // 10)
            if 0 <= index <= 10:
                counts[search_count][index] += 1  # ボトルネック帯域の出現回数
        except ValueError:
            print(f"Invalid width value: {width}")

# 各世代におけるボトルネック帯域の割合を計算
ratios = [[0] * 11 for _ in range(1000)]  # ボトルネック帯域ごとの割合を格納

for search_count in range(1000):
    total = sum(counts[search_count])  # 0帯域を含む総和
    if total > 0:  # 出現回数がある場合のみ割合を計算
        for i in range(11):  # 0を含む全ての帯域を対象に計算
            ratios[search_count][i] = counts[search_count][i] / total * 100

# 平均を取るために転置し、各帯域ごとに100回のシミュレーションの平均を計算
average_ratios = [0] * 11

for i in range(11):  # 0も含む全ての帯域を対象に平均を計算
    total_ratio = sum(row[i] for row in ratios)  # 各ボトルネック帯域の割合の総和
    average_ratios[i] = total_ratio / len(ratios)  # 平均割合

# 結果を表示
print("ボトルネック帯域ごとの平均割合 (0も含む):")
for i in range(11):
    print(f"帯域幅 {i * 10}: {average_ratios[i]:.2f}%")

# ! グラフ描写---------------------------------------------------------------
# 棒グラフの棒のカラー
color = [
    "#4F71BE",
    "#DE8344",
    "#A5A5A5",
    "#F1C242",
    "#6A99D0",
    "#7EAB54",
    "#2D4374",
    "#934D21",
    "#636363",
    "#937424",
    "#355D8D",
]

# グラフ描写用のラベル（0, 10, 20, 30,... の表記に変更）
labels = [f"{i * 10}" for i in range(11)]  # 0を含む

# 棒グラフを描画
plt.bar(labels, average_ratios, color=color, width=0.8)

# グラフの設定
plt.xlabel("Bottleneck Bandwidth")
plt.ylabel("Average Percentage (%)")
plt.title("Average Routing Percentage by Bottleneck Bandwidth (Including 0)")
plt.show()
