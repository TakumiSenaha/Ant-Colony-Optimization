import csv

import matplotlib.pyplot as plt

# ファイル名を設定
csv_file_name = "./simulation_result/log_ant.csv"

# ! データの読み込みと初期化 ------------------------------------------------

# データを読み込み
data = []
with open(csv_file_name, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            # 行データをfloat型に変換して追加
            data.append(list(map(float, row)))
        except ValueError:
            print(f"Error converting row to float: {row}")
            # エラー時はスキップ
            continue

# 世代数をデータから取得して、必要な構造を初期化
num_generations = len(data)  # 世代数（データ行数）
counts = [[0] * 11 for _ in range(num_generations)]  # 出現回数カウント用
ratios = [[0] * 11 for _ in range(num_generations)]  # 割合計算用

# ! 出現回数の集計 ----------------------------------------------------------

for search_count, row in enumerate(data):
    for width in row:
        # 幅が 0 のデータは無視
        if width == 0:
            continue
        try:
            index = int(width // 10)  # 帯域幅を10刻みでカテゴリ化
            if 0 <= index <= 10:
                counts[search_count][index] += 1  # 対応するカテゴリをインクリメント
            else:
                print(f"Width out of range: {width}")
        except (ValueError, IndexError):
            print(f"Invalid width value: {width}")

# ! 割合の計算 --------------------------------------------------------------

for search_count in range(num_generations):
    total = sum(counts[search_count])  # 各世代の総数を計算
    if total > 0:  # 総数が0でない場合にのみ割合を計算
        for i in range(11):  # 全ての帯域（0～100）に対して計算
            ratios[search_count][i] = counts[search_count][i] / total * 100

# 平均割合を計算
average_ratios = [0] * 11
for i in range(11):
    total_ratio = sum(row[i] for row in ratios)  # 各帯域の割合を合計
    average_ratios[i] = total_ratio / len(ratios)  # 平均値を計算

# 合計が100%であることを確認
print(f"Sum of averages: {sum(average_ratios):.2f}%")

# ! 結果の出力 --------------------------------------------------------------

print("ボトルネック帯域ごとの平均割合:")
for i in range(1, 11):  # 0をスキップ
    print(f"帯域幅 {i * 10}: {average_ratios[i]:.2f}%")

# ! グラフの描画 -------------------------------------------------------------

# カラーリストとラベルを設定
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
labels = [f"{i * 10}" for i in range(1, 11)]  # 0を除外

# グラフデータとラベルが一致しない場合の警告と修正
if len(average_ratios[1:]) != len(labels):
    print("Warning: Label count and average_ratios count do not match.")
    labels = [f"{i * 10}" for i in range(len(average_ratios[1:]))]
    color = color[: len(average_ratios[1:])]

# 棒グラフを描画
plt.bar(labels, average_ratios[1:], color=color, width=0.8)  # 0を除外したデータを使用

# グラフの設定
plt.xlabel("Bottleneck Bandwidth")
plt.ylabel("Average Percentage (%)")
plt.title("Average Routing Percentage by Bottleneck Bandwidth (Excluding 0)")
plt.show()
