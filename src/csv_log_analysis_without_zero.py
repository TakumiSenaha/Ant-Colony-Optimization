import csv

import japanize_matplotlib
import matplotlib.pyplot as plt

csv_file_name = "./simulation_result/log_ant.csv"
export_image_name = "./simulation_result/result_previous.SVG"

# ! 結果の集計---------------------------------------------------------------

# CSVファイルを読み込む
with open(csv_file_name, "r") as f:
    reader = csv.reader(f)
    data = []
    for row in reader:
        try:
            data.append(list(map(float, row)))
        except ValueError:
            print(f"Error converting row to float: {row}")

# 縦列が探索回数(1,2,3...)・横行(10,20...100)がその探索回数におけるwidthの出現回数
counts = [[0] * 10 for _ in range(len(data[0]))]  # 0 Mbps を除外して 10 区間に修正

# 集計 (帯域幅 0 を除外)
for row in data:
    for search_count, width in enumerate(row):
        try:
            if width > 0:  # 帯域幅 0 を除外
                index = int(width // 10) - 1  # インデックスを調整 (-1で帯域幅 0 を除外)
                if 0 <= index < 10:  # インデックスが正しい範囲にあるかチェック
                    counts[search_count][index] += 1
        except ValueError:
            print(f"Error processing width: {width}")

# 横行のwidthの出現回数の総和 (帯域幅 0 を除外した結果)
totals = [sum(col) for col in counts]

# 縦列が探索回数・横行がその探索回数におけるwidthの割合
ratios = [
    [count * 100 / total if total > 0 else 0 for count in row]
    for row, total in zip(counts, totals)
]

# ! グラフ描写---------------------------------------------------------------
# 棒グラフの棒のカラー (元の順序を維持、0 Mbps を削除)
color = [
    "#4F71BE",  # 100 Mbps
    "#DE8344",  # 90 Mbps
    "#A5A5A5",  # 80 Mbps
    "#F1C242",  # 70 Mbps
    "#6A99D0",  # 60 Mbps
    "#7EAB54",  # 50 Mbps
    "#2D4374",  # 40 Mbps
    "#934D21",  # 30 Mbps
    "#636363",  # 20 Mbps
    "#937424",  # 10 Mbps
]

# データの左右反転
reversed_data = [row[::-1] for row in ratios]

# 表示するラベルの用意
labels = list(range(len(reversed_data)))

# 各行の合計値を計算し、それを元に割合を計算
totals = [sum(row) for row in reversed_data]
proportions = [
    [val * 100 / total if total > 0 else 0 for val in row]
    for total, row in zip(totals, reversed_data)
]

# 積み上げ棒グラフを描画(width = 100 -> width = 10)の順に描写
# 積み上げのために二次元配列を転置
transpose = list(map(list, zip(*proportions)))

# グラフ描画
bottom = [0] * len(labels)
plt.figure(figsize=(10, 6))
for i, row in enumerate(transpose):
    plt.bar(
        labels,
        row,
        width=1.0,
        bottom=bottom,
        color=color[i],  # Specify corresponding color
        label=f"Bandwidth {10 * (10 - i)} Mbps",  # Maintain original order in label
    )
    bottom = [sum(x) for x in zip(bottom, row)]

# Graph settings
plt.ylim((0, 100))
plt.xlabel("Generation", fontsize=20)
plt.ylabel("Routing Ratio [%]", fontsize=20)

# Reverse legend order
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles[::-1],
    labels[::-1],
    title="Bottleneck Bandwidth (Mbps)",
    title_fontsize=12,
    fontsize=10,
    loc="lower right",
)

plt.gca().tick_params(axis="both", which="major", labelsize=20)

plt.tight_layout()
plt.show()
