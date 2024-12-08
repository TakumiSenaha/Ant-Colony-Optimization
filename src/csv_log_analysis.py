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

# 縦列が探索回数(1,2,3...)・横行(0,10,20...100)がその探索回数におけるwidthの出現回数
counts = [[0] * 11 for _ in range(len(data[0]))]

# 集計
for row in data:
    for search_count, width in enumerate(row):
        try:
            index = int(width // 10)
            if 0 <= index <= 10:  # インデックスが正しい範囲にあるかチェック
                counts[search_count][index] += 1
        except ValueError:
            print(f"Error processing width: {width}")

# 横行のwidthの出現回数の総和
totals = [sum(col) for col in counts]

# 縦列が探索回数・横行がその探索回数におけるwidthの割合
ratios = [
    [count * 100 / total if total > 0 else 0 for count in row]
    for row, total in zip(counts, totals)
]

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
        color=color[i],
        label=f"帯域幅 {10 * (10 - i)}",
    )
    bottom = [sum(x) for x in zip(bottom, row)]

# グラフの設定
plt.ylim((0, 100))
plt.xlabel("世代", fontsize=20)
plt.ylabel("ルーティング割合 [%]", fontsize=20)

# 凡例を取得して逆順に設定
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles[::-1],
    labels[::-1],
    title="ボトルネック帯域 (Mbps)",
    title_fontsize=12,
    fontsize=10,
    loc="center right",
)

plt.gca().tick_params(axis="both", which="major", labelsize=20)

plt.tight_layout()
plt.show()
