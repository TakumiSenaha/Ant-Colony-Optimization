import csv

import japanize_matplotlib
import matplotlib.pyplot as plt

csv_file_name = "./simulation_result/log_ant.csv"
export_image_name = "./simulation_result/result_previous.SVG"

# 結果の集計---------------------------------------------------------------

# CSVファイルを読み込む
with open(csv_file_name, "r") as f:
    reader = csv.reader(f)
    data = [list(map(float, row)) for row in reader]

# 縦列が探索回数(1,2,3...)・横行(0,10,20...100)がその探索回数におけるwidthの出現回数
counts = [[0] * 11 for _ in range(len(data[0]))]

# 集計
for row in data:
    for search_count, width in enumerate(row):
        counts[search_count][int(width) // 10] += 1

# 横行のwidthの出現回数の総和
totals = [sum(col) for col in counts]

# 縦列が探索回数・横行がその探索回数におけるwidthの割合
ratios: list[list[float]] = [
    [count * 100 / total if total > 0 else 0 for count in row]
    for row, total in zip(counts, totals)
]

# グラフ描写---------------------------------------------------------------

# データの用意
data = ratios

# ボトルネック帯域が100であるものの確率を散布図で描画
x_values = list(range(len(data)))
y_values = [row[10] for row in data]  # 100の幅に対応する列はインデックス10

plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color="black", s=5)  # 点のサイズを調整

# グラフの設定
plt.ylim((0, 100))

plt.xlabel("世代", fontsize=20)
plt.ylabel("最適経路の割合 [%]", fontsize=20)

# 縦軸と横軸の囲いを表示
ax = plt.gca()
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)

# 軸の枠線を強調（黒色、線幅を設定）
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(0.5)

# フォントサイズを2倍に設定
ax.tick_params(axis="both", which="major", labelsize=20)

# 四角で囲む
ax.set_frame_on(True)
ax.patch.set_edgecolor("black")
ax.patch.set_linewidth(1.5)

plt.tight_layout()
plt.savefig(export_image_name, format="svg")
plt.show()
