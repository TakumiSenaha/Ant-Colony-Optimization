import csv
import os

import japanize_matplotlib
import matplotlib.pyplot as plt

# CSVファイル名
previous_csv_file_name = "./simulation_result/log_ant.csv"
dynamic_csv_file_name = "./simulation_result/log_ant_dynamic_volatillization.csv"
export_image_name = "./simulation_result/result_comparison.svg"


# 結果の集計関数
# データを集計し、世代ごとの割合を計算します
def process_csv_data(csv_file_name):
    with open(csv_file_name, "r") as f:
        reader = csv.reader(f)
        data = [list(map(float, row)) for row in reader]

    counts = [[0] * 11 for _ in range(len(data[0]))]

    # 各世代ごとの帯域幅集計
    for row in data:
        for search_count, width in enumerate(row):
            # if width > 0:  # 探索失敗 (帯域幅が0) を除外
            counts[search_count][int(width) // 10] += 1

    totals = [sum(col) for col in counts]

    # 各世代の割合計算
    ratios = [
        [count * 100 / total if total > 0 else 0 for count in row]
        for row, total in zip(counts, totals)
    ]
    return ratios


# データ読み込みと処理
previous_ratios = process_csv_data(previous_csv_file_name)
dynamic_ratios = None
if os.path.exists(dynamic_csv_file_name):
    dynamic_ratios = process_csv_data(dynamic_csv_file_name)
dynamic_ratios = None

# ボトルネック帯域幅が100Mbpsの場合の割合を抽出
x_values = list(range(len(previous_ratios)))
previous_y_values = [
    row[10] for row in previous_ratios
]  # インデックス10は帯域幅100Mbps

dynamic_y_values = None
if dynamic_ratios is not None:
    dynamic_y_values = [row[10] for row in dynamic_ratios]

# グラフ描画
plt.figure(figsize=(10, 6))

# 既存の揮発式の結果を描画
plt.scatter(
    x_values,
    previous_y_values,
    color="black",
    s=5,
)

# 完全分散型揮発式の結果を描画
if dynamic_y_values is not None:
    plt.scatter(x_values, dynamic_y_values, color="blue", s=5, label="完全分散型揮発式")

# グラフ設定
plt.ylim((0, 100))
plt.xlabel("世代", fontsize=20)
plt.ylabel("最適経路の割合 [%]", fontsize=20)
plt.legend(fontsize=15, loc="lower right")  # 凡例の位置とフォントサイズ

# 軸と枠線の強調
ax = plt.gca()
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(0.5)

# 軸フォントサイズの調整
ax.tick_params(axis="both", which="major", labelsize=20)

# 囲み線を設定
ax.set_frame_on(True)
ax.patch.set_edgecolor("black")
ax.patch.set_linewidth(1.5)

# レイアウト調整と保存
plt.tight_layout()
plt.savefig(export_image_name, format="svg")
plt.show()
