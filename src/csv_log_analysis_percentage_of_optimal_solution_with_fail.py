import csv

import japanize_matplotlib
import matplotlib.pyplot as plt

# ファイル名
input_csv_file_name = "./simulation_result/log_ant.csv"
output_image_name = "./simulation_result/log_ant_analysis.svg"


# 結果の集計関数
def process_csv_data(csv_file_name):
    """
    CSVデータを集計し、探索失敗率と成功した割合を計算する
    - 探索失敗（0）の数をカウント
    - 成功した探索（0以外）について割合を計算
    """
    with open(csv_file_name, "r") as f:
        reader = csv.reader(f)
        data = [list(map(float, row)) for row in reader]

    # 初期化
    counts = [[0] * 11 for _ in range(len(data[0]))]  # 各帯域幅区間（10刻み）のカウント
    failure_counts = [0] * len(data[0])  # 探索失敗（0の数）
    success_totals = [0] * len(data[0])  # 成功（0以外の合計）

    # データ処理
    for row in data:
        for generation_index, width in enumerate(row):
            if width == 0:  # 探索失敗
                failure_counts[generation_index] += 1
            else:  # 成功
                counts[generation_index][int(width) // 10] += 1
                success_totals[generation_index] += 1

    # 成功した割合を計算
    success_ratios = [
        [count * 100 / total if total > 0 else 0 for count in row]
        for row, total in zip(counts, success_totals)
    ]
    return success_ratios, failure_counts


# データの処理
success_ratios, failure_counts = process_csv_data(input_csv_file_name)

# ボトルネック帯域幅が100Mbpsの場合の割合を抽出
x_values = list(range(len(success_ratios)))
success_y_values = [row[10] for row in success_ratios]  # インデックス10は帯域幅100Mbps

# 探索失敗率の計算
failure_rates = [
    f * 100 / (f + s) if (f + s) > 0 else 0
    for f, s in zip(failure_counts, [sum(row) for row in success_ratios])
]

# グラフ描画
plt.figure(figsize=(10, 6))

# 成功した割合（帯域幅100Mbps）を描画
plt.scatter(
    x_values,
    success_y_values,
    color="blue",
    s=5,
    label="成功割合（帯域幅100Mbps）",
)

# 探索失敗率を描画
plot_failure_rates = True  # Trueにすると探索失敗率を描画
if plot_failure_rates:
    plt.plot(
        x_values,
        failure_rates,
        color="red",
        linestyle="--",
        label="探索失敗率",
    )

# グラフ設定
plt.ylim((0, 100))
plt.xlabel("世代", fontsize=20)
plt.ylabel("割合 [%]", fontsize=20)
plt.legend(fontsize=15, loc="center right")  # 凡例の位置とフォントサイズ

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
plt.savefig(output_image_name, format="svg")
plt.show()
