import csv
import os

import japanize_matplotlib
import matplotlib.pyplot as plt

# ===== 解析設定 =====
# シミュレーションで設定したアリの数を指定してください
ANT_NUM = 10

# グラフ描画設定
AXIS_LABEL_FONTSIZE = 28  # 軸ラベルのフォントサイズ (12-14pt推奨)
TICK_LABEL_FONTSIZE = 24  # 目盛りラベルのフォントサイズ (10-12pt推奨)
# ===================

# CSVファイル名
csv_file_name = "./simulation_result/log_ant_available_bandwidth.csv"
export_image_name = "./simulation_result/result_optimal_percentage.svg"


def process_csv_data(file_path, ant_num):
    """
    CSVデータを読み込み、世代ごとの最適解発見率を計算する。
    ant_numの値に応じて処理を切り替える。
    """
    data = []
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # 空の行をスキップ
                    data.append([int(val) for val in row])
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return []

    if not data:
        print(f"警告: CSVファイル '{file_path}' が空です。")
        return []

    num_simulations = len(data)
    optimal_percentages = []

    if ant_num == 1:
        # === ANT_NUM = 1 の場合の処理 (従来通り) ===
        print("ANT_NUM = 1 として集計します。")
        if not data[0]:
            return []  # 最初の行が空の場合
        num_generations = len(data[0])

        for gen_idx in range(num_generations):
            # その世代で成功(1)したシミュレーションの数を数える
            count_optimal = sum(row[gen_idx] == 1 for row in data)
            percentage = (count_optimal / num_simulations) * 100
            optimal_percentages.append(percentage)

    else:
        # === ANT_NUM > 1 の場合の処理 (チャンク処理) ===
        print(f"ANT_NUM = {ant_num} として集計します。")
        if not data[0]:
            return []  # 最初の行が空の場合
        total_log_entries = len(data[0])
        num_generations = total_log_entries // ant_num

        for gen_idx in range(num_generations):
            generation_success_count = 0
            # 各シミュレーション（各行）について処理
            for sim_row in data:
                start_index = gen_idx * ant_num
                end_index = start_index + ant_num
                generation_chunk = sim_row[start_index:end_index]

                # その世代のチャンク内に1が一つでもあれば、そのシミュレーションはその世代で成功と見なす
                if 1 in generation_chunk:
                    generation_success_count += 1

            percentage = (generation_success_count / num_simulations) * 100
            optimal_percentages.append(percentage)

    return optimal_percentages


# データ読み込みと処理
optimal_percentages = process_csv_data(csv_file_name, ANT_NUM)

if optimal_percentages:  # データが正常に処理された場合のみグラフ描画
    x_values = list(range(len(optimal_percentages)))
    y_values = optimal_percentages

    # グラフ描画（論文標準形式：箱型）
    plt.figure(figsize=(10, 7))  # 白銀比に近い比率
    plt.plot(
        x_values,
        y_values,
        marker="o",
        linestyle="-",
        color="black",
        linewidth=2.0,  # 線幅を太く（0.02cm以上相当）
        markersize=3,  # マーカーサイズを適度に
    )

    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Optimal Path Selection Ratio [%]", fontsize=AXIS_LABEL_FONTSIZE)

    # 論文標準の軸設定（箱型：全ての枠線を表示）
    ax = plt.gca()
    ax.spines["top"].set_visible(True)  # 上枠線を表示
    ax.spines["right"].set_visible(True)  # 右枠線を表示
    ax.spines["left"].set_visible(True)  # 左枠線を表示
    ax.spines["bottom"].set_visible(True)  # 下枠線を表示

    # 全ての枠線を黒色、適切な線幅に設定
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)  # 枠線の線幅

    # 目盛りの設定
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONTSIZE,  # 目盛りラベルのフォントサイズ
        direction="out",  # 目盛りを外向きに
        length=6,  # 主目盛りの長さ
        width=1.5,  # 目盛り線の太さ
        color="black",
    )

    # 副目盛りの設定
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3,  # 副目盛りの長さ（主目盛りより短く）
        width=1.0,  # 副目盛り線の太さ
        color="black",
    )

    # 副目盛りを有効化
    ax.minorticks_on()

    plt.tight_layout()
    plt.savefig(export_image_name, format="svg")
    print(f"グラフを {export_image_name} に保存しました。")
    plt.show()
