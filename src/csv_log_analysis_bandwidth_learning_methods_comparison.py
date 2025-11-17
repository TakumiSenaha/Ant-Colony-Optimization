"""
可用帯域学習手法の比較分析スクリプト

4つの異なる予測手法・周期性検出手法の組み合わせを比較してグラフを生成します。
- AR(1)予測 + 自己相関周期性検出
- MA予測 + 自己相関周期性検出
- EMA予測 + 自己相関周期性検出
- AR(1)予測 + ウェーブレット周期性検出
"""

import csv

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt

# ===== 解析設定 =====
# シミュレーションで設定したアリの数を指定してください
ANT_NUM = 10

# グラフ描画設定
AXIS_LABEL_FONTSIZE = 28  # 軸ラベルのフォントサイズ
TICK_LABEL_FONTSIZE = 24  # 目盛りラベルのフォントサイズ
LEGEND_FONTSIZE = 14  # 凡例のフォントサイズ

# ===== 比較ファイルの定義 =====
FILES_COMPARISON = {
    "./simulation_result/log_ant_available_bandwidth_ar1_autocorr.csv": "AR(1)予測 + 自己相関",
    "./simulation_result/log_ant_available_bandwidth_ma_autocorr.csv": "MA予測 + 自己相関",
    "./simulation_result/log_ant_available_bandwidth_ema_autocorr.csv": "EMA予測 + 自己相関",
    "./simulation_result/log_ant_available_bandwidth_ar1_wavelet.csv": "AR(1)予測 + ウェーブレット",
}

# ===== カスタム色設定 =====
# 4系列用の色設定（論文標準：色覚多様性に配慮）
COLORS_AND_STYLES = [
    ("#E31A1C", "-", 2.5),  # AR(1)予測 + 自己相関: 赤
    ("#1F78B4", "-", 2.5),  # MA予測 + 自己相関: 青
    ("#33A02C", "-", 2.5),  # EMA予測 + 自己相関: 緑
    ("#FF7F00", "-", 2.5),  # AR(1)予測 + ウェーブレット: オレンジ
]

# 出力ファイル名
OUTPUT_FILENAME = "./simulation_result/result_bandwidth_learning_methods_comparison.svg"


def process_csv_data(file_path, ant_num):  # noqa: C901
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


def plot_comparison_graph(files_dict, output_filename, colors_and_styles):
    """
    複数のCSVファイルを比較してグラフを描画する（論文標準スタイル）

    Args:
        files_dict: {ファイルパス: 凡例ラベル（日本語）} の辞書
        output_filename: 出力ファイル名
        colors_and_styles: [(色, 線種, 線幅), ...] のリスト
    """
    # 論文標準スタイル：白銀比に近い比率
    plt.figure(figsize=(10, 7))

    color_idx = 0
    for file_path, label in files_dict.items():
        optimal_percentages = process_csv_data(file_path, ANT_NUM)

        if not optimal_percentages:
            print(f"警告: {file_path} のデータが空です。スキップします。")
            color_idx += 1
            continue

        x_values = list(range(len(optimal_percentages)))
        y_values = optimal_percentages

        color, linestyle, linewidth = colors_and_styles[
            color_idx % len(colors_and_styles)
        ]

        plt.plot(
            x_values,
            y_values,
            marker="o",
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,  # 論文標準：線幅を太く（0.02cm以上相当）
            markersize=5,  # 論文標準：マーカーサイズを適度に
            label=label,  # 日本語の凡例ラベル
        )
        color_idx += 1

    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Optimal Path Selection Ratio [%]", fontsize=AXIS_LABEL_FONTSIZE)
    # タイトルは付けない（論文標準）

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
        labelsize=TICK_LABEL_FONTSIZE,
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

    # 凡例を追加（4系列なので2列で表示）
    ax.legend(
        loc="best",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        ncol=2,  # 2列で表示
    )

    plt.tight_layout()
    plt.savefig(output_filename, format="svg")
    print(f"✅ グラフを {output_filename} に保存しました。")
    plt.show()  # 画面に表示


# ===== メイン処理 =====
if __name__ == "__main__":
    print("=" * 70)
    print("可用帯域学習手法の比較分析スクリプト")
    print("=" * 70)
    print(f"\n📊 比較対象: {len(FILES_COMPARISON)}つの学習手法")
    print(f"   出力ファイル: {OUTPUT_FILENAME}")
    print("\n比較内容:")
    for file_path, label in FILES_COMPARISON.items():
        print(f"  - {label}: {file_path}")

    print("\n" + "=" * 70)
    print("グラフ生成中...")
    print("=" * 70)

    plot_comparison_graph(
        FILES_COMPARISON,
        OUTPUT_FILENAME,
        COLORS_AND_STYLES,
    )

    print("\n" + "=" * 70)
    print("✅ グラフ生成が完了しました！")
    print("=" * 70)
