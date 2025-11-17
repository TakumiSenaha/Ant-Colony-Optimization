"""
BKB学習パラメータの比較分析スクリプト（統合版）

複数のCSVファイルを比較してグラフを生成します。
FILES_COMPARISONをコメントアウトすることで、自由に表示する系列を変更できます。
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
LEGEND_FONTSIZE = 12  # 凡例のフォントサイズ（小さめに設定）

# ===== 比較セットの定義 =====
# 以下のFILES_COMPARISONをコメントアウト/アンコメントすることで、
# 自由に表示する系列を変更できます。

# --- 比較セット1: 忘却率1.0、0.99、0.9 × 記憶観測値数10、100、1000 = 9系列 ---
FILES_COMPARISON_1 = {
    # 忘却率1.0
    "./simulation_result/log_ant_available_bandwidth_1_10.csv": "忘却率1.0, 記憶観測値数: 10",
    # "./simulation_result/log_ant_available_bandwidth_1_100.csv": "忘却率1.0, 記憶観測値数: 100",
    # "./simulation_result/log_ant_available_bandwidth_1_1000.csv": "忘却率1.0, 記憶観測値数: 1000",
    # 忘却率0.99
    "./simulation_result/log_ant_available_bandwidth_0999_10.csv": "忘却率0.99, 記憶観測値数: 10",
    # "./simulation_result/log_ant_available_bandwidth_0999_100.csv": "忘却率0.99, 記憶観測値数: 100",
    # "./simulation_result/log_ant_available_bandwidth_0999_1000.csv": "忘却率0.99, 記憶観測値数: 1000",
    # 忘却率0.9
    "./simulation_result/log_ant_available_bandwidth_09_10.csv": "忘却率0.9, 記憶観測値数: 10",
    # "./simulation_result/log_ant_available_bandwidth_09_100.csv": "忘却率0.9, 記憶観測値数: 100",
    # "./simulation_result/log_ant_available_bandwidth_09_1000.csv": "忘却率0.9, 記憶観測値数: 1000",
}

# --- 比較セット2: 忘却率1.0と0.9の比較（リングバッファサイズ1000） ---
FILES_COMPARISON_2 = {
    "./simulation_result/log_ant_available_bandwidth_1_0.csv": "忘却率1.0, 記憶観測値数: 1000",
    "./simulation_result/log_ant_available_bandwidth_0_9.csv": "忘却率0.9, 記憶観測値数: 1000",
}

# --- 比較セット3: 忘却率1.0の記憶観測値数比較（10, 100, 1000） ---
FILES_COMPARISON_3 = {
    "./simulation_result/log_ant_available_bandwidth_1_10.csv": "忘却率1.0, 記憶観測値数: 10",
    "./simulation_result/log_ant_available_bandwidth_1_100.csv": "忘却率1.0, 記憶観測値数: 100",
    "./simulation_result/log_ant_available_bandwidth_1_1000.csv": "忘却率1.0, 記憶観測値数: 1000",
}

# --- 比較セット4: 忘却率0.9の記憶観測値数比較（10, 100, 1000） ---
FILES_COMPARISON_4 = {
    "./simulation_result/log_ant_available_bandwidth_09_10.csv": "忘却率0.9, 記憶観測値数: 10",
    "./simulation_result/log_ant_available_bandwidth_09_100.csv": "忘却率0.9, 記憶観測値数: 100",
    "./simulation_result/log_ant_available_bandwidth_09_1000.csv": "忘却率0.9, 記憶観測値数: 1000",
}

# ===== カスタム色設定 =====
# 各比較セットに対してカスタム色を指定する場合
# Noneの場合は自動色選択、リストを指定するとその色を使用
# 例: [("#E31A1C", "-", 2.0), ("#1F78B4", "-", 2.0), ...]
# 形式: [(色, 線種, 線幅), ...]

# 比較セット1用の色（9系列）
COLORS_SET_1 = [
    # 忘却率1.0
    ("#E31A1C", "-", 2.0),  # 記憶観測値数10: 赤
    ("#1F78B4", "-", 2.0),  # 記憶観測値数100: 青
    ("#33A02C", "-", 2.0),  # 記憶観測値数1000: 緑
    # 忘却率0.99
    ("#FF7F00", "-", 2.0),  # 記憶観測値数10: オレンジ
    ("#6A3D9A", "-", 2.0),  # 記憶観測値数100: 紫
    ("#B15928", "-", 2.0),  # 記憶観測値数1000: 茶色
    # 忘却率0.9
    ("#FB9A99", "-", 2.0),  # 記憶観測値数10: ピンク
    ("#A6CEE3", "-", 2.0),  # 記憶観測値数100: 水色
    ("#B2DF8A", "-", 2.0),  # 記憶観測値数1000: ライムグリーン
]

# 比較セット2用の色（2系列）
COLORS_SET_2 = [
    ("black", "-", 2.0),  # 忘却率1.0
    ("blue", "-", 2.0),  # 忘却率0.9
]

# ===== 使用する比較セット =====
# 表示したい比較セットをコメントアウト/アンコメントしてください
# 複数のセットを同時に使用できます（それぞれ別のグラフが生成されます）

COMPARISON_SETS = [
    # ("比較セット名", FILES_COMPARISON, 出力ファイル名, 色設定)
    (
        "忘却率×記憶観測値数（9系列）",
        FILES_COMPARISON_1,
        "./simulation_result/result_bkb_evap_window_comparison.svg",
        COLORS_SET_1,
    ),
    # (
    #     "忘却率比較（1.0 vs 0.9）",
    #     FILES_COMPARISON_2,
    #     "./simulation_result/result_bkb_evap_comparison.svg",
    #     COLORS_SET_2,
    # ),
    # (
    #     "忘却率1.0の記憶観測値数比較",
    #     FILES_COMPARISON_3,
    #     "./simulation_result/result_bkb_evap_1.0_comparison.svg",
    #     None,
    # ),
    # (
    #     "忘却率0.9の記憶観測値数比較",
    #     FILES_COMPARISON_4,
    #     "./simulation_result/result_bkb_evap_0.9_comparison.svg",
    #     None,
    # ),
]

# 自動色パレット（カスタム色がNoneの場合に使用）
AUTO_COLORS = [
    "black",
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
]


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


def plot_comparison_graph(
    files_dict, output_filename, colors_and_styles=None, title_suffix=""
):
    """
    複数のCSVファイルを比較してグラフを描画する（論文標準スタイル）

    Args:
        files_dict: {ファイルパス: 凡例ラベル（日本語）} の辞書
        output_filename: 出力ファイル名
        colors_and_styles: [(色, 線種, 線幅), ...] のリスト
                          （Noneの場合は自動選択）
        title_suffix: 未使用（互換性のため保持）
    """
    if colors_and_styles is None:
        # 自動色選択
        colors_and_styles = [
            (AUTO_COLORS[i % len(AUTO_COLORS)], "-", 2.0)
            for i in range(len(files_dict))
        ]

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

    # 凡例を追加（系列数に応じて列数を調整、日本語対応）
    num_series = len(files_dict)
    ncol = 2 if num_series > 5 else 1

    ax.legend(
        loc="best",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        ncol=ncol,
    )

    plt.tight_layout()
    plt.savefig(output_filename, format="svg")
    print(f"✅ グラフを {output_filename} に保存しました。")
    plt.show()  # 画面に表示


# ===== メイン処理 =====
if __name__ == "__main__":
    print("=" * 70)
    print("BKB学習パラメータの比較分析スクリプト（統合版）")
    print("=" * 70)

    # 各比較セットに対してグラフを生成
    generated_count = 0
    for (
        set_name,
        files_dict,
        output_filename,
        custom_colors,
    ) in COMPARISON_SETS:
        if not files_dict:  # 空の辞書はスキップ
            continue

        print(f"\n📊 {set_name} のグラフを生成中...")
        print(f"   系列数: {len(files_dict)}")
        print(f"   出力ファイル: {output_filename}")

        # カスタム色が定義されている場合はそれを使用
        colors = custom_colors if custom_colors is not None else None

        plot_comparison_graph(
            files_dict,
            output_filename,
            colors_and_styles=colors,
            title_suffix=set_name,
        )
        generated_count += 1

    print("\n" + "=" * 70)
    print(f"✅ {generated_count}個のグラフ生成が完了しました！")
    print("=" * 70)
