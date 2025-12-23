import csv
import os

import matplotlib.pyplot as plt

# ===== 解析設定 =====
# シミュレーションで設定したアリの数を指定してください
ANT_NUM = 10

# グラフ描画設定
AXIS_LABEL_FONTSIZE = 26  # 軸ラベルのフォントサイズ（[%]が見切れないように少し小さく）
TICK_LABEL_FONTSIZE = 24  # 目盛りラベルのフォントサイズ
FIGURE_WIDTH = 10  # グラフの横幅（論文形式で統一）
FIGURE_HEIGHT = 7  # グラフの縦幅（論文形式で統一）
# ===================

# ===== CSVファイル名 =====
# 手法の違い：
# - 既存手法（Baseline）: ある探索で得られた経路上の最大帯域エッジと最小帯域エッジを学習
#   [論文] IEICE ComEX 2025: "A distributed approach for maximum-bandwidth-path discovery
#          using ant colony optimization" (Senaha et al., DOI: 10.23919/comex.2025XBL0099)
# - 提案手法（Proposed）: ある探索で得られた経路上の最小帯域エッジ（ボトルネック帯域値）のみを学習
#                         バッファを用いたり、その値を忘れることで帯域変動に対応

# 1. 既存手法（最大帯域 = 最適解の環境）
# IEICE ComEX 2025 (Senaha et al.)
baseline_traditional_csv = (
    "./simulation_result/baseline_ieice_comex_2025_max_bandwidth_optimal.csv"
)

# 2. 既存手法（ランダム帯域環境）
# IEICE ComEX 2025 (Senaha et al.)
baseline_random_csv = (
    "./simulation_result/baseline_ieice_comex_2025_random_bandwidth.csv"
)

# 3. 提案手法（最大帯域 = 最適解の環境）
proposed_traditional_csv = "./simulation_result/proposed_max_bandwidth_optimal.csv"

# 4. 提案手法（ランダム帯域環境）
proposed_random_csv = "./simulation_result/proposed_random_bandwidth.csv"

# 帯域変動を考慮した比較用
# 5. 既存手法（帯域変動環境）
# IEICE ComEX 2025 (Senaha et al.)
baseline_fluctuation_csv = (
    "./simulation_result/baseline_ieice_comex_2025_bandwidth_fluctuation.csv"
)

# 6. 提案手法（帯域変動環境）
proposed_fluctuation_csv = "./simulation_result/proposed_bandwidth_fluctuation.csv"

export_image_name = "./simulation_result/result_comparison_four_methods.svg"
export_image_name_eps = "./simulation_result/result_comparison_four_methods.eps"
export_image_name_fluctuation = (
    "./simulation_result/result_comparison_bandwidth_fluctuation.svg"
)
export_image_name_fluctuation_eps = (
    "./simulation_result/result_comparison_bandwidth_fluctuation.eps"
)
# ========================


def process_csv_data(file_path, ant_num):
    """
    CSVデータを読み込み、世代ごとの最適解発見率を計算する。
    ant_numの値に応じて処理を切り替える。
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []

    data = []
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # 空の行をスキップ
                    data.append([int(val) for val in row])
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []

    if not data:
        print(f"Warning: CSV file '{file_path}' is empty.")
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


# ===== データ読み込みと処理 =====
baseline_traditional_percentages = process_csv_data(baseline_traditional_csv, ANT_NUM)
baseline_random_percentages = process_csv_data(baseline_random_csv, ANT_NUM)
proposed_traditional_percentages = process_csv_data(proposed_traditional_csv, ANT_NUM)
proposed_random_percentages = process_csv_data(proposed_random_csv, ANT_NUM)

# データが存在するかチェック
has_data = any(
    [
        baseline_traditional_percentages,
        baseline_random_percentages,
        proposed_traditional_percentages,
        proposed_random_percentages,
    ]
)

if has_data:
    # グラフ描画（論文標準形式：箱型）
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # 1. 既存手法（Environment 1: Ideal - 最大帯域 = 最適解）
    if baseline_traditional_percentages:
        x_values = list(range(len(baseline_traditional_percentages)))
        plt.plot(
            x_values,
            baseline_traditional_percentages,
            marker="o",
            linestyle="-",
            color="lightgray",
            linewidth=2.0,
            markersize=3,
            label="Previous method (Environment 1)",
        )

    # 2. 既存手法（Environment 2: Random）
    if baseline_random_percentages:
        x_values = list(range(len(baseline_random_percentages)))
        plt.plot(
            x_values,
            baseline_random_percentages,
            marker="o",
            linestyle="--",
            color="gainsboro",
            linewidth=2.0,
            markersize=3,
            label="Previous method (Environment 2)",
        )

    # 3. 提案手法（Environment 1: Ideal - 最大帯域 = 最適解）
    if proposed_traditional_percentages:
        x_values = list(range(len(proposed_traditional_percentages)))
        plt.plot(
            x_values,
            proposed_traditional_percentages,
            marker="s",
            linestyle="-",
            color="black",
            linewidth=2.0,
            markersize=3,
            label="Proposed method (Environment 1)",
        )

    # 4. 提案手法（Environment 2: Random）
    if proposed_random_percentages:
        x_values = list(range(len(proposed_random_percentages)))
        plt.plot(
            x_values,
            proposed_random_percentages,
            marker="s",
            linestyle="--",
            color="gray",
            linewidth=2.0,
            markersize=3,
            label="Proposed method (Environment 2)",
        )

    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Optimal Path Selection Ratio [%]", fontsize=AXIS_LABEL_FONTSIZE)
    plt.legend(fontsize=16, loc="best", frameon=True, ncol=1)

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
    plt.savefig(export_image_name_eps, format="eps")
    print(f"Graph saved to: {export_image_name}")
    print(f"Graph saved to: {export_image_name_eps}")

    # ===== Statistical Information Display =====
    print("\n" + "=" * 60)
    print("Statistical Summary")
    print("=" * 60)

    if baseline_traditional_percentages:
        print("\n1. Previous Method (Environment 1):")
        final_rate = baseline_traditional_percentages[-1]
        avg_rate = sum(baseline_traditional_percentages) / len(
            baseline_traditional_percentages
        )
        print(f"   Final generation success rate: {final_rate:.2f}%")
        print(f"   Average success rate: {avg_rate:.2f}%")

    if baseline_random_percentages:
        print("\n2. Previous Method (Environment 2):")
        final_rate = baseline_random_percentages[-1]
        avg_rate = sum(baseline_random_percentages) / len(baseline_random_percentages)
        print(f"   Final generation success rate: {final_rate:.2f}%")
        print(f"   Average success rate: {avg_rate:.2f}%")

    if proposed_traditional_percentages:
        print("\n3. Proposed Method (Environment 1):")
        final_rate = proposed_traditional_percentages[-1]
        avg_rate = sum(proposed_traditional_percentages) / len(
            proposed_traditional_percentages
        )
        print(f"   Final generation success rate: {final_rate:.2f}%")
        print(f"   Average success rate: {avg_rate:.2f}%")

    if proposed_random_percentages:
        print("\n4. Proposed Method (Environment 2):")
        final_rate = proposed_random_percentages[-1]
        avg_rate = sum(proposed_random_percentages) / len(proposed_random_percentages)
        print(f"   Final generation success rate: {final_rate:.2f}%")
        print(f"   Average success rate: {avg_rate:.2f}%")

    plt.show()

    # ===== 帯域変動を考慮した比較グラフ（2つ目のグラフ）=====
    # データ読み込み
    baseline_fluctuation_percentages = process_csv_data(
        baseline_fluctuation_csv, ANT_NUM
    )
    proposed_fluctuation_percentages = process_csv_data(
        proposed_fluctuation_csv, ANT_NUM
    )

    # データが存在するかチェック
    has_fluctuation_data = any(
        [baseline_fluctuation_percentages, proposed_fluctuation_percentages]
    )

    if has_fluctuation_data:
        # グラフ描画（論文標準形式：箱型）
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

        # 既存手法（帯域変動環境）
        if baseline_fluctuation_percentages:
            x_values = list(range(len(baseline_fluctuation_percentages)))
            plt.plot(
                x_values,
                baseline_fluctuation_percentages,
                marker="o",
                linestyle="-",
                color="lightgray",
                linewidth=2.0,
                markersize=3,
                label="Previous method (Fluctuation environment)",
            )

        # 提案手法（帯域変動環境）
        if proposed_fluctuation_percentages:
            x_values = list(range(len(proposed_fluctuation_percentages)))
            plt.plot(
                x_values,
                proposed_fluctuation_percentages,
                marker="s",
                linestyle="-",
                color="black",
                linewidth=2.0,
                markersize=3,
                label="Proposed method (Fluctuation environment)",
            )

        plt.ylim((0, 105))
        plt.xlim(left=0)
        plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("Optimal Path Selection Ratio [%]", fontsize=AXIS_LABEL_FONTSIZE)
        plt.legend(fontsize=16, loc="best", frameon=True, ncol=1)

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
        plt.savefig(export_image_name_fluctuation, format="svg")
        plt.savefig(export_image_name_fluctuation_eps, format="eps")
        print(f"Graph saved to: {export_image_name_fluctuation}")
        print(f"Graph saved to: {export_image_name_fluctuation_eps}")

        # 統計情報の表示
        print("\n" + "=" * 60)
        print("Bandwidth Fluctuation Comparison - Statistical Summary")
        print("=" * 60)

        if baseline_fluctuation_percentages:
            print("\nPrevious Method (Fluctuation Environment):")
            final_rate = baseline_fluctuation_percentages[-1]
            avg_rate = sum(baseline_fluctuation_percentages) / len(
                baseline_fluctuation_percentages
            )
            print(f"   Final generation success rate: {final_rate:.2f}%")
            print(f"   Average success rate: {avg_rate:.2f}%")

        if proposed_fluctuation_percentages:
            print("\nProposed Method (Fluctuation Environment):")
            final_rate = proposed_fluctuation_percentages[-1]
            avg_rate = sum(proposed_fluctuation_percentages) / len(
                proposed_fluctuation_percentages
            )
            print(f"   Final generation success rate: {final_rate:.2f}%")
            print(f"   Average success rate: {avg_rate:.2f}%")

        plt.show()
    else:
        print(
            "\nWarning: No bandwidth fluctuation data to plot. "
            "Please check CSV files."
        )

else:
    print("Error: No data to plot. Please check CSV files.")
