"""
最適解到達率の分析スクリプト

既存実装（csv_log_analysis_percentage_of_optimal_solution_use_modified_dijkstra.py）と
同じ形式で、CSVログから最適解到達率を計算し、グラフを描画します。
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import japanize_matplotlib
except ImportError:
    print(
        "⚠️  Warning: japanize_matplotlib not available. Japanese labels may not display correctly."
    )

# ===== 解析設定 =====
# シミュレーションで設定したアリの数を指定してください
ANT_NUM = 10

# グラフ描画設定
AXIS_LABEL_FONTSIZE = 28  # 軸ラベルのフォントサイズ
TICK_LABEL_FONTSIZE = 24  # 目盛りラベルのフォントサイズ
# ===================


def process_csv_data(file_path: Path, ant_num: int) -> list:
    """
    CSVデータを読み込み、世代ごとの最適解発見率を計算する。

    Args:
        file_path: CSVファイルのパス
        ant_num: アリの数

    Returns:
        世代ごとの最適解到達率のリスト（パーセンテージ）
    """
    data = []
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # 空の行をスキップ
                    data.append([int(val) for val in row])
    except FileNotFoundError:
        print(f"❌ エラー: ファイルが見つかりません: {file_path}")
        return []
    except Exception as e:
        print(f"❌ エラー: ファイル読み込み中にエラーが発生しました: {e}")
        return []

    if not data:
        print(f"⚠️  警告: CSVファイル '{file_path}' が空です。")
        return []

    num_simulations = len(data)
    optimal_percentages = []

    if ant_num == 1:
        # === ANT_NUM = 1 の場合の処理 ===
        print(f"ANT_NUM = {ant_num} として集計します。")
        if not data[0]:
            return []
        num_generations = len(data[0])

        for gen_idx in range(num_generations):
            # その世代で成功(1)したシミュレーションの数を数える
            count_optimal = sum(row[gen_idx] == 1 for row in data)
            percentage = (count_optimal / num_simulations) * 100
            optimal_percentages.append(percentage)

    else:
        # === ANT_NUM > 1 の場合の処理（チャンク処理） ===
        print(f"ANT_NUM = {ant_num} として集計します。")
        if not data[0]:
            return []
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


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description="最適解到達率の分析スクリプト")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSVログファイルのパス（未指定の場合は最新の結果ディレクトリを検索）",
    )
    parser.add_argument(
        "--ants",
        type=int,
        default=10,
        help="アリの数（デフォルト: 10）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力画像ファイルのパス（未指定の場合はCSVと同じディレクトリ）",
    )
    args = parser.parse_args()

    # CSVファイルのパスを決定
    if args.csv:
        csv_file_path = Path(args.csv)
    else:
        # 最新の結果ディレクトリを検索
        results_dir = project_root / "results"
        if not results_dir.exists():
            print(f"❌ エラー: 結果ディレクトリが見つかりません: {results_dir}")
            return

        # タイムスタンプ順にソートして最新を取得
        result_dirs = sorted(results_dir.glob("*"), reverse=True)
        if not result_dirs:
            print(f"❌ エラー: 結果ディレクトリが空です: {results_dir}")
            return

        csv_file_path = result_dirs[0] / "log_ant_available_bandwidth.csv"
        if not csv_file_path.exists():
            print(f"❌ エラー: CSVファイルが見つかりません: {csv_file_path}")
            return

    print(f"📊 分析対象: {csv_file_path}")

    # データ読み込みと処理
    optimal_percentages = process_csv_data(csv_file_path, args.ants)

    if not optimal_percentages:
        print("❌ データが正常に処理されませんでした。")
        return

    # 出力画像ファイルのパスを決定
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_file_path.parent / "result_optimal_percentage.svg"

    # グラフ描画（論文標準形式：箱型）
    x_values = list(range(len(optimal_percentages)))
    y_values = optimal_percentages

    plt.figure(figsize=(10, 7))  # 白銀比に近い比率
    plt.plot(
        x_values,
        y_values,
        marker="o",
        linestyle="-",
        color="black",
        linewidth=2.0,  # 線幅を太く（0.02cm以上相当）
        markersize=5,  # マーカーサイズを適度に
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
    plt.savefig(output_path, format="svg")
    print(f"✅ グラフを保存しました: {output_path}")

    # 統計情報を表示
    if optimal_percentages:
        final_rate = optimal_percentages[-1]
        max_rate = max(optimal_percentages)
        avg_rate = sum(optimal_percentages) / len(optimal_percentages)
        print(f"\n📈 統計情報:")
        print(f"  最終世代の到達率: {final_rate:.2f}%")
        print(f"  最大到達率: {max_rate:.2f}%")
        print(f"  平均到達率: {avg_rate:.2f}%")


if __name__ == "__main__":
    main()
