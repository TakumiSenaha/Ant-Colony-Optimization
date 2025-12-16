"""
最適解到達率と品質スコアの分析スクリプト（新ログ対応）

- 入力: ant_solution_log.csv（共通形式）
- 出力: 任意の指標を選択してグラフ化

使い方（例）:
  # プロジェクトルートから
  python aco_moo_routing/analysis/analyze_optimal_percentage.py \
    --csv aco_moo_routing/results/proposed/static/bandwidth_only/ant_solution_log.csv \
    --generations 1000 --ants 10 --metric optimal_rate

  # 生成されるファイル: 指定CSVと同じフォルダに {metric}.svg
  # metric: optimal_rate | unique_optimal_rate | avg_quality | max_quality
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    print(
        "⚠️ japanize_matplotlib が見つかりません。日本語ラベルが正しく表示されない可能性があります。"
    )

# グラフ描画設定
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 18
FIGURE_WIDTH = 10  # グラフの横幅（論文形式で統一）
FIGURE_HEIGHT = 7  # グラフの縦幅（論文形式で統一）


def chunk_rows(
    rows: List[List[str]], ants: int, generations: int
) -> List[List[List[str]]]:
    """行リストをシミュレーション単位にチャンクする"""
    chunk_size = ants * generations
    if chunk_size <= 0:
        raise ValueError("ants と generations は正の整数である必要があります。")
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def load_ant_solution_log(
    file_path: Path, ants: int, generations: int
) -> Tuple[List[Dict], List[List[str]]]:
    """
    ant_solution_log.csv を読み込み、世代ごとの集計に備えた構造を返す。
    """
    rows: List[List[str]] = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            if r:
                rows.append(r)

    # ヘッダー行を除外（先頭が "generation" ならスキップ）
    if rows and rows[0] and rows[0][0].lower() == "generation":
        rows = rows[1:]

    if not rows:
        raise ValueError(f"CSVが空です: {file_path}")

    # シミュレーション単位に分割（複数シミュレーションを1ファイルに入れる場合を考慮）
    simulations = chunk_rows(rows, ants, generations)
    parsed: List[Dict] = []
    for sim_idx, sim_rows in enumerate(simulations):
        if len(sim_rows) != ants * generations:
            print(
                f"⚠️ シミュレーション {sim_idx} の行数が想定と異なります: "
                f"{len(sim_rows)} 行 (期待: {ants * generations})"
            )
        for r in sim_rows:
            try:
                gen = int(r[0])
                ant_id = int(r[1])
                bandwidth = float(r[2])
                delay = float(r[3])
                hops = int(r[4])
                is_optimal = int(r[5])
                optimal_index = int(r[6])
                is_unique_optimal = int(r[7])
                quality_score = float(r[8])
            except (ValueError, IndexError):
                # 不正行はスキップ
                continue
            parsed.append(
                {
                    "generation": gen,
                    "ant_id": ant_id,
                    "bandwidth": bandwidth,
                    "delay": delay,
                    "hops": hops,
                    "is_optimal": is_optimal,
                    "optimal_index": optimal_index,
                    "is_unique_optimal": is_unique_optimal,
                    "quality_score": quality_score,
                    "simulation": sim_idx,
                }
            )
    return parsed, rows


def aggregate_by_generation(
    parsed: List[Dict], ants: int, generations: int
) -> Dict[str, List[float]]:
    """世代ごとの指標を計算（最適解割合、ユニーク最適解割合、平均QSなど）"""
    gen_stats: Dict[str, List[float]] = {
        "optimal_rate": [],
        "unique_optimal_rate": [],
        "avg_quality": [],  # 全アリ（全シミュレーション）の平均QS
        "max_quality": [],  # シミュレーションごとの最大QSを平均した値
    }
    if not parsed:
        return gen_stats

    sims = max(p["simulation"] for p in parsed) + 1

    # 事前に (sim, gen) ごとにバケット化して計算量を削減
    bucket: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for p in parsed:
        bucket[(p["simulation"], p["generation"])].append(p)

    for g in range(generations):
        success_any = 0
        success_unique = 0
        qs_vals_all: List[float] = []
        max_qs_per_sim: List[float] = []
        for sim in range(sims):
            ants_rows = bucket.get((sim, g), [])
            if not ants_rows:
                continue

            if any(p["is_optimal"] == 1 for p in ants_rows):
                success_any += 1
            if any(p["is_unique_optimal"] == 1 for p in ants_rows):
                success_unique += 1

            qs_sim = [p["quality_score"] for p in ants_rows if p["quality_score"] >= 0]
            if qs_sim:
                qs_vals_all.extend(qs_sim)
                max_qs_per_sim.append(max(qs_sim))

        denom = sims if sims > 0 else 1
        gen_stats["optimal_rate"].append(100 * success_any / denom)
        gen_stats["unique_optimal_rate"].append(100 * success_unique / denom)
        gen_stats["avg_quality"].append(
            sum(qs_vals_all) / len(qs_vals_all) if qs_vals_all else 0.0
        )
        gen_stats["max_quality"].append(
            sum(max_qs_per_sim) / len(max_qs_per_sim) if max_qs_per_sim else 0.0
        )

    return gen_stats


def plot_metric(
    values: List[float],
    ylabel: str,
    output_path: Path,
    y_max: float = 105.0,
):
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    x_values = list(range(len(values)))
    plt.plot(
        x_values,
        values,
        marker="o",
        linestyle="-",
        color="black",
        linewidth=2.0,
        markersize=5,
    )
    plt.ylim((0, y_max))
    plt.xlim(left=0)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONTSIZE,
        direction="out",
        length=6,
        width=1.5,
        color="black",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3,
        width=1.0,
        color="black",
    )
    ax.minorticks_on()

    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    print(f"✅ グラフを保存しました: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ant_solution_log.csv から最適解到達率/品質スコアを可視化"
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="ant_solution_log.csv のパス"
    )
    parser.add_argument("--ants", type=int, default=10, help="1世代あたりのアリ数")
    parser.add_argument(
        "--generations",
        type=int,
        required=True,
        help="世代数（ログ行数から割り切れる値）",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="optimal_rate",
        choices=["optimal_rate", "unique_optimal_rate", "avg_quality", "max_quality"],
        help="可視化する指標",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力パス（未指定ならCSVと同じフォルダに metric.svg を出力）",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    parsed, _ = load_ant_solution_log(csv_path, args.ants, args.generations)
    stats = aggregate_by_generation(parsed, args.ants, args.generations)

    metric_values = stats.get(args.metric, [])
    if not metric_values:
        print(f"❌ 指標 {args.metric} の値がありません。")
        return

    output_path = (
        Path(args.output) if args.output else csv_path.parent / f"{args.metric}.svg"
    )
    ylabel_map = {
        "optimal_rate": "Optimal Solution Ratio [%]",
        "unique_optimal_rate": "Unique Optimal Solution Ratio [%]",
        # 品質スコア: 導出ボトルネック帯域 / 最適ボトルネック帯域
        "avg_quality": "Derived Bottleneck / Optimal Bottleneck",
        "max_quality": "Derived Bottleneck / Optimal Bottleneck",
    }
    plot_metric(
        metric_values,
        ylabel_map.get(args.metric, args.metric),
        output_path,
        105.0 if "rate" in args.metric else 1.05,
    )


if __name__ == "__main__":
    main()
