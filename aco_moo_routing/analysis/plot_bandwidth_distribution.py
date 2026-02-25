"""
Bandwidth distribution stacked bar chart script

For each generation, records the best bandwidth among 10 ants per simulation,
classifies them into 10-Mbps bins (0-10, 10-20, ..., 90-100, 100+ Mbps),
and creates a stacked bar chart aggregated over 100 simulations.

NOTE:
- In bandwidth fluctuating environments, the *absolute* bottleneck bandwidth bins are often
  not meaningful because the optimal bottleneck itself varies over time.
- This script supports a *relative* mode that bins by:
    quality_score = (found bottleneck bandwidth) / (optimal bottleneck bandwidth)
  which is already recorded in ant_solution_log.csv.

What this script analyzes (important):
- We do NOT count all ants.
- For each generation and each simulation, we take ONLY the best ant
  (max bandwidth / max quality_score).
  This means the stacked bars visualize the distribution of "best-of-generation solutions"
  across multiple simulations.

Usage:
  # From project root
  python aco_moo_routing/analysis/plot_bandwidth_distribution.py \
    --method proposed \
    --environment static \
    --opt-type bandwidth_only \
    --generations 1000 \
    --ants 10 \
    --simulations 100

  # Or specify CSV file directly
  python aco_moo_routing/analysis/plot_bandwidth_distribution.py \
    --csv aco_moo_routing/results/proposed/static/bandwidth_only/ant_solution_log.csv \
    --generations 1000 \
    --ants 10 \
    --simulations 100
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# japanize_matplotlib is not needed for English labels

# グラフ描画設定
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 18
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 6

# 帯域の分類（10刻み）: absolute mode
BANDWIDTH_BINS = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100),
    (100, float("inf")),  # 100以上
]


def sample_colormap(
    cmap_name: str,
    n: int,
    *,
    start: float = 0.25,
    end: float = 0.85,
    alpha: float = 0.9,
) -> List[Tuple[float, float, float, float]]:
    """
    Sample a matplotlib colormap into n colors.

    For academic plots, we avoid very bright/yellowish endpoints by default (start/end).
    """
    cmap = plt.get_cmap(cmap_name)
    denom = (n - 1) if n > 1 else 1
    colors: List[Tuple[float, float, float, float]] = []
    for i in range(n):
        t = start + (end - start) * (i / denom)
        r, g, b, _a = cmap(t)
        colors.append((float(r), float(g), float(b), float(alpha)))
    return colors


# 帯域ごとの色（元のコードから復元: csv_log_analysis.py を参考）
# 11個のビン（0-10, 10-20, ..., 90-100, 100+ Mbps）に対応
ABSOLUTE_COLORS = [
    "#4F71BE",  # 100 Mbps
    "#DE8344",  # 90 Mbps
    "#A5A5A5",  # 80 Mbps
    "#F1C242",  # 70 Mbps
    "#6A99D0",  # 60 Mbps
    "#7EAB54",  # 50 Mbps
    "#2D4374",  # 40 Mbps
    "#934D21",  # 30 Mbps
    "#636363",  # 20 Mbps
    "#937424",  # 10 Mbps
    "#355D8D",  # 0 Mbps
]

# 相対品質（quality_score）の分類: relative mode
# 区間定義:
# - 基本は [a, b)（左閉右開）
# - 最終ビンのみ [0.9, 1.0]（右端1.0を含む）
RELATIVE_BINS: List[Tuple[float, float]] = [
    (i / 10.0, (i + 1) / 10.0) for i in range(10)
]  # 0.0-0.1 ... 0.9-1.0

# relative mode 用の色（10個のビンに対応）
# 元のコードの色を参考に、高品質（1.0に近い）から低品質（0.0に近い）へ順に割り当て
RELATIVE_COLORS = [
    "#4F71BE",  # 0.9-1.0 (最高品質)
    "#6A99D0",  # 0.8-0.9
    "#7EAB54",  # 0.7-0.8
    "#F1C242",  # 0.6-0.7
    "#DE8344",  # 0.5-0.6
    "#A5A5A5",  # 0.4-0.5
    "#636363",  # 0.3-0.4
    "#937424",  # 0.2-0.3
    "#934D21",  # 0.1-0.2
    "#355D8D",  # 0.0-0.1 (最低品質)
]


def chunk_rows(
    rows: List[List[str]], ants: int, generations: int
) -> List[List[List[str]]]:
    """Chunk row list by simulation"""
    chunk_size = ants * generations
    if chunk_size <= 0:
        raise ValueError("ants と generations は正の整数である必要があります。")
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def load_ant_solution_log(file_path: Path, ants: int, generations: int) -> List[Dict]:
    """
    Load ant_solution_log.csv

    Returns:
        List of ant solutions (dictionary format)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # ヘッダーをスキップ
    if rows and rows[0][0] == "generation":
        rows = rows[1:]

    # シミュレーション単位にチャンク
    simulations = chunk_rows(rows, ants, generations)

    parsed: List[Dict] = []
    for sim_idx, sim_rows in enumerate(simulations):
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

            # ゴール未到達（bandwidth < 0）の場合はスキップ
            if bandwidth < 0:
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
    return parsed


def classify_bandwidth(bandwidth: float) -> int:
    """
    Classify bandwidth into 10-Mbps bins

    Args:
        bandwidth: Bandwidth value (Mbps)

    Returns:
        Bin index (0-10)
    """
    for idx, (min_bw, max_bw) in enumerate(BANDWIDTH_BINS):
        if min_bw <= bandwidth < max_bw:
            return idx
    # 100以上は最後の分類
    return len(BANDWIDTH_BINS) - 1


def classify_quality_ratio(quality_score: float) -> int:
    """
    Classify quality_score into 0.1 bins.

    Bins:
      [0.0,0.1), [0.1,0.2), ..., [0.8,0.9), [0.9,1.0]
    """
    if quality_score < 0:
        return -1
    if quality_score >= 1.0:
        return 9
    # left-closed right-open bins
    idx = int(quality_score * 10)
    return max(0, min(9, idx))


def aggregate_bandwidth_by_generation(
    parsed: List[Dict], generations: int, num_simulations: int, ants: int
) -> Dict[int, List[int]]:
    """
    Record the best bandwidth per generation per simulation, and aggregate by bandwidth bins

    Args:
        parsed: List of ant solutions
        generations: Number of generations
        num_simulations: Number of simulations
        ants: Number of ants per generation

    Returns:
        {generation: [bin0_count, bin1_count, ..., bin10_count], ...}
    """
    # 解析方針（absolute mode）:
    # - ant_solution_log.csv には「各世代×各アリ」の結果が入っている
    # - ただし本分析では、各世代・各シミュレーションについて「最良（最大帯域）」のアリ1つだけを代表値として採用する
    # - その代表値をビン分けし、シミュレーション数で集計して割合（%）として可視化する
    # Record best bandwidth for each generation and simulation
    # best_bandwidth[generation][simulation] = best_bandwidth
    best_bandwidth: Dict[int, Dict[int, float]] = defaultdict(dict)

    for entry in parsed:
        gen = entry["generation"]
        sim = entry["simulation"]
        bandwidth = entry["bandwidth"]

        # Record the best bandwidth for each generation and simulation
        if sim not in best_bandwidth[gen]:
            best_bandwidth[gen][sim] = bandwidth
        else:
            best_bandwidth[gen][sim] = max(best_bandwidth[gen][sim], bandwidth)

    # Count by generation and bandwidth bin
    # counts[generation][bin_index] = count
    counts: Dict[int, List[int]] = defaultdict(lambda: [0] * len(BANDWIDTH_BINS))

    for gen in range(generations):
        if gen in best_bandwidth:
            for sim in range(num_simulations):
                if sim in best_bandwidth[gen]:
                    best_bw = best_bandwidth[gen][sim]
                    bin_index = classify_bandwidth(best_bw)
                    counts[gen][bin_index] += 1

    return counts


def aggregate_quality_by_generation(
    parsed: List[Dict], generations: int, num_simulations: int
) -> Dict[int, List[int]]:
    """
    Record the best quality_score per generation per simulation, and aggregate by ratio bins.

    Returns:
        {generation: [bin0_count, ..., bin9_count], ...}
    """
    # 解析方針（relative mode / bandwidth_fluctuation 推奨）:
    # - 帯域変動があると、最適ボトルネック帯域（optimal_bottleneck）が世代ごとに変わる
    # - 絶対Mbpsではなく、相対指標 quality_score = found_bottleneck / optimal_bottleneck を用いる
    # - 各世代・各シミュレーションについて、quality_score が最大のアリ1つだけを代表値として採用する
    # - その代表値を [0.0,0.1),...,[0.9,1.0] に分類し、シミュレーション数で集計して割合（%）を描く
    best_quality: Dict[int, Dict[int, float]] = defaultdict(dict)

    for entry in parsed:
        gen = entry["generation"]
        sim = entry["simulation"]
        qs = entry.get("quality_score", -1.0)
        if qs is None or qs < 0:
            continue

        if sim not in best_quality[gen]:
            best_quality[gen][sim] = float(qs)
        else:
            best_quality[gen][sim] = max(best_quality[gen][sim], float(qs))

    counts: Dict[int, List[int]] = defaultdict(lambda: [0] * len(RELATIVE_BINS))

    for gen in range(generations):
        if gen in best_quality:
            for sim in range(num_simulations):
                if sim in best_quality[gen]:
                    best_qs = best_quality[gen][sim]
                    bin_index = classify_quality_ratio(best_qs)
                    if bin_index >= 0:
                        counts[gen][bin_index] += 1

    return counts


def calculate_percentages(counts: Dict[int, List[int]]) -> Dict[int, List[float]]:
    """
    Convert counts to percentages (%)

    Args:
        counts: {generation: [bin0_count, bin1_count, ...], ...}

    Returns:
        {generation: [bin0_percent, bin1_percent, ...], ...}
    """
    percentages: Dict[int, List[float]] = {}

    for gen, bin_counts in counts.items():
        total = sum(bin_counts)
        if total > 0:
            percentages[gen] = [count * 100.0 / total for count in bin_counts]
        else:
            percentages[gen] = [0.0] * len(bin_counts)

    return percentages


def plot_stacked_bar_chart(
    percentages: Dict[int, List[float]],
    output_path: Path,
    generations: int,
    *,
    bin_mode: str,
) -> None:
    """
    Plot stacked bar chart

    Args:
        percentages: {generation: [bin0_percent, bin1_percent, ...], ...}
        output_path: Output file path
        generations: Number of generations
    """
    # データを世代順にソート
    sorted_gens = sorted(percentages.keys())
    if not sorted_gens:
        print("⚠️ No data available. Cannot plot graph.")
        return

    # 各ビンのデータを抽出（積み上げ用）
    # - percentages[generation] は「その世代における代表値（最良解）が、各ビンに入った割合」
    # - 例えば simulations=100 の場合、各世代で最大100個の代表値（=各シミュレーションの最良1つ）が集計され、
    #   それを%に変換している
    n_bins = len(RELATIVE_BINS) if bin_mode == "relative" else len(BANDWIDTH_BINS)
    data_by_bin: List[List[float]] = [[] for _ in range(n_bins)]
    for gen in sorted_gens:
        for bin_idx, percent in enumerate(percentages[gen]):
            data_by_bin[bin_idx].append(percent)

    # グラフ描画
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    if bin_mode == "relative":
        bins: List[Tuple[float, float]] = RELATIVE_BINS
        colors = RELATIVE_COLORS
        legend_title = "Relative Bottleneck Ratio (Found / Optimal)"
    else:
        # absolute mode uses int/float bins; keep type hints separate to satisfy type checker
        bins_abs = BANDWIDTH_BINS
        colors = ABSOLUTE_COLORS
        legend_title = "Bottleneck Bandwidth"

    # 積み上げのために、上位ビン（高い値）を上に表示するため、逆順で処理
    bottom = [0.0] * len(sorted_gens)
    if bin_mode == "relative":
        for bin_idx in reversed(range(len(bins))):
            min_v, max_v = bins[bin_idx]
            # 表示は 0.0-0.1, ..., 0.9-1.0
            label = f"{min_v:.1f}–{max_v:.1f}"

            plt.bar(
                sorted_gens,
                data_by_bin[bin_idx],
                width=1.0,
                bottom=bottom,
                color=colors[bin_idx],
                label=label,
            )
            bottom = [b + d for b, d in zip(bottom, data_by_bin[bin_idx])]
    else:
        for bin_idx in reversed(range(len(bins_abs))):
            min_v, max_v = bins_abs[bin_idx]
            if max_v == float("inf"):
                label = f"{min_v}+ Mbps"
            else:
                label = f"{min_v}-{max_v} Mbps"

            plt.bar(
                sorted_gens,
                data_by_bin[bin_idx],
                width=1.0,
                bottom=bottom,
                color=colors[bin_idx],
                label=label,
            )
            bottom = [b + d for b, d in zip(bottom, data_by_bin[bin_idx])]

    # グラフの設定
    plt.ylim(0, 100)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Path Selection Ratio [%]", fontsize=AXIS_LABEL_FONTSIZE)

    # 凡例を取得して逆順に設定（100 Mbps が上に来るように）
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::-1],
        labels[::-1],
        title=legend_title,
        title_fontsize=12,
        fontsize=10,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        borderaxespad=0.3,
    )

    plt.gca().tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    # X軸の範囲を設定
    if sorted_gens:
        plt.xlim(sorted_gens[0] - 0.5, sorted_gens[-1] + 0.5)

    plt.tight_layout()

    # 両フォーマットで保存（compare_methods.py と同様）
    out_eps = output_path.with_suffix(".eps")
    out_svg = output_path.with_suffix(".svg")
    plt.savefig(str(out_eps), format="eps", dpi=300, bbox_inches="tight")
    plt.savefig(str(out_svg), format="svg", bbox_inches="tight")
    print(f"✅ Graph saved: {out_eps}")
    print(f"✅ Graph saved: {out_svg}")
    plt.close()


def find_csv_path(
    method: str, environment: str, opt_type: str, results_base: Path
) -> Path:
    """
    Generate CSV file path from method, environment, and optimization type

    Args:
        method: Method name (proposed, conventional, basic_aco_no_heuristic, etc.)
        environment: Environment name (static, manual, node_switching, bandwidth_fluctuation, etc.)
        opt_type: Optimization type (bandwidth_only, pareto, multi_objective, delay_constraint)
        results_base: Base path to results directory

    Returns:
        CSV file path
    """
    csv_path = results_base / method / environment / opt_type / "ant_solution_log.csv"
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Create stacked bar chart by bandwidth"
    )
    parser.add_argument(
        "--bin-mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help=(
            "Binning mode. relative=quality_score bins (recommended for bandwidth_fluctuation). "
            "absolute=Mbps bins."
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (if specified directly)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method name (proposed, conventional, basic_aco_no_heuristic, etc.)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Environment name (static, manual, node_switching, bandwidth_fluctuation, etc.)",
    )
    parser.add_argument(
        "--opt-type",
        type=str,
        default=None,
        help="Optimization type (bandwidth_only, pareto, multi_objective, delay_constraint)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        required=True,
        help="Number of generations",
    )
    parser.add_argument(
        "--ants",
        type=int,
        required=True,
        help="Number of ants per generation",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of simulations (default: 100)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="aco_moo_routing/results",
        help="Base path to results directory (default: aco_moo_routing/results)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: bandwidth_distribution.svg in the same directory as CSV)",
    )

    args = parser.parse_args()

    # CSVファイルのパスを決定
    if args.csv:
        csv_path = Path(args.csv)
    elif args.method and args.environment and args.opt_type:
        results_base = Path(args.results_dir)
        csv_path = find_csv_path(
            args.method, args.environment, args.opt_type, results_base
        )
    else:
        parser.error(
            "Please specify --csv or all of --method, --environment, --opt-type"
        )

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return

    # 出力ファイルパスを決定
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            csv_path.parent / "relative_bandwidth_distribution.svg"
            if args.bin_mode == "relative"
            else csv_path.parent / "bandwidth_distribution.svg"
        )

    print(f"📊 Loading CSV file: {csv_path}")
    print(
        f"📈 Generations: {args.generations}, Ants: {args.ants}, Simulations: {args.simulations}"
    )

    # CSVを読み込む
    parsed = load_ant_solution_log(csv_path, args.ants, args.generations)

    if not parsed:
        print("❌ No data available.")
        return

    print(f"✅ Loaded {len(parsed)} ant solutions.")

    # ここで「何を分布として描くか」を切り替える
    # - relative: quality_score（found/optimal）の分布（帯域変動環境に推奨）
    # - absolute: 絶対帯域(Mbps)の分布（最適帯域が固定に近い環境で有効）
    # どちらも「各世代・各シミュレーションで最良のアリ1つだけ」を代表値として採用する点が重要
    if args.bin_mode == "relative":
        # 相対品質で集計（各世代・各シミュレーションで最良の quality_score を記録）
        counts = aggregate_quality_by_generation(
            parsed, args.generations, args.simulations
        )
    else:
        # 帯域ごとに集計（各世代・各シミュレーションで最良の帯域を記録）
        counts = aggregate_bandwidth_by_generation(
            parsed, args.generations, args.simulations, args.ants
        )

    # 割合に変換
    percentages = calculate_percentages(counts)

    # グラフを描画
    plot_stacked_bar_chart(
        percentages, output_path, args.generations, bin_mode=args.bin_mode
    )

    print("✅ Done!")


if __name__ == "__main__":
    main()
