"""
遅延制約付き（提案手法）の any / unique 成功率を ant_solution_log.csv から可視化するスクリプト。
- Any Optimal  : is_optimal 列 (col=5)
- Unique Optimal: is_unique_optimal 列 (col=7)

使い方（例）:
  # プロジェクトルートから
  python aco_moo_routing/analysis/compare_delay_constraint.py \
    --generations 1000 --ants 10 \
    --constraints 5 10 15 \
    --subdir-template "delay_constraint_{c}ms" \
    --opt-type delay_constraint

  # 出力: analysis/delay_constraint_comparison.eps と .svg
"""

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# グラフ描画設定
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 18
FIGURE_WIDTH = 10  # グラフの横幅（論文形式で統一）
FIGURE_HEIGHT = 7  # グラフの縦幅（論文形式で統一）


def load_ant_solution_log(
    file_path: Path, ants: int, generations: int
) -> List[List[int]]:
    rows: List[List[str]] = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            if r:
                rows.append(r)

    if not rows:
        raise ValueError(f"CSVが空です: {file_path}")

    # ヘッダ行をスキップ（先頭が generation ならヘッダとみなす）
    data_rows = rows[1:] if rows and rows[0][0].lower() == "generation" else rows

    chunk_size = ants * generations
    if len(data_rows) % chunk_size != 0:
        print(
            f"⚠️ 行数が期待と合いません: {len(data_rows)} 行 "
            f"(期待: {chunk_size} の倍数)"
        )
    sims = len(data_rows) // chunk_size
    sims = max(sims, 1)

    int_rows: List[List[int]] = []
    for r in data_rows:
        try:
            int_rows.append([int(float(v)) for v in r])
        except ValueError:
            int_rows.append([-1] * len(r))

    simulations: List[List[int]] = []
    for i in range(sims):
        simulations.append(int_rows[i * chunk_size : (i + 1) * chunk_size])
    return simulations


def success_rates(
    sim_rows: List[List[int]], ants: int, generations: int, col: int
) -> List[float]:
    rates: List[float] = []
    sims = len(sim_rows)
    for g in range(generations):
        success = 0
        for sim in sim_rows:
            start = g * ants
            end = start + ants
            chunk = sim[start:end]
            if any(val >= 1 for val in (row[col] for row in chunk)):
                success += 1
        denom = sims if sims > 0 else 1
        rates.append(100 * success / denom)
    return rates


def plot_delay_constraint_comparison(
    unique_optimal: Dict[float, List[float]],
    any_optimal: Dict[float, List[float]],
    output_base: Path,
):
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    colors = ["#000000", "#4A4A4A", "#808080", "#A0A0A0"]
    markers = ["o", "s", "^", "D"]

    # Unique
    for i, (constraint, vals) in enumerate(sorted(unique_optimal.items())):
        x_values = list(range(len(vals)))
        plt.plot(
            x_values,
            vals,
            marker=markers[i % len(markers)],
            linestyle="-",
            color=colors[i % len(colors)],
            linewidth=2.5,
            markersize=5,
            label=f"Unique Optimal (≤{constraint:.0f}ms)",
        )

    # Any
    for i, (constraint, vals) in enumerate(sorted(any_optimal.items())):
        x_values = list(range(len(vals)))
        plt.plot(
            x_values,
            vals,
            marker=markers[i % len(markers)],
            linestyle="--",
            color=colors[i % len(colors)],
            linewidth=2.0,
            markersize=4,
            label=f"Any Optimal (≤{constraint:.0f}ms)",
        )

    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Optimal Path Selection Ratio [%]", fontsize=AXIS_LABEL_FONTSIZE)
    plt.legend(fontsize=14, loc="best", frameon=True, ncol=2)

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
    out_eps = output_base.with_suffix(".eps")
    out_svg = output_base.with_suffix(".svg")
    plt.savefig(str(out_eps), format="eps", dpi=300)
    plt.savefig(str(out_svg), format="svg")
    print(f"✅ グラフを保存しました: {out_eps}")
    print(f"✅ グラフを保存しました: {out_svg}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="遅延制約付き提案手法の any / unique 成功率を可視化（ant_solution_log対応）"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="results ディレクトリ（未指定なら aco_moo_routing/results）",
    )
    parser.add_argument(
        "--constraints",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 15.0],
        help="遅延制約(ms)のリスト",
    )
    parser.add_argument(
        "--subdir-template",
        type=str,
        default="delay_constraint_{c}ms",
        help="制約ごとのサブディレクトリ名テンプレート（{c} が数値に置換される）",
    )
    parser.add_argument(
        "--opt-type",
        type=str,
        default="delay_constraint",
        help="最適化タイプサブディレクトリ (delay_constraint など)",
    )
    parser.add_argument("--ants", type=int, default=10, help="1世代あたりのアリ数")
    parser.add_argument("--generations", type=int, required=True, help="世代数")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力ファイルパス（未指定なら analysis/ に delay_constraint_comparison.{fmt}）",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    default_results = script_dir.parent / "results"
    results_root = Path(args.results_dir) if args.results_dir else default_results

    unique_optimal_data: Dict[float, List[float]] = {}
    any_optimal_data: Dict[float, List[float]] = {}

    for constraint in args.constraints:
        subdir = args.subdir_template.format(c=int(constraint))
        csv_path = (
            results_root / "proposed" / subdir / args.opt_type / "ant_solution_log.csv"
        )
        if not csv_path.exists():
            print(f"⚠️ スキップ: {csv_path} がありません。")
            continue

        try:
            sim_rows = load_ant_solution_log(csv_path, args.ants, args.generations)
        except ValueError as e:
            print(f"⚠️ {e}")
            continue

        any_rates = success_rates(sim_rows, args.ants, args.generations, col=5)
        unique_rates = success_rates(sim_rows, args.ants, args.generations, col=7)
        any_optimal_data[constraint] = any_rates
        unique_optimal_data[constraint] = unique_rates

    if not any_optimal_data and not unique_optimal_data:
        print("⚠️ 有効なデータがありません。")
        return

    output_base = (
        Path(args.output) if args.output else script_dir / "delay_constraint_comparison"
    )
    plot_delay_constraint_comparison(unique_optimal_data, any_optimal_data, output_base)
    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
