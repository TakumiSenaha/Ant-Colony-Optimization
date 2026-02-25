"""
遅延制約付き（提案手法）の Any optimal / Unique optimal 成功率を可視化

【データソース】
- ant_solution_log.csv の以下の列を使用:
  - Any optimal (is_optimal):        列5（制約を満たす最大帯域経路）
  - Unique optimal (is_unique_optimal): 列7（制約を満たす最大帯域かつ最小遅延経路）

【論文での説明】
- Any optimal:    遅延制約を満たす経路の中で、最大のボトルネック帯域を持つ経路への到達率
- Unique optimal: 上記の中で、さらに最小の遅延を持つ一意な経路（辞書式最適解）への到達率

【実行コマンド例】

# 遅延制約3段階（≤5ms, ≤10ms, ≤15ms）の比較グラフを生成
cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/aco_moo_routing

python analysis/compare_delay_constraint.py \
  --generations 1000 \
  --ants 10 \
  --constraints 5 10 15 \
  --output results/analysis/delay_constraint_comparison

# カスタムディレクトリ構造の場合
python analysis/compare_delay_constraint.py \
  --results-dir ./results \
  --generations 1000 \
  --ants 10 \
  --constraints 5 10 15 \
  --subdir-template "delay_constraint_{c}ms" \
  --opt-type delay_constraint \
  --output results/analysis/delay_constraint_comparison

【出力ファイル】
- delay_constraint_comparison.eps (論文用EPS形式)
- delay_constraint_comparison.svg (プレビュー用SVG形式)
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
) -> List[List[List[int]]]:
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

    simulations: List[List[List[int]]] = []
    for i in range(sims):
        simulations.append(int_rows[i * chunk_size : (i + 1) * chunk_size])
    return simulations


def success_rates(
    sim_rows: List[List[List[int]]], ants: int, generations: int, col: int
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
    """
    遅延制約付き環境の成功率を比較するグラフを生成

    【グラフの構成】
    - 実線: Unique optimal（辞書式最適解、最大帯域＋最小遅延）
    - 破線: Any optimal（最大帯域を持つ任意の経路）
    - 色: 遅延制約値（≤5ms=青、≤10ms=オレンジ、≤15ms=緑）

    【論文での解釈】
    - 厳しい制約（≤5ms）ほど収束が速い（解空間が狭いため）
    - 緩い制約（≤15ms）ではUnique optimalとAny optimalの差が大きい（複数の準最適解への分散）
    """
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # 論文向け配色（色覚多様性に配慮: Okabe-Ito パレット）
    constraint_colors = {
        5.0: "#0072B2",  # 青
        10.0: "#E69F00",  # オレンジ
        15.0: "#009E73",  # 緑
    }

    # マーカーを制約ごとに統一（Unique/Anyで同じマーカー）
    constraint_markers = {
        5.0: "o",  # 丸
        10.0: "s",  # 四角
        15.0: "^",  # 三角
    }

    # 制約の順序
    sorted_constraints = sorted(unique_optimal.keys())

    # Unique Optimal（実線）を先に描画
    for constraint in sorted_constraints:
        if constraint not in unique_optimal:
            continue

        vals = unique_optimal[constraint]
        x_values = list(range(len(vals)))
        color = constraint_colors.get(constraint, "#000000")
        marker = constraint_markers.get(constraint, "o")

        # 制約のラベル（絶対値で表示）
        label_suffix = f"≤{constraint:.0f}ms"

        plt.plot(
            x_values,
            vals,
            marker=marker,
            linestyle="-",  # 実線
            color=color,
            linewidth=2.0,
            markersize=4,
            markevery=50,  # 50世代ごとにマーカーを表示（見やすさ）
            label=f"Unique optimal ({label_suffix})",
            alpha=0.9,
        )

    # Any Optimal（破線）を描画
    for constraint in sorted_constraints:
        if constraint not in any_optimal:
            continue

        vals = any_optimal[constraint]
        x_values = list(range(len(vals)))
        color = constraint_colors.get(constraint, "#000000")
        marker = constraint_markers.get(constraint, "o")

        # 制約のラベル（絶対値で表示）
        label_suffix = f"≤{constraint:.0f}ms"

        plt.plot(
            x_values,
            vals,
            marker=marker,
            linestyle="--",  # 破線
            color=color,
            linewidth=2.0,
            markersize=4,
            markevery=50,  # 50世代ごとにマーカーを表示
            label=f"Any optimal ({label_suffix})",
            alpha=0.7,
        )

    # 軸範囲とラベル
    plt.ylim((0, 105))
    plt.xlim(left=0)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Optimal Path Selection Rate [%]", fontsize=AXIS_LABEL_FONTSIZE)

    # 凡例を整理
    plt.legend(
        fontsize=12,
        loc="best",
        frameon=True,
        ncol=1,
    )

    # 軸の装飾
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

    # EPS形式（論文投稿用）とSVG形式（プレビュー用）で保存
    out_eps = output_base.with_suffix(".eps")
    out_svg = output_base.with_suffix(".svg")
    plt.savefig(str(out_eps), format="eps", dpi=300)
    plt.savefig(str(out_svg), format="svg")

    print("\n✅ グラフを保存しました:")
    print(f"   📄 EPS: {out_eps}")
    print(f"   🖼️  SVG: {out_svg}")
    plt.close()


def main():
    """
    メイン関数

    【実験データの準備】
    1. config.yamlで以下を設定して実験を実行:
       - method: "proposed"
       - delay_constraint: enabled: true
       - max_delay: 5, 10, 15 の3段階で実験

    2. 結果は以下のディレクトリに保存される:
       results/proposed/delay_constraint_5ms/delay_constraint/ant_solution_log.csv
       results/proposed/delay_constraint_10ms/delay_constraint/ant_solution_log.csv
       results/proposed/delay_constraint_15ms/delay_constraint/ant_solution_log.csv

    3. このスクリプトを実行してグラフを生成:
       python analysis/compare_delay_constraint.py --generations 1000 --ants 10

    【生成されるグラフ】
    - 図5.X: 遅延制約環境下における最適解への収束率
    - Unique optimal (実線): 辞書式最適解（最大帯域＋最小遅延）
    - Any optimal (破線): 最大帯域を持つ任意の経路
    - ≤5ms (青): 厳しい制約 → 高速収束
    - ≤10ms (オレンジ): 中程度の制約
    - ≤15ms (緑): 緩い制約 → 準最適解への分散が見られる
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="遅延制約付き提案手法の Any/Unique 成功率を可視化（論文図5.X用）"
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
