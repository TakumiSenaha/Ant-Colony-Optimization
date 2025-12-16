"""
従来手法（Conventional）と提案手法（Proposed）を新ログ形式（ant_solution_log.csv）
で比較するスクリプト。環境と最適化タイプを指定して、任意の指標を可視化できます。

使い方（例）:
  # プロジェクトルートから
  python aco_moo_routing/analysis/compare_conventional_vs_proposed.py \
    --generations 1000 --ants 10 \
    --environments static \
    --opt-type bandwidth_only \
    --metric is_optimal

  # 出力: results/analysis/{env}/comparison_{env}_{metric}.eps と .svg

比較可能な指標（--metric オプション）:
  1. is_optimal: 最適解到達率 [%]
     - 各世代で、少なくとも1匹のアリが最適解を発見したシミュレーションの割合
     - 値の範囲: 0.0 ~ 100.0（100%が全てのシミュレーションで最適解を発見）
  
  2. is_unique_optimal: ユニーク最適解到達率 [%]
     - 各世代で、少なくとも1匹のアリがユニーク最適解（最良の最適解）を発見した
       シミュレーションの割合
     - 値の範囲: 0.0 ~ 100.0
  
  3. avg_quality: 平均品質スコア
     - 定義: (発見した経路のボトルネック帯域) / (最適解のボトルネック帯域)
     - 値の範囲: 0.0 ~ 1.0（1.0が最適解に一致）
     - 集計: 各世代において、全シミュレーションの全アリの品質スコアの平均値
     - 意味: 探索全体の品質を評価。1.0に近いほど、最適解に近い品質の解を
             発見できていることを示す
  
  4. max_quality: 最大品質スコアの平均
     - 定義: (発見した経路のボトルネック帯域) / (最適解のボトルネック帯域)
     - 値の範囲: 0.0 ~ 1.0（1.0が最適解に一致）
     - 集計: 各世代において、各シミュレーション内の全アリの品質スコアから
             最大値を取得し、全シミュレーションの最大値の平均を計算
     - 意味: 最良解発見能力を評価。avg_qualityと異なり、各シミュレーションで
             最も良い解のみを考慮するため、最良解発見の性能をより直接的に評価

品質スコアの詳細説明:
  品質スコアは、アリが発見した経路の品質を最適解に対する相対的な値で表現します。
  
  - 計算式: quality_score = found_bottleneck_bandwidth / optimal_bottleneck_bandwidth
  - 例: 最適解のボトルネック帯域が100Mbpsの場合
        * 100Mbpsの解を発見 → quality_score = 1.0（最適解）
        * 80Mbpsの解を発見 → quality_score = 0.8（最適解の80%の品質）
        * 50Mbpsの解を発見 → quality_score = 0.5（最適解の50%の品質）
  
  - avg_qualityとmax_qualityの違い:
    * avg_quality: 全アリの品質スコアの平均 → 探索全体の品質を評価
    * max_quality: 各シミュレーションの最良解の平均 → 最良解発見能力を評価
  
  - 論文での説明例:
    "品質スコアは、発見した経路のボトルネック帯域を最適解のボトルネック帯域で
    割った値であり、1.0に近いほど最適解に近い品質の解を発見できていることを
    示す。avg_qualityは探索全体の品質を、max_qualityは最良解発見能力を評価する
    指標として用いる。"
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
    """
    ant_solution_log.csv を読み込み、世代×アリのチャンクに分けた整数リストを返す。
    戻り値は simulations × (generations*ants) の2次元配列。
    """
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


def load_ant_solution_log_float(
    file_path: Path, ants: int, generations: int
) -> List[List[List[float]]]:
    """
    ant_solution_log.csv を読み込み、浮動小数点値として返す（品質スコア用）。
    戻り値は simulations × (generations*ants) の2次元配列。
    """
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

    float_rows: List[List[float]] = []
    for r in data_rows:
        try:
            float_rows.append([float(v) for v in r])
        except ValueError:
            float_rows.append([-1.0] * len(r))

    simulations: List[List[List[float]]] = []
    for i in range(sims):
        simulations.append(float_rows[i * chunk_size : (i + 1) * chunk_size])
    return simulations


def success_rates(
    sim_rows: List[List[List[int]]], ants: int, generations: int, col: int
) -> List[float]:
    """
    指定列（is_optimal=5, is_unique_optimal=7 など）を用いて
    世代ごとの成功率（少なくとも1匹が条件を満たす割合）を算出。
    """
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


def quality_scores_avg(
    sim_rows: List[List[List[float]]], ants: int, generations: int, col: int = 8
) -> List[float]:
    """
    品質スコア（quality_score=8）の世代ごとの平均値を算出。

    品質スコアの定義:
    - quality_score = (発見した経路のボトルネック帯域) / (最適解のボトルネック帯域)
    - 値の範囲: 0.0 ~ 1.0（1.0が最適解に一致）
    - 1.0に近いほど、最適解に近い品質の解を発見できていることを示す

    集計方法:
    - 各世代において、全シミュレーションの全アリの品質スコアを収集
    - それらの平均値を計算
    - 例: 世代gで100匹のアリがそれぞれ品質スコアを持っている場合、
          それら100個の値の平均を世代gの値とする

    意味:
    - 1.0 = 最適解を発見（または最適解と同等の品質）
    - 0.8 = 最適解の80%の品質（最適解のボトルネック帯域が100Mbpsなら、80Mbpsの解を発見）
    - 0.5 = 最適解の50%の品質
    - 世代が進むにつれて1.0に近づく = 学習が進み、より良い解を発見できるようになっている
    """
    scores: List[float] = []
    sims = len(sim_rows)
    for g in range(generations):
        all_scores: List[float] = []
        for sim in sim_rows:
            start = g * ants
            end = start + ants
            chunk = sim[start:end]
            for row in chunk:
                if len(row) > col and row[col] >= 0:
                    all_scores.append(row[col])
        if all_scores:
            scores.append(sum(all_scores) / len(all_scores))
        else:
            scores.append(0.0)
    return scores


def quality_scores_max(
    sim_rows: List[List[List[float]]], ants: int, generations: int, col: int = 8
) -> List[float]:
    """
    品質スコア（quality_score=8）の世代ごとの最大値の平均を算出。

    品質スコアの定義:
    - quality_score = (発見した経路のボトルネック帯域) / (最適解のボトルネック帯域)
    - 値の範囲: 0.0 ~ 1.0（1.0が最適解に一致）

    集計方法:
    - 各世代において、各シミュレーション内の全アリの品質スコアから最大値を取得
    - 全シミュレーションの最大値の平均を計算
    - 例: 世代gで10シミュレーション実行した場合、
          各シミュレーションで10匹のアリのうち最大の品質スコアを取得し、
          それら10個の最大値の平均を世代gの値とする

    意味:
    - avg_qualityとの違い:
      * avg_quality: 全アリの平均（探索の全体的な品質を評価）
      * max_quality: 各シミュレーションの最良解の平均（最良解発見能力を評価）
    - 1.0に近い = 各シミュレーションで最適解に近い解を発見できている
    - 世代が進むにつれて1.0に近づく = 最良解発見能力が向上している
    """
    scores: List[float] = []
    sims = len(sim_rows)
    for g in range(generations):
        max_scores_per_sim: List[float] = []
        for sim in sim_rows:
            start = g * ants
            end = start + ants
            chunk = sim[start:end]
            sim_scores = [row[col] for row in chunk if len(row) > col and row[col] >= 0]
            if sim_scores:
                max_scores_per_sim.append(max(sim_scores))
        if max_scores_per_sim:
            scores.append(sum(max_scores_per_sim) / len(max_scores_per_sim))
        else:
            scores.append(0.0)
    return scores


def plot_series(
    series: Dict[str, List[float]],
    ylabel: str,
    output_base: Path,
    _title: str = "",
    y_max: float = 105.0,
):
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    markers = ["o", "s", "^", "D", "v", "<", ">"]
    # 提案手法を黒、既存手法をグレーに
    color_map = {
        "Proposed": "black",
        "ACS": "darkgray",  # Ant Colony System
        "Conventional": "darkgray",  # 後方互換性のため
    }

    for idx, (label, values) in enumerate(series.items()):
        x_values = list(range(len(values)))
        # ラベルに応じて色を決定（デフォルトはグレー）
        color = color_map.get(label, "gray")
        plt.plot(
            x_values,
            values,
            marker=markers[idx % len(markers)],
            linestyle="-",
            color=color,
            linewidth=2.0,
            markersize=4,
            label=label,
        )

    plt.ylim((0, y_max))
    plt.xlim(left=0)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    plt.legend(fontsize=14, loc="best", frameon=True, ncol=1)

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
    # 両フォーマットで保存
    out_eps = output_base.with_suffix(".eps")
    out_svg = output_base.with_suffix(".svg")
    plt.savefig(str(out_eps), format="eps", dpi=300)
    plt.savefig(str(out_svg), format="svg")
    print(f"✅ グラフを保存しました: {out_eps}")
    print(f"✅ グラフを保存しました: {out_svg}")
    plt.show()  # プレビュー表示
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="conventional vs proposed を ant_solution_log.csv で比較"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="results ディレクトリ（指定がなければスクリプトの親）",
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=[
            "static",
            "node_switching",
            "bandwidth_fluctuation",
            "delay_constraint",
        ],
        help="環境ディレクトリ名リスト",
    )
    parser.add_argument(
        "--opt-type",
        type=str,
        default="bandwidth_only",
        help="最適化タイプサブディレクトリ (bandwidth_only / delay_constraint / pareto など)",
    )
    parser.add_argument("--ants", type=int, default=10, help="1世代あたりのアリ数")
    parser.add_argument("--generations", type=int, required=True, help="世代数")
    parser.add_argument(
        "--metric",
        type=str,
        default="is_optimal",
        choices=["is_optimal", "is_unique_optimal", "avg_quality", "max_quality"],
        help="比較指標: is_optimal, is_unique_optimal, avg_quality, max_quality",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力先（未指定なら環境ごとに results/analysis 配下へ）",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    default_results = script_dir.parent / "results"
    results_root = Path(args.results_dir) if args.results_dir else default_results

    for env in args.environments:
        series: Dict[str, List[float]] = {}
        for method, label in [
            ("conventional", "ACS"),  # Ant Colony System
            ("proposed", "Proposed"),
        ]:
            path = results_root / method / env / args.opt_type / "ant_solution_log.csv"
            if not path.exists():
                print(f"⚠️ スキップ: {path} がありません。")
                continue
            try:
                if args.metric in ["avg_quality", "max_quality"]:
                    # 品質スコアの場合は浮動小数点値で読み込む
                    sims_float = load_ant_solution_log_float(
                        path, args.ants, args.generations
                    )
                    if args.metric == "avg_quality":
                        values = quality_scores_avg(
                            sims_float, args.ants, args.generations, col=8
                        )
                    else:  # max_quality
                        values = quality_scores_max(
                            sims_float, args.ants, args.generations, col=8
                        )
                else:
                    # is_optimal または is_unique_optimal
                    col_idx = 5 if args.metric == "is_optimal" else 7
                    sims = load_ant_solution_log(path, args.ants, args.generations)
                    values = success_rates(sims, args.ants, args.generations, col_idx)
            except ValueError as e:
                print(f"⚠️ {e}")
                continue
            series[label] = values

        if not series:
            print(f"⚠️ 環境 {env} で有効なデータがありません。")
            continue

        out_dir = (
            Path(args.output_dir)
            if args.output_dir
            else results_root / "analysis" / env
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path_base = out_dir / f"comparison_{env}_{args.metric}"
        title = f"{env} ({args.metric})"
        if args.metric == "is_optimal":
            ylabel = "Optimal Path Selection Ratio [%]"
            y_max = 105.0
        elif args.metric == "is_unique_optimal":
            ylabel = "Unique Optimal Selection Ratio [%]"
            y_max = 105.0
        elif args.metric == "avg_quality":
            ylabel = "Derived Bottleneck / Optimal Bottleneck"
            y_max = 1.05
        else:  # max_quality
            ylabel = "Derived Bottleneck / Optimal Bottleneck"
            y_max = 1.05
        plot_series(series, ylabel, out_path_base, title, y_max=y_max)


if __name__ == "__main__":
    main()
