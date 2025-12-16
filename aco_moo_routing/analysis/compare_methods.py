"""
4手法（Basic ACO w/o Heuristic, Basic ACO w/ Heuristic, Previous Method, Proposed Method）
を任意に選択して比較するスクリプト。

複数の手法と複数の環境を同時に描画できます。

【使用例】
プロジェクトルートから実行してください。

# 1. 提案手法のみを描画
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods proposed \
  --environments static \
  --opt-type bandwidth_only \
  --metric is_optimal

# 2. 4手法すべてを比較
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods basic_aco_no_heuristic basic_aco_with_heuristic previous proposed \
  --environments static \
  --opt-type bandwidth_only \
  --metric is_optimal

# 3. 複数環境を同時に描画
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods proposed \
  --environments static node_switching bandwidth_fluctuation \
  --opt-type bandwidth_only \
  --metric is_optimal

# 4. 複数手法×複数環境
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods previous proposed \
  --environments static bandwidth_fluctuation \
  --opt-type bandwidth_only \
  --metric is_optimal

# 5. 品質スコア（avg_quality）で比較
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods basic_aco_no_heuristic basic_aco_with_heuristic previous proposed \
  --environments static \
  --opt-type bandwidth_only \
  --metric avg_quality

# 6. 手動設定トポロジ環境（manual）で比較
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods basic_aco_no_heuristic basic_aco_with_heuristic previous proposed \
  --environments manual \
  --opt-type bandwidth_only \
  --metric is_optimal

【比較可能な指標（--metric オプション）】
  1. is_optimal: 最適解到達率 [%]
     - 各世代で、少なくとも1匹のアリが最適解を発見したシミュレーションの割合
  
  2. is_unique_optimal: ユニーク最適解到達率 [%]
     - 各世代で、少なくとも1匹のアリがユニーク最適解を発見したシミュレーションの割合
  
  3. avg_quality: 平均品質スコア (0.0 ~ 1.0)
     - 全アリの品質スコアの平均値
     - 探索全体の品質を評価
  
  4. max_quality: 最大品質スコアの平均 (0.0 ~ 1.0)
     - 各シミュレーションの最良解の平均値
     - 最良解発見能力を評価

【手法名のマッピング】
  - basic_aco_no_heuristic: Basic ACO w/o Heuristic (β=0)
  - basic_aco_with_heuristic: Basic ACO w/ Heuristic (β=1)
  - previous: Previous Method (Edge-based learning)
  - proposed: Proposed Method (Node-based learning)

【環境名】
  - manual: 手動設定トポロジ（最適経路を100Mbpsに設定）
  - static: 静的ランダムグラフ
  - node_switching: コンテンツ要求ノード変動
  - bandwidth_fluctuation: 帯域変動

【出力】
  - ファイル名: comparison_{methods}_{environments}_{metric}.eps と .svg
  - 保存先: results/analysis/ ディレクトリ（デフォルト）
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt

# グラフ描画設定
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 18
FIGURE_WIDTH = 10  # グラフの横幅（論文形式で統一）
FIGURE_HEIGHT = 7  # グラフの縦幅（論文形式で統一）

# 手法名のマッピング（ディレクトリ名 → 表示名）
METHOD_LABELS = {
    "basic_aco_no_heuristic": "Basic ACO w/o Heuristic",
    "basic_aco_with_heuristic": "Basic ACO w/ Heuristic",
    "previous": "Previous Method",
    "proposed": "Proposed Method",
    "conventional": "Conventional",  # 後方互換性のため
}

# 色のマッピング（モノクロ、提案手法が一番黒、それ以外は段階的に灰色）
# 順序: proposed (黒) > previous (濃いグレー) > basic_aco_with_heuristic (中グレー) > basic_aco_no_heuristic (薄いグレー)
METHOD_COLORS = {
    "proposed": "black",  # 一番黒（提案手法）
    "previous": "dimgray",  # 濃いグレー（先行研究）
    "basic_aco_with_heuristic": "gray",  # 中程度のグレー（基本ACO w/ ヒューリスティック）
    "basic_aco_no_heuristic": "lightgray",  # 薄いグレー（基本ACO w/o ヒューリスティック）
    "conventional": "darkgray",  # 従来手法（後方互換性のため）
}

# 線スタイルのマッピング（環境ごと）
ENV_LINESTYLES = {
    "manual": "-",
    "static": "-",
    "node_switching": "--",
    "bandwidth_fluctuation": "-.",
    "delay_constraint": ":",
}

# マーカーのリスト
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


def load_ant_solution_log(
    file_path: Path, ants: int, generations: int
) -> List[List[List[int]]]:
    """
    ant_solution_log.csv を読み込み、世代×アリのチャンクに分けた整数リストを返す。
    """
    rows: List[List[str]] = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            if r:
                rows.append(r)

    if not rows:
        raise ValueError(f"CSVが空です: {file_path}")

    # ヘッダ行をスキップ
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
    """
    rows: List[List[str]] = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            if r:
                rows.append(r)

    if not rows:
        raise ValueError(f"CSVが空です: {file_path}")

    # ヘッダ行をスキップ
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
    """
    scores: List[float] = []
    sims = len(sim_rows)
    for g in range(generations):
        max_scores_per_sim: List[float] = []
        for sim in sim_rows:
            start = g * ants
            end = start + ants
            chunk = sim[start:end]
            sim_scores = [
                row[col] for row in chunk if len(row) > col and row[col] >= 0
            ]
            if sim_scores:
                max_scores_per_sim.append(max(sim_scores))
        if max_scores_per_sim:
            scores.append(sum(max_scores_per_sim) / len(max_scores_per_sim))
        else:
            scores.append(0.0)
    return scores


def plot_series(
    series: Dict[str, Tuple[List[float], str, str]],
    ylabel: str,
    output_base: Path,
    y_max: float = 105.0,
):
    """
    複数の系列を描画する。

    series のキーはラベル、値は (values, method, env) のタプル。
    """
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    marker_idx = 0
    for label, data_tuple in series.items():
        values, method, env = data_tuple
        x_values = list(range(len(values)))

        # 色と線スタイルを決定
        color = METHOD_COLORS.get(method, "gray")
        linestyle = ENV_LINESTYLES.get(env, "-")
        marker = MARKERS[marker_idx % len(MARKERS)]
        marker_idx += 1

        plt.plot(
            x_values,
            values,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=2.0,
            markersize=4,
            label=label,
        )

    plt.ylim((0, y_max))
    plt.xlim(left=0)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    plt.legend(fontsize=12, loc="best", frameon=True, ncol=1)

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
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="4手法を任意に選択して比較するスクリプト"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="results ディレクトリ（指定がなければスクリプトの親）",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=[
            "basic_aco_no_heuristic",
            "basic_aco_with_heuristic",
            "previous",
            "proposed",
            "conventional",
        ],
        help="比較する手法（複数指定可能）",
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        required=True,
        help="環境ディレクトリ名リスト（複数指定可能）",
    )
    parser.add_argument(
        "--opt-type",
        type=str,
        default="bandwidth_only",
        help="最適化タイプサブディレクトリ",
    )
    parser.add_argument("--ants", type=int, default=10, help="1世代あたりのアリ数")
    parser.add_argument("--generations", type=int, required=True, help="世代数")
    parser.add_argument(
        "--metric",
        type=str,
        default="is_optimal",
        choices=["is_optimal", "is_unique_optimal", "avg_quality", "max_quality"],
        help="比較指標",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力先（未指定なら results/analysis 配下へ）",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    default_results = script_dir.parent / "results"
    results_root = Path(args.results_dir) if args.results_dir else default_results

    # 系列データを収集（ラベル → (values, method, env) のタプル）
    series: Dict[str, Tuple[List[float], str, str]] = {}

    for method in args.methods:
        for env in args.environments:
            path = results_root / method / env / args.opt_type / "ant_solution_log.csv"
            if not path.exists():
                print(f"⚠️ スキップ: {path} がありません。")
                continue

            # ラベルを生成（手法名 + 環境名）
            method_label = METHOD_LABELS.get(method, method)
            if len(args.environments) > 1 or len(args.methods) > 1:
                # 複数環境または複数手法の場合は環境名も含める
                label = f"{method_label} ({env})"
            else:
                # 単一環境かつ単一手法の場合は手法名のみ
                label = method_label

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

            series[label] = (values, method, env)

    if not series:
        print("⚠️ 有効なデータがありません。")
        return

    # 出力ディレクトリとファイル名を決定
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_root / "analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ファイル名を生成（手法名と環境名を含める）
    methods_str = "_".join(args.methods)
    envs_str = "_".join(args.environments)
    filename = f"comparison_{methods_str}_{envs_str}_{args.metric}"
    out_path_base = out_dir / filename

    # Y軸ラベルと最大値を決定
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

    plot_series(series, ylabel, out_path_base, y_max=y_max)


if __name__ == "__main__":
    main()

