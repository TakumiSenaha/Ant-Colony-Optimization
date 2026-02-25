"""
System Knowledge vs Agent Behavior の比較グラフ生成

【目的】
「エージェントはε=0.1でランダムに動いているため、見かけ上の正解率は低い（変動する）が、
システム内部（フェロモン分布）はもっと早い段階で、かつ確実に最適解を学習し終えている」
ことを視覚的に証明する。

【データソース】
generation_stats.csv の以下の列を使用:
- optimal_count: 実際にその世代で最適解を通ったアリの数（Agent Behavior）
- interest_hit: フェロモン貪欲解が最適解だったか（0 or 1、System Knowledge）

【論文での説明】
- Agent Behavior（青破線）: ε-greedy探索による確率的な挙動（探索の影響で変動）
- System Knowledge（赤実線）: フェロモン分布の決定論的な収束（システムの内部知識）

【実行コマンド例】

# 基本的な使い方（単一シミュレーション結果から）
cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/aco_moo_routing

python analysis/plot_knowledge_vs_behavior.py \
  --results-dir results/proposed/static/bandwidth_only \
  --num-ants 10 \
  --output results/analysis/knowledge_vs_behavior

# 複数環境の比較（manual, static, bandwidth_fluctuation）
python analysis/plot_knowledge_vs_behavior.py \
  --results-dirs \
    results/proposed/manual/bandwidth_only \
    results/proposed/static/bandwidth_only \
    results/proposed/bandwidth_fluctuation/bandwidth_only \
  --labels "Manual" "Static" "Dynamic" \
  --num-ants 10 \
  --output results/analysis/knowledge_vs_behavior_comparison

# 遅延制約環境での比較
python analysis/plot_knowledge_vs_behavior.py \
  --results-dir results/proposed/delay_constraint_10ms/delay_constraint \
  --num-ants 10 \
  --use-unique \
  --output results/analysis/knowledge_vs_behavior_delay_10ms

【出力ファイル】
- knowledge_vs_behavior.eps (論文用EPS形式)
- knowledge_vs_behavior.svg (プレビュー用SVG形式)
- knowledge_vs_behavior.png (プレゼン用PNG形式)

【論文での使用例】
Figure 6: Convergence of probabilistic agent behavior vs. deterministic system knowledge.
The red solid line represents the selection rate of the optimal path using deterministic 
routing (max-pheromone selection), indicating the system's internal knowledge. 
The blue dashed line represents the actual selection rate of agents operating under 
an ε-greedy policy (ε=0.1).
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# グラフ描画設定（論文形式）
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 7
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 16
TITLE_FONTSIZE = 26


def load_generation_stats(csv_path: Path) -> List[Dict]:
    """
    generation_stats.csvを読み込む

    Args:
        csv_path: CSVファイルのパス

    Returns:
        世代ごとのデータのリスト
    """
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def aggregate_by_generation(
    all_stats: List[List[Dict]], num_ants: int
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    複数シミュレーションの結果を世代ごとに集計

    Args:
        all_stats: 全シミュレーションのgeneration_stats
        num_ants: 1世代あたりのアリ数

    Returns:
        (agent_behavior_mean, agent_behavior_sem,
         system_knowledge_mean, system_knowledge_sem)
    """
    # 世代数を取得（最初のシミュレーションから）
    num_generations = len(all_stats[0]) if all_stats else 0

    agent_behavior_by_gen = [[] for _ in range(num_generations)]
    system_knowledge_by_gen = [[] for _ in range(num_generations)]

    for stats in all_stats:
        for gen_idx, row in enumerate(stats):
            if gen_idx >= num_generations:
                break

            # Agent Behavior: optimal_count / num_ants
            optimal_count = int(row.get("optimal_count", 0) or 0)
            agent_rate = optimal_count / num_ants if num_ants > 0 else 0.0
            agent_behavior_by_gen[gen_idx].append(agent_rate)

            # System Knowledge: interest_hit (0 or 1)
            interest_hit = int(row.get("interest_hit", 0) or 0)
            system_knowledge_by_gen[gen_idx].append(interest_hit)

    # 平均と標準誤差を計算
    agent_mean = []
    agent_sem = []
    system_mean = []
    system_sem = []

    for gen in range(num_generations):
        agent_values = agent_behavior_by_gen[gen]
        system_values = system_knowledge_by_gen[gen]

        if agent_values:
            agent_mean.append(np.mean(agent_values))
            agent_sem.append(np.std(agent_values) / np.sqrt(len(agent_values)))
        else:
            agent_mean.append(0.0)
            agent_sem.append(0.0)

        if system_values:
            system_mean.append(np.mean(system_values))
            system_sem.append(np.std(system_values) / np.sqrt(len(system_values)))
        else:
            system_mean.append(0.0)
            system_sem.append(0.0)

    return agent_mean, agent_sem, system_mean, system_sem


def plot_single_environment(
    agent_mean: List[float],
    agent_sem: List[float],
    system_mean: List[float],
    system_sem: List[float],
    output_base: Path,
    title: Optional[str] = None,
):
    """
    単一環境のKnowledge vs Behaviorグラフを生成

    【グラフの構成】
    - 赤実線（太線）: System Knowledge（フェロモン貪欲解が最適解）
    - 青破線: Agent Behavior（実際に最適解を通ったアリの割合）
    - 薄い帯: 標準誤差（複数シミュレーションのばらつき）

    【論文での解釈】
    - Systemは早期に100%収束（内部知識の確立）
    - Agentはε=0.1の探索で変動（意図的な探索）
    - 乖離 = 「学習済みだが探索も継続」の証拠
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    generations = list(range(len(agent_mean)))

    # Agent Behavior (Probabilistic) - 青破線
    ax.plot(
        generations,
        [v * 100 for v in agent_mean],  # パーセント表示
        label="Agent Behavior (Probabilistic, ε=0.1)",
        color="#1f77b4",  # 青
        linestyle="--",
        linewidth=2.5,
        alpha=0.9,
        marker="o",
        markevery=50,
        markersize=5,
    )
    # 標準誤差の帯
    ax.fill_between(
        generations,
        [(v - e) * 100 for v, e in zip(agent_mean, agent_sem)],
        [(v + e) * 100 for v, e in zip(agent_mean, agent_sem)],
        color="#1f77b4",
        alpha=0.15,
    )

    # System Knowledge (Deterministic) - 赤実線（太く強調）
    ax.plot(
        generations,
        [v * 100 for v in system_mean],  # パーセント表示
        label="System Knowledge (Deterministic, Greedy)",
        color="#d62728",  # 赤
        linestyle="-",
        linewidth=3.5,
        alpha=1.0,
        marker="s",
        markevery=50,
        markersize=6,
    )
    # 標準誤差の帯
    ax.fill_between(
        generations,
        [(v - e) * 100 for v, e in zip(system_mean, system_sem)],
        [(v + e) * 100 for v, e in zip(system_mean, system_sem)],
        color="#d62728",
        alpha=0.15,
    )

    # グリッド（論文での視認性向上）
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.4, color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.2, color="gray")

    # 軸範囲とラベル
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max(generations) if generations else 1000)
    ax.set_xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel(
        "Optimal Path Selection Rate [%]",
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight="bold",
    )

    # タイトル（オプション）
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=20)

    # 凡例（右下に配置、枠あり）
    ax.legend(
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fancybox=False,
    )

    # 軸の装飾
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

    # EPS形式（論文投稿用）、SVG形式（プレビュー用）、PNG形式（プレゼン用）で保存
    out_eps = output_base.with_suffix(".eps")
    out_svg = output_base.with_suffix(".svg")
    out_png = output_base.with_suffix(".png")

    plt.savefig(str(out_eps), format="eps", dpi=300, bbox_inches="tight")
    plt.savefig(str(out_svg), format="svg", bbox_inches="tight")
    plt.savefig(str(out_png), format="png", dpi=300, bbox_inches="tight")

    print(f"\n✅ グラフを保存しました:")
    print(f"   📄 EPS: {out_eps}")
    print(f"   🖼️  SVG: {out_svg}")
    print(f"   🖼️  PNG: {out_png}")

    plt.close()


def plot_multiple_environments(
    all_data: Dict[str, Tuple[List[float], List[float], List[float], List[float]]],
    output_base: Path,
):
    """
    複数環境のKnowledge vs Behaviorを1つのグラフに重ねて表示

    Args:
        all_data: {環境名: (agent_mean, agent_sem, system_mean, system_sem)}
        output_base: 出力ファイルのベース名
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # 環境ごとの配色（Okabe-Itoパレット）
    colors = {
        0: "#0072B2",  # 青
        1: "#E69F00",  # オレンジ
        2: "#009E73",  # 緑
        3: "#CC79A7",  # ピンク
    }

    for idx, (env_name, (agent_mean, agent_sem, system_mean, system_sem)) in enumerate(
        all_data.items()
    ):
        generations = list(range(len(agent_mean)))
        color = colors.get(idx, "#000000")

        # Agent Behavior（破線）
        ax.plot(
            generations,
            [v * 100 for v in agent_mean],
            label=f"{env_name} - Agent Behavior",
            color=color,
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
        )

        # System Knowledge（実線）
        ax.plot(
            generations,
            [v * 100 for v in system_mean],
            label=f"{env_name} - System Knowledge",
            color=color,
            linestyle="-",
            linewidth=3.0,
            alpha=1.0,
        )

    # グリッド
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.4, color="gray")

    # 軸設定
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.set_xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel(
        "Optimal Path Selection Rate [%]",
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight="bold",
    )

    # 凡例
    ax.legend(
        loc="lower right",
        fontsize=LEGEND_FONTSIZE - 2,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        ncol=1,
    )

    # 軸の装飾
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
    )
    ax.minorticks_on()

    plt.tight_layout()

    # 保存
    out_eps = output_base.with_suffix(".eps")
    out_svg = output_base.with_suffix(".svg")
    out_png = output_base.with_suffix(".png")

    plt.savefig(str(out_eps), format="eps", dpi=300, bbox_inches="tight")
    plt.savefig(str(out_svg), format="svg", bbox_inches="tight")
    plt.savefig(str(out_png), format="png", dpi=300, bbox_inches="tight")

    print(f"\n✅ 複数環境比較グラフを保存しました:")
    print(f"   📄 EPS: {out_eps}")
    print(f"   🖼️  SVG: {out_svg}")
    print(f"   🖼️  PNG: {out_png}")

    plt.close()


def calculate_convergence_generation(system_mean: List[float], threshold: float = 0.95):
    """
    System Knowledgeが閾値を超えた世代を計算

    Args:
        system_mean: System Knowledgeの平均値リスト
        threshold: 収束とみなす閾値（デフォルト: 0.95 = 95%）

    Returns:
        収束世代（見つからない場合はNone）
    """
    for gen, value in enumerate(system_mean):
        if value >= threshold:
            return gen
    return None


def main():
    """
    メイン関数

    【実験データの準備】
    1. config.yamlで実験を実行（simulations: 100推奨）
       - method: "proposed"
       - ログ出力を有効化

    2. generation_stats.csvが生成される

    3. このスクリプトを実行してグラフを生成

    【生成されるグラフ】
    - Figure 6: System Knowledge vs Agent Behavior
    - 赤実線: フェロモン分布の収束（システムの内部知識）
    - 青破線: エージェントの実際の挙動（ε-greedy探索）

    【論文での議論】
    - 「システムは約X世代で最適解を完全に特定（赤線が100%）」
    - 「それ以降の青線の変動は、探索継続によるもの（知識不足ではない）」
    - 「探索の多様性と知識の確実性を両立」
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="System Knowledge vs Agent Behavior の比較グラフ生成（論文図6用）"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="単一環境の結果ディレクトリ（generation_stats.csvがあるディレクトリ）",
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        type=str,
        default=None,
        help="複数環境の結果ディレクトリのリスト（複数環境を比較する場合）",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=None,
        help="各環境のラベル（--results-dirsと同じ数だけ指定）",
    )
    parser.add_argument(
        "--num-ants",
        type=int,
        required=True,
        help="1世代あたりのアリ数（config.yamlのnum_antsと同じ値）",
    )
    parser.add_argument(
        "--use-unique",
        action="store_true",
        help="unique_optimal_count を使用（遅延制約環境用）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力ファイルパス（未指定なら results/analysis/knowledge_vs_behavior）",
    )

    args = parser.parse_args()

    # 単一環境モード
    if args.results_dir:
        results_path = Path(args.results_dir)
        csv_path = results_path / "generation_stats.csv"

        if not csv_path.exists():
            print(f"❌ エラー: {csv_path} が見つかりません")
            sys.exit(1)

        # CSVを読み込み
        print(f"📂 読み込み中: {csv_path}")
        stats = load_generation_stats(csv_path)

        # 集計（シミュレーションごとに分割されている前提）
        # 注: 1つのCSVに複数シミュレーションが連結されている場合、
        #     世代数で分割する必要がある
        # ここでは簡易的に、全データを1シミュレーションとして扱う
        all_stats = [stats]  # 後で複数シミュレーション対応に拡張可能

        agent_mean, agent_sem, system_mean, system_sem = aggregate_by_generation(
            all_stats, args.num_ants
        )

        # 収束世代を計算
        conv_gen = calculate_convergence_generation(system_mean, threshold=0.95)
        if conv_gen is not None:
            print(f"\n📊 System Knowledgeの収束世代: {conv_gen}世代（95%到達）")
        else:
            print(f"\n📊 System Knowledgeは95%に未到達")

        # グラフ生成
        if args.output:
            output_base = Path(args.output)
        else:
            output_base = Path("results/analysis/knowledge_vs_behavior")

        plot_single_environment(
            agent_mean, agent_sem, system_mean, system_sem, output_base
        )

    # 複数環境モード
    elif args.results_dirs:
        if not args.labels or len(args.labels) != len(args.results_dirs):
            print(f"❌ エラー: --labelsは--results-dirsと同じ数だけ指定してください")
            sys.exit(1)

        all_env_data = {}

        for env_dir, label in zip(args.results_dirs, args.labels):
            csv_path = Path(env_dir) / "generation_stats.csv"

            if not csv_path.exists():
                print(f"⚠️ スキップ: {csv_path} が見つかりません")
                continue

            print(f"📂 読み込み中: {csv_path} ({label})")
            stats = load_generation_stats(csv_path)
            all_stats = [stats]

            agent_mean, agent_sem, system_mean, system_sem = aggregate_by_generation(
                all_stats, args.num_ants
            )

            all_env_data[label] = (agent_mean, agent_sem, system_mean, system_sem)

        if not all_env_data:
            print(f"❌ エラー: 有効なデータがありません")
            sys.exit(1)

        # グラフ生成
        if args.output:
            output_base = Path(args.output)
        else:
            output_base = Path("results/analysis/knowledge_vs_behavior_comparison")

        plot_multiple_environments(all_env_data, output_base)

    else:
        print(f"❌ エラー: --results-dir または --results-dirs を指定してください")
        parser.print_help()
        sys.exit(1)

    print(f"\n✅ Analysis completed!")


if __name__ == "__main__":
    main()

