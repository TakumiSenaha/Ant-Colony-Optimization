"""
実験実行スクリプト

config.yamlの設定に基づき、ACOシミュレーションを実行し、パレートフロンティアと比較評価を行います。
"""

import csv
import random
import sys
from datetime import datetime
from pathlib import Path

import yaml

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from aco_routing.algorithms.aco_solver import ACOSolver
from aco_routing.algorithms.pareto_solver import ParetoSolver
from aco_routing.algorithms.single_objective_solver import (
    bottleneck_capacity,
    max_load_path,
)
from aco_routing.core.graph import RoutingGraph
from aco_routing.utils.metrics import MetricsCalculator
from aco_routing.utils.visualization import Visualizer


def load_config(config_path: Path) -> dict:
    """
    設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        設定辞書
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def compute_optimal_solutions(
    config: dict, graph: RoutingGraph, start_node: int, goal_node: int
) -> list:
    """
    最適解を計算（単一最適解 or パレートフロンティア）

    Args:
        config: 設定辞書
        graph: ルーティンググラフ
        start_node: 開始ノード
        goal_node: 目的地ノード

    Returns:
        最適解のリスト [(bandwidth, delay, hops), ...]
    """
    optimal_solutions = []

    # 帯域のみ最適化の場合
    if config["experiment"]["target_objectives"] == ["bandwidth"]:
        try:
            optimal_path = max_load_path(graph.graph, start_node, goal_node)
            optimal_bottleneck = bottleneck_capacity(graph.graph, optimal_path)
            optimal_solutions = [(optimal_bottleneck, 0.0, 0)]
            print(f"  Optimal Bottleneck: {optimal_bottleneck} Mbps")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not calculate optimal solution: {e}")

    # パレート最適化の場合
    elif config["pareto"]["enabled"]:
        print("  Calculating Pareto Frontier (exact solution)...")
        pareto_solver = ParetoSolver(
            graph.graph, config["pareto"]["max_labels_per_node"]
        )
        try:
            pareto_frontier_with_paths = pareto_solver.find_pareto_frontier(
                start_node, goal_node
            )
            # 経路情報を除いて統一形式に変換
            optimal_solutions = [
                (pf[0], pf[1], pf[2]) for pf in pareto_frontier_with_paths
            ]
            print(f"  Pareto Frontier: {len(optimal_solutions)} solutions found.")
            print("\n  Pareto Optimal Solutions:")
            for i, sol in enumerate(optimal_solutions, 1):
                bandwidth, delay, hops = sol
                print(
                    f"    Solution {i}: Bandwidth={bandwidth:.0f} Mbps, Delay={delay:.0f} ms, Hops={hops}"
                )
        except Exception as e:
            print(f"  Error calculating Pareto Frontier: {e}")

    return optimal_solutions


def run_single_simulation(
    config: dict,
    sim: int,
    num_simulations: int,
    generations: int,
    metrics_calculator: MetricsCalculator,
) -> tuple:
    """
    1回のシミュレーションを実行

    Args:
        config: 設定辞書
        sim: シミュレーション番号（0-indexed）
        num_simulations: 総シミュレーション数
        generations: 世代数
        metrics_calculator: 評価指標計算オブジェクト

    Returns:
        (ant_log, final_solutions, pdr, dr, hv, optimal_solutions)
    """
    print(f"\n{'='*80}")
    print(f"Simulation {sim + 1}/{num_simulations}")
    print(f"{'='*80}")

    # グラフを生成
    num_nodes = config["graph"]["num_nodes"]
    graph = RoutingGraph(num_nodes, config)

    # スタートとゴールをランダムに選択
    start_node = random.randint(0, num_nodes - 1)
    goal_node = random.choice([n for n in range(num_nodes) if n != start_node])
    print(f"Start: {start_node}, Goal: {goal_node}")

    # 最適解を計算
    optimal_solutions = compute_optimal_solutions(config, graph, start_node, goal_node)

    # ACOを実行
    print("Running ACO...")
    aco_solver = ACOSolver(config, graph)
    results, ant_log = aco_solver.run(
        start_node,
        goal_node,
        generations,
        optimal_solutions=optimal_solutions,
        metrics_calculator=metrics_calculator,
    )

    # 最終世代のACO解を収集
    final_solutions = []
    for result in results[-100:]:  # 最後の100世代
        final_solutions.extend(result["solutions"])
    final_solutions = list(set(final_solutions))  # 重複除去
    print(
        f"ACO Solutions (final 100 generations): {len(final_solutions)} unique solutions"
    )

    # 評価指標を計算
    pdr, dr, hv = None, None, None
    if optimal_solutions:
        pdr = metrics_calculator.calculate_pareto_discovery_rate(
            final_solutions, optimal_solutions
        )
        dr = metrics_calculator.calculate_dominance_rate(
            final_solutions, optimal_solutions
        )
        hv = metrics_calculator.calculate_hypervolume(final_solutions)

        print("\nMetrics:")
        print(f"  Discovery Rate: {pdr:.3f}")
        print(f"  Dominance Rate: {dr:.3f}")
        print(f"  Hypervolume: {hv:.3f}")

    # 最適解到達率の表示（最適解インデックス >= 0 の割合）
    final_success_rate = (
        sum(1 for idx in ant_log if idx >= 0) / len(ant_log) if ant_log else 0
    )
    print(f"Optimal Solution Discovery Rate: {final_success_rate:.3f}")

    return ant_log, final_solutions, pdr, dr, hv, optimal_solutions


def save_and_visualize_results(
    config: dict,
    results_dir: Path,
    all_ant_logs: list,
    all_pareto_discovery_rates: list,
    all_dominance_rates: list,
    all_hypervolumes: list,
    all_optimal_solutions: list,
    visualizer: Visualizer,
) -> None:
    """
    結果を集計し、可視化する

    Args:
        config: 設定辞書
        results_dir: 結果出力ディレクトリ
        all_ant_logs: 全シミュレーションのant_logリスト
        all_pareto_discovery_rates: 全シミュレーションのPDRリスト
        all_dominance_rates: 全シミュレーションのDRリスト
        all_hypervolumes: 全シミュレーションのHVリスト
        all_optimal_solutions: 全シミュレーションの最適解リスト
        visualizer: Visualizerオブジェクト
    """
    print(f"\n{'='*80}")
    print("Summary of All Simulations")
    print(f"{'='*80}")

    # ファイル名のサフィックスを生成（target_objectivesから）
    target_objectives = config["experiment"]["target_objectives"]
    suffix = "_".join(
        target_objectives
    )  # 例: "bandwidth_delay" または "bandwidth_delay_hops"

    # 評価指標の平均値を計算
    if all_pareto_discovery_rates:
        avg_pdr = sum(all_pareto_discovery_rates) / len(all_pareto_discovery_rates)
        avg_dr = sum(all_dominance_rates) / len(all_dominance_rates)
        avg_hv = sum(all_hypervolumes) / len(all_hypervolumes)

        print(f"Average Pareto Discovery Rate: {avg_pdr:.3f}")
        print(f"Average Dominance Rate: {avg_dr:.3f}")
        print(f"Average Hypervolume: {avg_hv:.3f}")

        # サマリー可視化
        if config["output"]["save_graphs"]:
            metrics_summary = {
                "Pareto Discovery Rate": avg_pdr,
                "Dominance Rate": avg_dr,
            }
            # ファイル名にサフィックスを追加
            base_name = "metrics_summary.png"
            name_parts = base_name.rsplit(".", 1)
            filename = f"{name_parts[0]}_{suffix}.{name_parts[1]}"
            visualizer.plot_metrics_summary(
                metrics_summary,
                filename=filename,
            )

    # 最適解選択率の遷移グラフを生成
    if config["output"]["save_graphs"] and all_ant_logs:
        num_ants = config["experiment"]["num_ants"]
        # ファイル名にサフィックスを追加
        base_name = "optimal_selection_rate.svg"
        name_parts = base_name.rsplit(".", 1)
        filename = f"{name_parts[0]}_{suffix}.{name_parts[1]}"
        visualizer.plot_optimal_selection_rate(
            all_ant_logs,
            num_ants,
            filename=filename,
        )

        # パレート最適解が複数ある場合、積み上げ棒グラフを生成
        # 各シミュレーションの最適解が同じかどうかをチェック
        if all_optimal_solutions and len(all_optimal_solutions[0]) > 1:
            # 全シミュレーションで共通の最適解を使用（最初のシミュレーションの最適解）
            # 注: 各シミュレーションで最適解が異なる可能性があるが、
            #     可視化のため最初のシミュレーションの最適解を使用
            common_optimal_solutions = all_optimal_solutions[0]
            base_name_stacked = "optimal_solution_selection_stacked.svg"
            name_parts_stacked = base_name_stacked.rsplit(".", 1)
            filename_stacked = (
                f"{name_parts_stacked[0]}_{suffix}.{name_parts_stacked[1]}"
            )
            visualizer.plot_optimal_solution_selection_stacked(
                all_ant_logs,
                num_ants,
                common_optimal_solutions,
                filename=filename_stacked,
            )


def main():
    """メイン実験ループ"""
    # ===== 設定読み込み =====
    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)

    print("=" * 80)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Target Objectives: {config['experiment']['target_objectives']}")
    print("=" * 80)

    # ===== 出力ディレクトリの作成 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / config["output"]["results_dir"] / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}\n")

    # ===== CSVログファイルの初期化 =====
    log_csv_path = results_dir / "log_ant_available_bandwidth.csv"
    if log_csv_path.exists():
        log_csv_path.unlink()
    log_csv_path.touch()

    # ===== 可視化・評価指標オブジェクトの初期化 =====
    visualizer = Visualizer(results_dir)
    metrics_calculator = MetricsCalculator(
        config["pareto"]["reference_point"],
        config["experiment"]["target_objectives"],
    )

    # ===== シミュレーション実行 =====
    num_simulations = config["experiment"]["simulations"]
    generations = config["experiment"]["generations"]

    # 結果を保存するリスト
    all_pareto_discovery_rates = []
    all_dominance_rates = []
    all_hypervolumes = []
    all_ant_logs = []
    all_optimal_solutions = []

    for sim in range(num_simulations):
        # 1回のシミュレーションを実行
        ant_log, final_solutions, pdr, dr, hv, optimal_solutions = (
            run_single_simulation(
                config, sim, num_simulations, generations, metrics_calculator
            )
        )

        # CSVログに書き込み
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        # 結果を保存
        all_ant_logs.append(ant_log)
        all_optimal_solutions.append(optimal_solutions)
        if pdr is not None:
            all_pareto_discovery_rates.append(pdr)
            all_dominance_rates.append(dr)
            all_hypervolumes.append(hv)

    # ===== 結果の集計と可視化 =====
    save_and_visualize_results(
        config,
        results_dir,
        all_ant_logs,
        all_pareto_discovery_rates,
        all_dominance_rates,
        all_hypervolumes,
        all_optimal_solutions,
        visualizer,
    )

    print(f"\n✅ Experiment completed! Results saved to: {results_dir}")
    print(f"📊 CSV Log: {log_csv_path}")


if __name__ == "__main__":
    main()
