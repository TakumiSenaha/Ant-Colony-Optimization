"""
実験実行スクリプト

config.yamlの設定に基づき、ACOシミュレーションを実行し、パレートフロンティアと比較評価を行います。
"""

import csv
import random
import shutil
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from aco_routing.algorithms.aco_solver import ACOSolver
from aco_routing.algorithms.conventional_aco_solver import ConventionalACOSolver
from aco_routing.algorithms.pareto_solver import ParetoSolver
from aco_routing.algorithms.previous_method_aco_solver import PreviousMethodACOSolver
from aco_routing.algorithms.single_objective_solver import (
    bottleneck_capacity,
    calculate_path_delay,
    find_all_max_bottleneck_paths_with_delay_constraint,
    max_load_path,
    max_load_path_with_delay_constraint,
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


def print_run_summary(
    config: dict,
    environment: str,
    aco_method: str,
    opt_type: str,
) -> None:
    """
    実行開始時のサマリーを表示
    """
    exp = config["experiment"]
    graph_conf = config["graph"]
    delay_conf = exp.get("delay_constraint", {})
    fluct_conf = graph_conf.get("fluctuation", {})
    start_switch = exp.get("start_switching", {})

    print("=" * 80)
    print("Run Summary")
    print("=" * 80)
    print(f"Experiment Name : {exp.get('name')}")
    print(f"ACO Method      : {aco_method}")
    print(f"Environment     : {environment}")
    print(f"Opt Type        : {opt_type}")
    print(f"Objectives      : {exp.get('target_objectives')}")
    print(f"Simulations     : {exp.get('simulations')} runs")
    print(f"Generations     : {exp.get('generations')} per run")
    print(f"Ants per gen    : {exp.get('num_ants')}")
    print("-" * 80)
    print("Delay Constraint")
    print(f"  enabled       : {delay_conf.get('enabled', False)}")
    if delay_conf.get("enabled", False):
        print(f"  max_delay(ms) : {delay_conf.get('max_delay')}")
    print("-" * 80)
    print("Start Switching")
    print(f"  enabled       : {start_switch.get('enabled', False)}")
    if start_switch.get("enabled", False):
        print(f"  interval      : {start_switch.get('switch_interval')}")
        print(f"  start_nodes   : {start_switch.get('start_nodes')}")
    print("-" * 80)
    print("Bandwidth Fluctuation")
    print(f"  enabled       : {fluct_conf.get('enabled', False)}")
    if fluct_conf.get("enabled", False):
        print(f"  model         : {fluct_conf.get('model')}")
        print(f"  target_method : {fluct_conf.get('target_method')}")
        print(f"  target_pct    : {fluct_conf.get('target_percentage')}")
        print(f"  update_intvl  : {fluct_conf.get('update_interval')}")
    print("-" * 80)
    print("Graph")
    print(f"  num_nodes     : {graph_conf.get('num_nodes')}")
    print(f"  num_edges     : {graph_conf.get('num_edges')}")
    print(f"  graph_type    : {graph_conf.get('graph_type')}")
    print(f"  bandwidth_rng : {graph_conf.get('bandwidth_range')}")
    print(f"  delay_rng     : {graph_conf.get('delay_range')}")
    print("=" * 80)


def recalculate_pheromone_min_max_for_manual_environment(
    graph: RoutingGraph,
    optimal_path: list,
    aco_solver,
    aco_method: str,
    config: dict,
) -> None:
    """
    manual環境で最適経路のフェロモンmin/maxを再計算

    ACOSolver初期化後に呼び出す必要がある（ConventionalACOSolverが_reinitialize_pheromones()を呼ぶため）

    Args:
        graph: ルーティンググラフ
        optimal_path: 最適経路
        aco_solver: 初期化されたACOソルバー
        aco_method: ACO手法名
        config: 設定辞書
    """
    print("\n=== Recalculating pheromone min/max for manual environment ===")

    # ACO手法に応じて適切なmin_pheromone値を取得
    if aco_method in [
        "conventional",
        "basic_aco_no_heuristic",
        "basic_aco_with_heuristic",
    ]:
        # ConventionalACOSolverは正規化スケール（min=0.01, max=10.0）を使用
        # manual環境でも正規化スケールを維持（帯域100Mbps → η=1.0）
        base_min_pheromone = aco_solver.min_pheromone
        use_normalized_scale = True
    else:
        # 提案手法・先行研究はconfigの値を使用
        base_min_pheromone = config["aco"]["min_pheromone"]
        use_normalized_scale = False

    for u, v in zip(optimal_path[:-1], optimal_path[1:]):
        # min_pheromone: 次数（degree）に基づいて計算（双方向で異なる）
        degree_u = len(list(graph.graph.neighbors(u)))
        degree_v = len(list(graph.graph.neighbors(v)))

        if use_normalized_scale:
            # ConventionalACOSolver: 正規化スケール（固定値）
            graph.graph.edges[u, v]["min_pheromone"] = aco_solver.min_pheromone
            graph.graph.edges[v, u]["min_pheromone"] = aco_solver.min_pheromone
            graph.graph.edges[u, v]["max_pheromone"] = aco_solver.max_pheromone
            graph.graph.edges[v, u]["max_pheromone"] = aco_solver.max_pheromone
        else:
            # 提案手法・先行研究: 次数と帯域に基づいて計算
            graph.graph.edges[u, v]["min_pheromone"] = (
                base_min_pheromone * 3 // degree_u
            )
            graph.graph.edges[v, u]["min_pheromone"] = (
                base_min_pheromone * 3 // degree_v
            )
            # max_pheromone: 帯域幅の5乗（100^5 = 10,000,000,000）
            graph.graph.edges[u, v]["max_pheromone"] = int(100.0**5)
            graph.graph.edges[v, u]["max_pheromone"] = int(100.0**5)

        # デバッグ出力（最初のエッジのみ）
        if u == optimal_path[0]:
            print(
                f"  Edge ({u} → {v}): "
                f"min={graph.graph.edges[u, v]['min_pheromone']}, "
                f"max={graph.graph.edges[u, v]['max_pheromone']}"
            )

    print(f"Recalculated pheromone min/max for {len(optimal_path)-1} edges")
    print("=========================================================\n")


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
        # 遅延制約が有効な場合は「帯域最大化かつ制約内遅延」で最適解を定義
        delay_constraint = config["experiment"].get("delay_constraint", {})
        if delay_constraint.get("enabled", False):
            max_delay = delay_constraint.get("max_delay", float("inf"))
            try:
                # 遅延制約を満たす経路の中で、最大ボトルネック帯域を持つ全ての経路を探索
                all_optimal_paths = find_all_max_bottleneck_paths_with_delay_constraint(
                    graph.graph,
                    start_node,
                    goal_node,
                    max_delay,
                    bandwidth_weight="bandwidth",
                    delay_weight="delay",
                )

                if not all_optimal_paths:
                    raise nx.NetworkXNoPath("No optimal paths found")

                # 全ての最適解をリストに追加
                optimal_solutions = []
                for path in all_optimal_paths:
                    optimal_bottleneck = bottleneck_capacity(graph.graph, path)
                    optimal_delay = calculate_path_delay(graph.graph, path)
                    optimal_hops = len(path) - 1
                    optimal_solutions.append(
                        (optimal_bottleneck, optimal_delay, optimal_hops)
                    )

                # 最適解が複数ある場合は全て表示
                if len(optimal_solutions) == 1:
                    print(
                        f"  Optimal Solution (bandwidth/delay, delay constraint ≤{max_delay}ms): "
                        f"Bandwidth={optimal_solutions[0][0]} Mbps, "
                        f"Delay={optimal_solutions[0][1]:.1f} ms, "
                        f"Hops={optimal_solutions[0][2]}, "
                        f"Score (B/D)={optimal_solutions[0][0]/optimal_solutions[0][1]:.2f}"
                    )
                else:
                    print(
                        f"  Found {len(optimal_solutions)} optimal solution(s) "
                        f"(with delay constraint ≤{max_delay}ms):"
                    )
                    for idx, (bw, delay, hops) in enumerate(optimal_solutions, 1):
                        score = bw / delay if delay > 0 else 0.0
                        print(
                            f"    [{idx}] Bandwidth={bw} Mbps, "
                            f"Delay={delay:.1f} ms, Hops={hops}, "
                            f"Score (B/D)={score:.2f}"
                        )
            except Exception as e:
                print(
                    f"  ⚠️  Warning: Could not calculate optimal solution with delay constraint: {e}"
                )
        else:
            # 遅延制約なしの場合（従来通り）
            try:
                optimal_path = max_load_path(graph.graph, start_node, goal_node)
                optimal_bottleneck = bottleneck_capacity(graph.graph, optimal_path)
                optimal_solutions = [(optimal_bottleneck, 0.0, 0)]
                print(f"  Optimal Bottleneck: {optimal_bottleneck} Mbps")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not calculate optimal solution: {e}")

    # 多目的最適化の場合（bandwidth/delay または bandwidth/delay/hops）
    elif set(config["experiment"]["target_objectives"]) in [
        {"bandwidth", "delay"},
        {"bandwidth", "delay", "hops"},
    ]:
        # 遅延制約が有効な場合、bandwidth/delayスコアが最大の経路を最適解とする
        delay_constraint = config["experiment"].get("delay_constraint", {})
        if delay_constraint.get("enabled", False):
            max_delay = delay_constraint.get("max_delay", float("inf"))
            try:
                optimal_path = max_load_path_with_delay_constraint(
                    graph.graph,
                    start_node,
                    goal_node,
                    max_delay,
                    bandwidth_weight="bandwidth",
                    delay_weight="delay",
                )
                optimal_bottleneck = bottleneck_capacity(graph.graph, optimal_path)
                optimal_delay = calculate_path_delay(graph.graph, optimal_path)
                optimal_hops = len(optimal_path) - 1
                # 遅延制約を満たす経路の中で、最大ボトルネック帯域を持つ全ての経路を探索
                all_optimal_paths = find_all_max_bottleneck_paths_with_delay_constraint(
                    graph.graph,
                    start_node,
                    goal_node,
                    max_delay,
                    bandwidth_weight="bandwidth",
                    delay_weight="delay",
                )

                if not all_optimal_paths:
                    raise nx.NetworkXNoPath("No optimal paths found")

                # 全ての最適解をリストに追加
                optimal_solutions = []
                for path in all_optimal_paths:
                    optimal_bottleneck = bottleneck_capacity(graph.graph, path)
                    optimal_delay = calculate_path_delay(graph.graph, path)
                    optimal_hops = len(path) - 1
                    optimal_solutions.append(
                        (optimal_bottleneck, optimal_delay, optimal_hops)
                    )

                # 最適解が複数ある場合は全て表示
                if len(optimal_solutions) == 1:
                    optimal_score = (
                        optimal_solutions[0][0] / optimal_solutions[0][1]
                        if optimal_solutions[0][1] > 0
                        else 0.0
                    )
                    print(
                        f"  Optimal Solution (bandwidth/delay, delay constraint ≤{max_delay}ms): "
                        f"Bandwidth={optimal_solutions[0][0]} Mbps, "
                        f"Delay={optimal_solutions[0][1]:.1f} ms, "
                        f"Hops={optimal_solutions[0][2]}, "
                        f"Score (B/D)={optimal_score:.2f}"
                    )
                else:
                    print(
                        f"  Found {len(optimal_solutions)} optimal solution(s) "
                        f"(with delay constraint ≤{max_delay}ms):"
                    )
                    for idx, (bw, delay, hops) in enumerate(optimal_solutions, 1):
                        score = bw / delay if delay > 0 else 0.0
                        print(
                            f"    [{idx}] Bandwidth={bw} Mbps, "
                            f"Delay={delay:.1f} ms, Hops={hops}, "
                            f"Score (B/D)={score:.2f}"
                        )
            except Exception as e:
                print(
                    f"  ⚠️  Warning: Could not calculate optimal solution with delay constraint: {e}"
                )
        else:
            # 遅延制約なしの場合はパレート最適化として扱う
            if config["pareto"]["enabled"]:
                pass  # パレート最適化の処理に進む
            else:
                # パレート最適化が無効な場合は、bandwidth/delayスコアが最大の経路を探索
                # これは全経路を探索する必要があるため、簡易的にmax_load_pathを使用
                try:
                    optimal_path = max_load_path(graph.graph, start_node, goal_node)
                    optimal_bottleneck = bottleneck_capacity(graph.graph, optimal_path)
                    optimal_delay = calculate_path_delay(graph.graph, optimal_path)
                    optimal_hops = len(optimal_path) - 1
                    optimal_solutions = [
                        (optimal_bottleneck, optimal_delay, optimal_hops)
                    ]
                    print(
                        f"  Optimal Solution (bandwidth/delay): "
                        f"Bandwidth={optimal_bottleneck} Mbps, "
                        f"Delay={optimal_delay:.1f} ms, Hops={optimal_hops}"
                    )
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
    aco_method: str,
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
        (ant_log, final_solutions, pdr, dr, hv, optimal_solutions, results, pch_at_k, aco_ranking)
    """
    print(f"\n{'='*80}")
    print(f"Simulation {sim + 1}/{num_simulations}")
    print(f"{'='*80}")

    # グラフを生成し、最適解が見つかるまで再生成
    num_nodes = config["graph"]["num_nodes"]
    max_retries = 100  # 最大再試行回数
    retry_count = 0
    optimal_solutions = []

    delay_constraint_enabled = (
        config.get("experiment", {}).get("delay_constraint", {}).get("enabled", False)
    )

    while retry_count < max_retries:
        # グラフを生成
        graph = RoutingGraph(num_nodes, config)

        # スタートとゴールをランダムに選択
        start_node = random.randint(0, num_nodes - 1)
        goal_node = random.choice([n for n in range(num_nodes) if n != start_node])

        if retry_count == 0:
            print(f"Start: {start_node}, Goal: {goal_node}")
        else:
            print(f"Retry {retry_count}: Start: {start_node}, Goal: {goal_node}")

        # 手動設定トポロジ（Environment 1）の場合、最適経路の帯域を100Mbpsに設定
        # 注意: フェロモン関連の設定はACOSolver初期化後に行う（ConventionalACOSolverが_reinitialize_pheromones()を呼ぶため）
        graph_type = config["graph"]["graph_type"]
        optimal_path_for_manual = None
        if graph_type == "manual":
            try:
                # 最適経路を見つける（後でフェロモン設定に使用）
                optimal_path_for_manual = max_load_path(
                    graph.graph, start_node, goal_node, weight="bandwidth"
                )
                print(f"Optimal path: {' -> '.join(map(str, optimal_path_for_manual))}")

                # 最適経路の各エッジの帯域幅を100Mbpsに設定（双方向）
                for u, v in zip(
                    optimal_path_for_manual[:-1], optimal_path_for_manual[1:]
                ):
                    graph.graph.edges[u, v]["bandwidth"] = 100.0
                    graph.graph.edges[v, u]["bandwidth"] = 100.0
                    graph.graph.edges[u, v]["original_bandwidth"] = 100.0
                    graph.graph.edges[v, u]["original_bandwidth"] = 100.0
                    # 先行研究用の学習値も更新
                    graph.graph.edges[u, v]["local_min_bandwidth"] = 100.0
                    graph.graph.edges[v, u]["local_min_bandwidth"] = 100.0
                    graph.graph.edges[u, v]["local_max_bandwidth"] = 100.0
                    graph.graph.edges[v, u]["local_max_bandwidth"] = 100.0
                    print(f"Set optimal path edge ({u} → {v}) to bandwidth=100 Mbps")

                print("Optimal path bandwidth set to 100 Mbps")
                print(
                    "(Pheromone min/max will be recalculated after ACO solver initialization)"
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not set optimal path: {e}")
                # 最適経路が見つからない場合は再生成
                retry_count += 1
                if retry_count < max_retries:
                    continue
                else:
                    break

        # 最適解を計算
        optimal_solutions = compute_optimal_solutions(
            config, graph, start_node, goal_node
        )

        # 遅延制約が有効な場合、最適解が見つからない場合は再生成
        if delay_constraint_enabled:
            if optimal_solutions and len(optimal_solutions) > 0:
                # 最適解が見つかった
                break
            else:
                # 最適解が見つからなかった場合は再生成
                retry_count += 1
                if retry_count < max_retries:
                    print(
                        "  ⚠️  No path found satisfying delay constraint. "
                        "Regenerating network..."
                    )
                    continue
                else:
                    print(
                        f"  ⚠️  Failed to find a valid path after "
                        f"{max_retries} attempts."
                    )
                    # 空の最適解リストで続行
                    break
        else:
            # 遅延制約が無効な場合は、最適解が見つからなくても続行
            break

    if retry_count > 0 and retry_count < max_retries:
        print(f"  ✓ Found valid path after {retry_count} retry(ies).")

    # ACO手法を選択（引数で受け取ったaco_methodを使用）
    if aco_method == "conventional":
        # 従来手法（beta_bandwidthの値によって動作が変わる）
        beta_bw = config["aco"].get("beta_bandwidth", 0)
        if beta_bw == 0:
            print("Running Basic ACO w/o Heuristic (β=0)...")
        else:
            print(f"Running Basic ACO w/ Heuristic (β={beta_bw})...")
        aco_solver = ConventionalACOSolver(config, graph)
    elif aco_method == "basic_aco_no_heuristic":
        # 従来手法1：基本ACO（ヒューリスティックなし）
        print("Running Basic ACO w/o Heuristic (β=0)...")
        # beta_bandwidthを強制的に0に設定（オーバーライド）
        config["aco"]["beta_bandwidth_override"] = 0
        aco_solver = ConventionalACOSolver(config, graph)
    elif aco_method == "basic_aco_with_heuristic":
        # 従来手法2：基本ACO（ヒューリスティックあり）
        print("Running Basic ACO w/ Heuristic (β=1)...")
        # beta_bandwidthを強制的に1に設定（オーバーライド）
        config["aco"]["beta_bandwidth_override"] = 1
        aco_solver = ConventionalACOSolver(config, graph)
    elif aco_method == "previous":
        # 先行研究（Previous Method）：エッジベースの学習
        print("Running Previous Method (Edge-based learning)...")
        aco_solver = PreviousMethodACOSolver(config, graph)
    else:
        print("Running Proposed ACO (with BKB/BLD/BKH learning)...")
        aco_solver = ACOSolver(config, graph)

    # 【重要】manual環境の場合、ACOSolver初期化後にフェロモンmin/maxを再計算
    # ConventionalACOSolverは_reinitialize_pheromones()で全エッジを上書きするため、
    # ACOSolver初期化の後に最適経路のフェロモンmin/maxを設定し直す必要がある
    if graph_type == "manual" and optimal_path_for_manual is not None:
        recalculate_pheromone_min_max_for_manual_environment(
            graph, optimal_path_for_manual, aco_solver, aco_method, config
        )

    # ACOを実行
    results, ant_logs = aco_solver.run(
        start_node,
        goal_node,
        generations,
        optimal_solutions=optimal_solutions,
        metrics_calculator=metrics_calculator,
    )

    # ant_logsはタプル (ant_log_unique_optimal, ant_log_any_optimal) または単一のリスト
    # 提案手法の場合はタプル、従来手法の場合は単一のリスト
    if isinstance(ant_logs, tuple):
        ant_log = ant_logs  # タプルのまま返す（提案手法）
    else:
        # 従来手法：単一のリストをそのまま返す
        ant_log = ant_logs

    # 最終世代のACO解を収集
    final_solutions = []
    for result in results[-100:]:  # 最後の100世代
        final_solutions.extend(result["solutions"])
    final_solutions = list(set(final_solutions))  # 重複除去
    print(
        f"ACO Solutions (final 100 generations): {len(final_solutions)} unique solutions"
    )

    # 評価指標を計算するためのヘルパー関数を定義
    def match_optimal_local(
        solution, optimal_solutions_list, delay_constraint_on=False, max_delay=None
    ):
        """
        最適解判定
        - 帯域のみ最適化: 最適ボトルネック帯域以上なら一致
        - 遅延制約あり    : 帯域が最適以上 かつ 遅延が max_delay 以下なら一致
        """
        if not optimal_solutions_list:
            return -1
        b, d, h = solution
        for idx, (opt_b, opt_d, opt_h) in enumerate(optimal_solutions_list):
            bw_tol = max(1e-6, abs(opt_b) * 1e-6)
            if b + bw_tol < opt_b:
                continue
            if delay_constraint_on and max_delay is not None:
                delay_tol = max(1e-3, abs(max_delay) * 1e-3)
                if d > max_delay + delay_tol:
                    continue
            return idx
        return -1

    def quality_score_for_local(solution, optimal_solutions_list):
        """
        品質スコア: 最良帯域に対する比（0.0〜1.0）
        （遅延制約ありでも帯域比でスコア化）
        """
        b, _, _ = solution
        if not optimal_solutions_list:
            return -1
        best_b = max(opt_b for opt_b, _, _ in optimal_solutions_list)
        if best_b <= 0:
            return -1
        return max(0.0, min(1.0, b / best_b))

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

        print("\n  ---- Metrics ----")
        print(f"  Discovery Rate: {pdr:.3f}")
        print(f"  Dominance Rate: {dr:.3f}")
        print(f"  Hypervolume: {hv:.3f}")

    # 最適解到達率の表示（最適解インデックス >= 0 の割合）
    # 統一形式: 0以上 = 最適解のインデックス、-1 = ゴール未到達、-2 = 非最適解
    if isinstance(ant_log, tuple):
        # 提案手法：2つのログを表示
        ant_log_unique_optimal, ant_log_any_optimal = ant_log
        final_success_rate_unique = (
            sum(1 for idx in ant_log_unique_optimal if idx >= 0)
            / len(ant_log_unique_optimal)
            if ant_log_unique_optimal
            else 0
        )
        final_success_rate_any = (
            sum(1 for idx in ant_log_any_optimal if idx >= 0) / len(ant_log_any_optimal)
            if ant_log_any_optimal
            else 0
        )
        print(f"\n  ---- Optimal Solution Discovery Rate ----")
        print(f"  Unique: {final_success_rate_unique:.3f}")
        print(f"  Any: {final_success_rate_any:.3f}")
    else:
        # 従来手法：単一のログを表示
        final_success_rate = (
            sum(1 for idx in ant_log if idx >= 0) / len(ant_log) if ant_log else 0
        )
        print(f"\n  ---- Optimal Solution Discovery Rate ----")
        print(f"  Rate: {final_success_rate:.3f}")

    # 各シミュレーションごとにPCH@Kを計算
    pch_at_k = None
    aco_ranking = None
    if optimal_solutions and results:
        # このシミュレーションのACO解を選択率でランキング
        path_counter: Counter[tuple] = Counter()
        for result in results:
            solutions = result.get("solutions", [])
            for solution in solutions:
                path = tuple[Any, ...](solution)
                path_counter[path] += 1

        # ランキングを作成（選択回数で降順）
        ranking = sorted(path_counter.items(), key=lambda x: x[1], reverse=True)
        # 形式を統一: [(path, selection_rate, count), ...]
        total_count = sum(count for _, count in ranking)
        aco_ranking = [
            (path, (count / total_count * 100) if total_count > 0 else 0.0, count)
            for path, count in ranking
        ]

        # PCH@Kを計算（K = 最適解の数）
        k = len(optimal_solutions)
        pch_at_k = metrics_calculator.calculate_pch_at_k(
            aco_ranking, optimal_solutions, k
        )
        print(f"  PCH@{k}: {pch_at_k:.2f}%")

        # 各シミュレーションごとにマッチログを出力
        print(f"\n  ---- Path Selection Ranking ----")
        print(
            f"  {'Rank':<6} {'Bandwidth':<12} {'Delay':<10} {'Hops':<6} {'Rate':<10} {'Count':<8} {'Match':<8}"
        )
        print("  " + "-" * 76)
        for i, (path, rate, count) in enumerate(aco_ranking[:20], 1):
            b, d, h = path
            # 最適解と一致するかチェック
            is_optimal = False
            for opt_b, opt_d, opt_h in optimal_solutions:
                bandwidth_match = abs(b - opt_b) < max(0.01, opt_b * 0.01)
                delay_match = abs(d - opt_d) < max(0.1, opt_d * 0.01)
                hops_match = h == opt_h
                if bandwidth_match and delay_match and hops_match:
                    is_optimal = True
                    break
            match_status = "✓ Match" if is_optimal else "-"
            print(
                f"  {i:<6} {b:<12.0f} {d:<10.0f} {h:<6} {rate:<10.2f}% {count:<8} {match_status:<8}"
            )

    # ===== 各世代の最良解が最適解だった確率を計算 =====
    best_solution_optimal_rate = None
    best_solution_quality_scores = []
    interest_optimal_rate = None

    if optimal_solutions and results:
        # 各世代の最良解を判定
        best_optimal_count = 0
        interest_optimal_count = 0
        total_generations = 0

        for gen_idx, gen_result in enumerate(results):
            solutions = gen_result.get("solutions", [])
            gen_optimal_solutions = gen_result.get(
                "optimal_solutions", optimal_solutions
            )
            if not gen_optimal_solutions:
                gen_optimal_solutions = optimal_solutions

            if solutions:
                total_generations += 1
                # 最良解を選択（品質スコアが最大のもの）
                best_solution = None
                best_quality = -1.0

                for sol in solutions:
                    qs = quality_score_for_local(sol, gen_optimal_solutions)
                    if qs > best_quality:
                        best_quality = qs
                        best_solution = sol

                if best_solution:
                    best_solution_quality_scores.append(best_quality)
                    # 最良解が最適解か判定
                    optimal_index = match_optimal_local(
                        best_solution,
                        gen_optimal_solutions,
                        delay_constraint_on=delay_constraint_enabled,
                        max_delay=config.get("experiment", {})
                        .get("delay_constraint", {})
                        .get("max_delay", None),
                    )
                    if optimal_index >= 0:
                        best_optimal_count += 1

            # interest（フェロモン貪欲）が最適解か判定
            interest_sol = gen_result.get("interest_solution")
            if interest_sol:
                optimal_index_interest = match_optimal_local(
                    interest_sol,
                    gen_optimal_solutions,
                    delay_constraint_on=delay_constraint_enabled,
                    max_delay=config.get("experiment", {})
                    .get("delay_constraint", {})
                    .get("max_delay", None),
                )
                if optimal_index_interest >= 0:
                    interest_optimal_count += 1

        # 確率を計算
        best_solution_optimal_rate = (
            best_optimal_count / total_generations * 100
            if total_generations > 0
            else 0.0
        )
        interest_optimal_rate = (
            interest_optimal_count / len(results) * 100 if results else 0.0
        )

        # ===== サマリー表示 =====
        print(f"\n  ---- Simulation Summary ----")
        print(f"  Best Solution Optimal Rate: {best_solution_optimal_rate:.2f}%")
        print(f"    (Percentage of generations where best solution was optimal)")
        if best_solution_quality_scores:
            avg_best_quality = sum(best_solution_quality_scores) / len(
                best_solution_quality_scores
            )
            max_best_quality = max(best_solution_quality_scores)
            min_best_quality = min(best_solution_quality_scores)
            print(f"  Best Solution Quality Score:")
            print(f"    Average: {avg_best_quality:.3f}")
            print(f"    Maximum: {max_best_quality:.3f}")
            print(f"    Minimum: {min_best_quality:.3f}")

        print(
            f"  Interest (Pheromone-Only Greedy) Optimal Rate: {interest_optimal_rate:.2f}%"
        )
        print(
            f"    (Percentage of generations where pheromone-only greedy found optimal solution)"
        )

    # シミュレーション終了
    print(f"\n{'='*80}")

    return (
        ant_log,
        final_solutions,
        pdr,
        dr,
        hv,
        optimal_solutions,
        results,
        pch_at_k,
        aco_ranking,
    )


def save_and_visualize_results(
    config: dict,
    results_dir: Path,
    all_ant_logs: list,
    all_pareto_discovery_rates: list,
    all_dominance_rates: list,
    all_hypervolumes: list,
    all_optimal_solutions: list,
    all_results: list,
    all_pch_at_k: list,
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

    # PCH@Kの平均値を計算
    if all_pch_at_k and len(all_pch_at_k) > 0:
        avg_pch_at_k = sum(all_pch_at_k) / len(all_pch_at_k)
        print(f"Average PCH@K: {avg_pch_at_k:.2f}%")
        print("  (K = number of optimal solutions per simulation)")

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
        # パレート最適解が複数ある場合、積み上げ棒グラフを生成
        # 単一最適解の場合も可視化する
        if all_optimal_solutions and len(all_optimal_solutions) > 0:
            # 最初のシミュレーションの最適解を使用
            # 注: 各シミュレーションで最適解が異なる可能性があるが、
            #     可視化のため最初のシミュレーションの最適解を使用
            if len(all_optimal_solutions[0]) > 0:
                common_optimal_solutions = all_optimal_solutions[0]
                base_name_stacked = "optimal_solution_selection_stacked.svg"
                name_parts_stacked = base_name_stacked.rsplit(".", 1)
                filename_stacked = (
                    f"{name_parts_stacked[0]}_{suffix}.{name_parts_stacked[1]}"
                )
                num_ants = config["experiment"]["num_ants"]
                visualizer.plot_optimal_solution_selection_stacked(
                    all_ant_logs,
                    num_ants,
                    common_optimal_solutions,
                    filename=filename_stacked,
                )

    # 全パスへの選択率ランキングを計算・可視化
    if all_results and config["output"]["save_graphs"]:
        num_ants = config["experiment"]["num_ants"]
        generations = config["experiment"]["generations"]
        num_simulations = config["experiment"]["simulations"]

        # 各シミュレーションの最適解を使用（各シミュレーションごとに比較）
        ranking = visualizer.calculate_path_selection_ranking(
            all_results,
            num_simulations,
            generations,
            num_ants,
            all_optimal_solutions if all_optimal_solutions else None,
        )

        # ランキングを表示（フィルタリング前）
        print(f"\n{'='*80}")
        print("Path Selection Ranking (Top 20) - Before Pareto Filtering")
        print(f"{'='*80}")
        print(
            f"{'Rank':<6} {'Bandwidth':<12} {'Delay':<10} {'Hops':<6} {'Rate':<10} {'Count':<8} {'Match':<8}"
        )
        print("-" * 80)
        for i, (path, rate, count, is_optimal) in enumerate(ranking[:20], 1):
            b, d, h = path
            match_status = "✓ Match" if is_optimal else "-"
            print(
                f"{i:<6} {b:<12.0f} {d:<10.0f} {h:<6} {rate:<10.2f}% {count:<8} {match_status:<8}"
            )

        # 事後的パレートフィルタリングを適用
        # 上位50個の候補から非支配解を抽出
        pareto_filtered = visualizer.filter_pareto_solutions(ranking, top_n=50)

        # フィルタリング後の結果を表示
        print(f"\n{'='*80}")
        print("Pareto-Optimal Solutions (After A Posteriori Filtering)")
        print(f"{'='*80}")
        print(f"Total candidates: {len(ranking)}")
        print(f"Pareto-optimal solutions: {len(pareto_filtered)}")
        print(
            f"{'Rank':<6} {'Bandwidth':<12} {'Delay':<10} {'Hops':<6} {'Rate':<10} {'Count':<8} {'Match':<8}"
        )
        print("-" * 80)
        for i, (path, rate, count, is_optimal) in enumerate(pareto_filtered, 1):
            b, d, h = path
            match_status = "✓ Match" if is_optimal else "-"
            print(
                f"{i:<6} {b:<12.0f} {d:<10.0f} {h:<6} {rate:<10.2f}% {count:<8} {match_status:<8}"
            )

        # 正解との一致率を計算（フィルタリング前）
        if all_optimal_solutions:
            # 全シミュレーションの最適解の平均数を計算
            avg_optimal_count = (
                sum(len(opt_sols) for opt_sols in all_optimal_solutions)
                / len(all_optimal_solutions)
                if all_optimal_solutions
                else 0
            )

            top_20_optimal_count = sum(1 for _, _, _, is_opt in ranking[:20] if is_opt)
            top_20_total_rate = sum(rate for _, rate, _, _ in ranking[:20])
            top_20_optimal_rate = sum(
                rate for _, rate, _, is_opt in ranking[:20] if is_opt
            )

            print(f"\n{'='*80}")
            print("Matching Summary (Top 20 - Before Filtering)")
            print(f"{'='*80}")
            print(
                f"Optimal Solutions Found: {top_20_optimal_count} paths (avg {avg_optimal_count:.1f} optimal solutions per simulation)"
            )
            print(f"Selection Rate of Optimal Solutions: {top_20_optimal_rate:.2f}%")
            print(f"Total Selection Rate (Top 20): {top_20_total_rate:.2f}%")

            # フィルタリング後の一致率を計算
            pareto_optimal_count = sum(
                1 for _, _, _, is_opt in pareto_filtered if is_opt
            )
            pareto_total_rate = sum(rate for _, rate, _, _ in pareto_filtered)
            pareto_optimal_rate = sum(
                rate for _, rate, _, is_opt in pareto_filtered if is_opt
            )

            print(f"\n{'='*80}")
            print("Matching Summary (Pareto-Optimal Solutions)")
            print(f"{'='*80}")
            print(f"Pareto-Optimal Solutions: {len(pareto_filtered)} paths")
            print(f"Optimal Solutions Found: {pareto_optimal_count} paths")
            print(f"Selection Rate of Optimal Solutions: {pareto_optimal_rate:.2f}%")
            print(f"Total Selection Rate (Pareto-Optimal): {pareto_total_rate:.2f}%")

        # ランキングを可視化（フィルタリング前）
        base_name_ranking = "path_selection_ranking.svg"
        name_parts_ranking = base_name_ranking.rsplit(".", 1)
        filename_ranking = f"{name_parts_ranking[0]}_{suffix}.{name_parts_ranking[1]}"
        visualizer.plot_path_selection_ranking(
            ranking, top_n=20, filename=filename_ranking
        )

        # フィルタリング後の結果も可視化
        base_name_pareto = "pareto_filtered_ranking.svg"
        name_parts_pareto = base_name_pareto.rsplit(".", 1)
        filename_pareto = f"{name_parts_pareto[0]}_{suffix}.{name_parts_pareto[1]}"
        visualizer.plot_path_selection_ranking(
            pareto_filtered, top_n=len(pareto_filtered), filename=filename_pareto
        )


def determine_environment(config: dict) -> str:
    """
    設定から環境タイプを判定

    Args:
        config: 設定辞書

    Returns:
        環境タイプ: "manual", "static", "node_switching", "bandwidth_fluctuation", "delay_constraint_{max_delay}ms"
    """
    # 手動設定トポロジ（Environment 1）の判定
    graph_type = config.get("graph", {}).get("graph_type", "barabasi_albert")
    if graph_type == "manual":
        return "manual"

    # 遅延制約が有効な場合は、最優先で"delay_constraint_{max_delay}ms"を返す
    delay_constraint_config = config.get("experiment", {}).get("delay_constraint", {})
    delay_constraint = delay_constraint_config.get("enabled", False)

    if delay_constraint:
        max_delay = delay_constraint_config.get("max_delay", 25.0)
        return f"delay_constraint_{max_delay:.0f}ms"

    start_switching = (
        config.get("experiment", {}).get("start_switching", {}).get("enabled", False)
    )
    fluctuation = config.get("graph", {}).get("fluctuation", {}).get("enabled", False)

    if start_switching and fluctuation:
        # 両方が有効な場合は、より動的な環境を優先
        return "bandwidth_fluctuation"
    elif start_switching:
        return "node_switching"
    elif fluctuation:
        return "bandwidth_fluctuation"
    else:
        return "static"


def main(config_dict: dict = None):
    """
    メイン実験ループ

    Args:
        config_dict: 設定辞書（Noneの場合はconfig.yamlから読み込む）
    """
    # ===== 設定読み込み =====
    if config_dict is None:
        config_path = project_root / "config" / "config.yaml"
        config = load_config(config_path)
    else:
        config = config_dict

    # ===== 遅延制約が有効な場合、target_objectivesを自動的に変更 =====
    # 制約を満たす中で、同じ帯域幅だった場合、一番遅延が低いものが最適
    # その差分を反映するため、多目的最適化（bandwidth / delay）を使用
    delay_constraint_enabled = (
        config.get("experiment", {}).get("delay_constraint", {}).get("enabled", False)
    )
    if delay_constraint_enabled:
        # 遅延制約が有効な場合、bandwidth / delayのスコアでフェロモン付加・揮発を行う
        original_objectives = config["experiment"]["target_objectives"]
        config["experiment"]["target_objectives"] = ["bandwidth", "delay"]
        print(
            f"Delay constraint enabled: "
            f"target_objectives changed from {original_objectives} "
            f"to {config['experiment']['target_objectives']}"
        )

    # ===== 環境と手法の判定 =====
    environment = determine_environment(config)
    aco_method = config["aco"].get("method", "proposed")

    # ===== 出力ディレクトリの作成 =====
    # 構造: results/{method}/{environment}/{opt_type}/
    results_base_dir = project_root / config["output"]["results_dir"]
    delay_constraint_enabled = (
        config.get("experiment", {}).get("delay_constraint", {}).get("enabled", False)
    )
    target_obj = config["experiment"]["target_objectives"]
    if delay_constraint_enabled:
        opt_type = "delay_constraint"
    elif target_obj == ["bandwidth"]:
        opt_type = "bandwidth_only"
    elif config["pareto"]["enabled"]:
        opt_type = "pareto"
    else:
        opt_type = "multi_objective"

    results_dir = results_base_dir / aco_method / environment / opt_type

    # 実行サマリー
    print_run_summary(config, environment, aco_method, opt_type)

    # 既存のディレクトリがある場合は削除（テストモードの場合はスキップ）
    # テストモード判定：generations=10 かつ simulations=1 かつ num_nodes=20
    is_test_mode = (
        config.get("experiment", {}).get("generations") == 10
        and config.get("experiment", {}).get("simulations") == 1
        and config.get("graph", {}).get("num_nodes") == 20
    )
    # 通常の実行では、既存の結果ディレクトリを削除して刷新
    if results_dir.exists() and not is_test_mode:
        shutil.rmtree(results_dir)
        print(f"Removed existing directory: {results_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}\n")

    # ===== CSVログファイルの初期化 =====
    log_csv_path = results_dir / "ant_log.csv"
    ant_solution_log_path = results_dir / "ant_solution_log.csv"
    interest_log_path = results_dir / "interest_log.csv"
    generation_stats_path = results_dir / "generation_stats.csv"

    for p in [
        log_csv_path,
        ant_solution_log_path,
        interest_log_path,
        generation_stats_path,
    ]:
        if p.exists():
            p.unlink()
            print(f"Removed existing file: {p}")

    # ant_log.csv（従来形式：互換のためヘッダーなし）
    with open(log_csv_path, "w", newline="") as f:
        pass
    # ant_solution_log.csv（新形式：ヘッダーあり）
    with open(ant_solution_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation",
                "ant_id",
                "bandwidth",
                "delay",
                "hops",
                "is_optimal",
                "optimal_index",
                "is_unique_optimal",
                "quality_score",
            ]
        )
    # interest_log.csv（世代ごとに1行）
    with open(interest_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation",
                "bandwidth",
                "delay",
                "hops",
                "is_optimal",
                "is_unique_optimal",
                "quality_score",
            ]
        )
    # generation_stats.csv（新形式：ヘッダーあり）
    with open(generation_stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation",
                "num_ants_reached",
                "avg_bandwidth",
                "max_bandwidth",
                "min_bandwidth",
                "std_bandwidth",
                "avg_delay",
                "max_delay",
                "min_delay",
                "std_delay",
                "avg_hops",
                "max_hops",
                "min_hops",
                "std_hops",
                "avg_quality_score",
                "max_quality_score",
                "min_quality_score",
                "std_quality_score",
                "optimal_count",
                "unique_optimal_count",
                "interest_hit",
            ]
        )
    print(
        f"Initialized log files: {log_csv_path}, {ant_solution_log_path}, {generation_stats_path}"
    )

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
    all_results = []  # 全シミュレーションのresultsを保存
    all_pch_at_k = []  # 全シミュレーションのPCH@Kを保存

    for sim in range(num_simulations):
        # 1回のシミュレーションを実行
        (
            ant_log,
            final_solutions,
            pdr,
            dr,
            hv,
            optimal_solutions,
            results,
            pch_at_k,
            aco_ranking,
        ) = run_single_simulation(
            config, sim, num_simulations, generations, metrics_calculator, aco_method
        )

        # CSVログに書き込み（2列：unique_optimal, any_optimal）
        # 提案手法の場合のみ2列、従来手法の場合は1列（後方互換性のため）
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if isinstance(ant_log, tuple):
                # 提案手法：2つのログをペアで書き込み
                ant_log_unique_optimal, ant_log_any_optimal = ant_log
                for unique_val, any_val in zip(
                    ant_log_unique_optimal, ant_log_any_optimal
                ):
                    writer.writerow([unique_val, any_val])
            else:
                # 従来手法：単一のログを2列目に空値を入れて書き込み
                for val in ant_log:
                    writer.writerow([val, val])

        # 結果を保存
        if isinstance(ant_log, tuple):
            all_ant_logs.append(ant_log)
        else:
            # 従来手法の場合もタプル形式に変換
            all_ant_logs.append((ant_log, ant_log))
        all_optimal_solutions.append(optimal_solutions)
        all_results.append(results)  # resultsも保存
        if pch_at_k is not None:
            all_pch_at_k.append(pch_at_k)
        if pdr is not None:
            all_pareto_discovery_rates.append(pdr)
            all_dominance_rates.append(dr)
            all_hypervolumes.append(hv)

        # ==== 新ログ (ant_solution_log / generation_stats) を追記 ====
        delay_constraint_enabled_sim = delay_constraint_enabled

        def match_optimal(
            solution, optimal_solutions_list, delay_constraint_on=False, max_delay=None
        ):
            """
            最適解判定
            - 帯域のみ最適化: 最適ボトルネック帯域以上なら一致
            - 遅延制約あり    : 帯域が最適以上 かつ 遅延が max_delay 以下なら一致
            """
            if not optimal_solutions_list:
                return -1
            b, d, h = solution
            for idx, (opt_b, opt_d, opt_h) in enumerate(optimal_solutions_list):
                bw_tol = max(1e-6, abs(opt_b) * 1e-6)
                if b + bw_tol < opt_b:
                    continue
                if delay_constraint_on and max_delay is not None:
                    delay_tol = max(1e-3, abs(max_delay) * 1e-3)
                    if d > max_delay + delay_tol:
                        continue
                return idx
            return -1

        def quality_score_for(solution, optimal_solutions_list):
            """
            品質スコア: 最良帯域に対する比（0.0〜1.0）
            （遅延制約ありでも帯域比でスコア化）
            """
            b, _, _ = solution
            if not optimal_solutions_list:
                return -1
            best_b = max(opt_b for opt_b, _, _ in optimal_solutions_list)
            if best_b <= 0:
                return -1
            return max(0.0, min(1.0, b / best_b))

        num_ants = config["experiment"]["num_ants"]
        ant_rows = []
        interest_rows = []
        for gen_idx, gen_result in enumerate(results):
            solutions = gen_result.get("solutions", [])
            interest_sol = gen_result.get("interest_solution")
            # 各世代の最適解を取得（帯域変動が有効な場合、変動後のグラフで計算された最適解）
            gen_optimal_solutions = gen_result.get(
                "optimal_solutions", optimal_solutions
            )
            if not gen_optimal_solutions:
                gen_optimal_solutions = optimal_solutions  # フォールバック

            # 到達したアリ
            for ant_id, sol in enumerate(solutions):
                b, d, h = sol
                optimal_index = match_optimal(
                    sol,
                    gen_optimal_solutions,  # 各世代の最適解を使用
                    delay_constraint_on=delay_constraint_enabled_sim,
                    max_delay=config.get("experiment", {})
                    .get("delay_constraint", {})
                    .get("max_delay", None),
                )
                is_optimal_flag = 1 if optimal_index >= 0 else 0
                if delay_constraint_enabled_sim and gen_optimal_solutions:
                    min_delay = min(opt[1] for opt in gen_optimal_solutions)
                    delay_tol = max(1e-3, abs(min_delay) * 1e-3)
                    is_unique = (
                        1 if is_optimal_flag and d <= min_delay + delay_tol else 0
                    )
                else:
                    is_unique = is_optimal_flag
                qs = quality_score_for(sol, gen_optimal_solutions)
                ant_rows.append(
                    [
                        gen_idx,
                        ant_id,
                        b,
                        d,
                        h,
                        is_optimal_flag,
                        optimal_index,
                        is_unique,
                        qs,
                    ]
                )
            # ========== interest: フェロモン貪欲で1経路だけ選んだ解を記録 ==========
            if interest_sol:
                b_best, d_best, h_best = interest_sol
                optimal_index_interest = match_optimal(
                    interest_sol,
                    gen_optimal_solutions,  # 各世代の最適解を使用
                    delay_constraint_on=delay_constraint_enabled_sim,
                    max_delay=config.get("experiment", {})
                    .get("delay_constraint", {})
                    .get("max_delay", None),
                )
                is_optimal_interest = 1 if optimal_index_interest >= 0 else 0
                if delay_constraint_enabled_sim and gen_optimal_solutions:
                    min_delay = min(opt[1] for opt in gen_optimal_solutions)
                    delay_tol = max(1e-3, abs(min_delay) * 1e-3)
                    is_unique_interest = (
                        1
                        if is_optimal_interest and d_best <= min_delay + delay_tol
                        else 0
                    )
                else:
                    is_unique_interest = is_optimal_interest
                qs_interest = quality_score_for(interest_sol, gen_optimal_solutions)
            else:
                b_best = d_best = h_best = -1
                is_optimal_interest = is_unique_interest = -1
                qs_interest = -1

            interest_rows.append(
                [
                    gen_idx,
                    b_best,
                    d_best,
                    h_best,
                    is_optimal_interest,
                    is_unique_interest,
                    qs_interest,
                ]
            )
            # 未到達アリを -1 で補完（行数を generations * num_ants に合わせる）
            miss = max(0, num_ants - len(solutions))
            for k in range(miss):
                ant_rows.append(
                    [
                        gen_idx,
                        len(solutions) + k,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                    ]
                )

        # 追記
        if ant_rows:
            with open(ant_solution_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(ant_rows)
        if interest_rows:
            with open(interest_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(interest_rows)

        # 世代ごとの集計
        def safe_mean(values):
            return sum(values) / len(values) if values else 0.0

        def safe_std(values):
            return statistics.stdev(values) if len(values) >= 2 else 0.0

        gen_rows = []
        for gen_idx in range(generations):
            rows_g = [r for r in ant_rows if r[0] == gen_idx]
            bw_list = [r[2] for r in rows_g if r[2] >= 0]
            delay_list = [r[3] for r in rows_g if r[3] >= 0]
            hops_list = [r[4] for r in rows_g if r[4] >= 0]
            qs_list = [r[8] for r in rows_g if r[8] >= 0]
            optimal_count = sum(1 for r in rows_g if r[5] == 1)
            unique_optimal_count = sum(1 for r in rows_g if r[7] == 1)
            num_ants_reached = len(bw_list)
            # interest (世代ごと1行)
            interest_row = next((r for r in interest_rows if r[0] == gen_idx), None)
            interest_hit = 1 if interest_row and interest_row[4] == 1 else 0

            gen_rows.append(
                [
                    gen_idx,
                    num_ants_reached,
                    safe_mean(bw_list),
                    max(bw_list) if bw_list else 0.0,
                    min(bw_list) if bw_list else 0.0,
                    safe_std(bw_list),
                    safe_mean(delay_list),
                    max(delay_list) if delay_list else 0.0,
                    min(delay_list) if delay_list else 0.0,
                    safe_std(delay_list),
                    safe_mean(hops_list),
                    max(hops_list) if hops_list else 0,
                    min(hops_list) if hops_list else 0,
                    safe_std(hops_list),
                    safe_mean(qs_list),
                    max(qs_list) if qs_list else 0.0,
                    min(qs_list) if qs_list else 0.0,
                    safe_std(qs_list),
                    optimal_count,
                    unique_optimal_count,
                    interest_hit,
                ]
            )

        if gen_rows:
            with open(generation_stats_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(gen_rows)

    # ===== 結果の集計と可視化 =====
    save_and_visualize_results(
        config,
        results_dir,
        all_ant_logs,
        all_pareto_discovery_rates,
        all_dominance_rates,
        all_hypervolumes,
        all_optimal_solutions,
        all_results,
        all_pch_at_k,
        visualizer,
    )

    print(f"\n✅ Experiment completed! Results saved to: {results_dir}")
    print(f"📊 CSV Log (legacy): {log_csv_path}")
    print(f"📊 CSV Log (ant solutions): {ant_solution_log_path}")
    print(f"📊 CSV Log (interest): {interest_log_path}")
    print(f"📊 CSV Log (generation stats): {generation_stats_path}")


if __name__ == "__main__":
    main()
