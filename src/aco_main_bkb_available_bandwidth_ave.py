import csv
import random

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import networkx as nx  # type: ignore[import-untyped]

from bandwidth_fluctuation_config import (
    BANDWIDTH_UPDATE_INTERVAL,
    initialize_ar1_states,
    print_fluctuation_settings,
    select_fluctuating_edges,
    update_available_bandwidth_ar1,
)
from bkb_learning import (
    evaporate_bkb_values,
    initialize_graph_nodes_for_simple_bkb,
    update_node_bkb_time_window_max,  # ★リングバッファ学習を追加★
)
from modified_dijkstra import max_load_path
from pheromone_update import (
    calculate_current_optimal_bottleneck,
    update_pheromone,
    volatilize_by_width,
)

# ===== シミュレーションパラメータ =====
V = 0.98  # フェロモン揮発率（残存率）
# V = 0.95  # より強い揮発（動的環境向け：古い経路を素早く忘却）
# V = 0.90  # 非常に強い揮発（高変動環境向け）
MIN_F = 100  # フェロモン最小値
MAX_F = 1000000000  # フェロモン最大値
TTL = 100  # AntのTime to Live

# ===== ACOパラメータ =====
ALPHA = 1.0  # フェロモンの影響度
BETA = 1.0  # ヒューリスティック情報(帯域幅)の影響度
EPSILON = 0.1  # ランダムに行動する固定確率
ANT_NUM = 10  # 世代ごとに探索するアリの数
GENERATION = 1000  # 総世代数
SIMULATIONS = 1  # シミュレーションの試行回数

# ===== BKBモデル用パラメータ（リングバッファサイズ1）=====
TIME_WINDOW_SIZE = 1  # リングバッファサイズ（直近1個の観測値のみ記憶）
BKB_EVAPORATION_RATE = 0.999  # BKB値の揮発率（リングバッファサイズ1では実質効果なし）
PENALTY_FACTOR = 0.5  # BKBを下回るエッジへのペナルティ

# ===== 動的帯域変動パラメータ（AR(1)モデル） =====
# 帯域変動パラメータは bandwidth_fluctuation_config.py で管理


class Ant:
    def __init__(
        self, current: int, destination: int, route: list[int], width: list[int]
    ):
        self.current = current
        self.destination = destination
        self.route = route
        self.width = width

    def __repr__(self):
        return (
            f"Ant(current={self.current}, destination={self.destination}, "
            f"route={self.route}, width={self.width})"
        )


def set_pheromone_min_max_by_degree_and_width(graph: nx.Graph) -> None:
    """
    ノードの隣接数と帯域幅に基づいて
    フェロモンの最小値と最大値を双方向に設定
    """
    for u, v in graph.edges():
        # ノードuとvの隣接ノード数を取得
        degree_u = len(list(graph.neighbors(u)))
        degree_v = len(list(graph.neighbors(v)))

        # フェロモン最小値を隣接ノード数に基づいて設定
        graph[u][v]["min_pheromone"] = MIN_F * 3 // degree_u
        graph[v][u]["min_pheromone"] = MIN_F * 3 // degree_v

        # 帯域幅に基づいてフェロモン最大値を設定
        width_u_to_v = graph[u][v]["weight"]
        width_v_to_u = graph[v][u]["weight"]

        graph[u][v]["max_pheromone"] = width_u_to_v**5
        graph[v][u]["max_pheromone"] = width_v_to_u**5


VOLATILIZATION_MODE = 3


# ===== 定数ε-Greedy法 =====
def ant_next_node_const_epsilon(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    current_optimal_bottleneck: int,
    generation_bandwidth_log: list[int],
    generation: int,
) -> None:
    """
    固定パラメータ(α, β, ε)を用いた、最もシンプルなε-Greedy法で次のノードを決定する。
    generation_bandwidth_log: 各世代でゴールに到達したアリのボトルネック帯域を記録
    """
    for ant in reversed(ant_list):
        neighbors = list(graph.neighbors(ant.current))
        candidates = [n for n in neighbors if n not in ant.route]

        if not candidates:
            ant_list.remove(ant)
            ant_log.append(0)
            continue  # 次のアリの処理へ

        # ===== 定数ε-Greedy選択 =====
        if random.random() < EPSILON:
            # 【探索】εの確率で、重みを無視してランダムに次ノードを選択
            next_node = random.choice(candidates)
        else:
            # 【活用】1-εの確率で、フェロモンと帯域幅に基づいて次ノードを選択
            pheromones = [graph[ant.current][n]["pheromone"] for n in candidates]
            widths = [graph[ant.current][n]["weight"] for n in candidates]

            # αとβは固定値を使用
            weight_pheromone = [p**ALPHA for p in pheromones]
            weight_width = [w**BETA for w in widths]
            weights = [p * w for p, w in zip(weight_pheromone, weight_width)]

            # 重みが全て0の場合や候補がない場合のフォールバック
            if not weights or sum(weights) == 0:
                next_node = random.choice(candidates)
            else:
                next_node = random.choices(candidates, weights=weights, k=1)[0]
        # =======================

        # --- antの状態更新 ---
        next_edge_bandwidth = graph[ant.current][next_node]["weight"]
        ant.route.append(next_node)
        ant.width.append(next_edge_bandwidth)
        ant.current = next_node

        # --- ゴール判定 ---
        if ant.current == ant.destination:
            bottleneck_bw = min(ant.width) if ant.width else 0
            # ★★★ 共通モジュールを使用したフェロモン更新 ★★★
            update_pheromone(
                ant,
                graph,
                generation,
                max_pheromone=MAX_F,
                bkb_update_func=lambda g, n, b, gen: update_node_bkb_time_window_max(
                    g, n, b, gen, time_window_size=TIME_WINDOW_SIZE
                ),
                pheromone_increase_func=None,  # シンプル版を使用
                observe_bandwidth_func=None,  # 帯域監視は未使用
            )
            ant_log.append(1 if bottleneck_bw >= current_optimal_bottleneck else 0)
            generation_bandwidth_log.append(bottleneck_bw)
            ant_list.remove(ant)
        elif len(ant.route) >= TTL:
            ant_log.append(0)
            ant_list.remove(ant)


def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """
    Barabási-Albertモデルでグラフを生成
    - 各ノードにBKB統計属性（平均・分散）を初期化
    - 各エッジに帯域幅(weight)等を初期化
    """
    graph = nx.barabasi_albert_graph(num_nodes, num_edges)

    # ===== BKBモデル用の属性を初期化（リングバッファ学習）=====
    initialize_graph_nodes_for_simple_bkb(graph)
    # =======================================================================

    for u, v in graph.edges():
        # リンクの帯域幅(weight)をランダムに設定
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight

        # 初期帯域幅を保存（変動の基準値として使用）
        graph[u][v]["original_weight"] = weight

        # NOTE: local_min/max_bandwidth は新しいアプローチでは使わなくなりますが、
        #       段階的な移行のため一旦残します。
        graph[u][v]["local_min_bandwidth"] = weight
        graph[u][v]["local_max_bandwidth"] = weight

        # フェロモン値を初期化
        graph[u][v]["pheromone"] = MIN_F
        graph[u][v]["max_pheromone"] = MAX_F
        graph[u][v]["min_pheromone"] = MIN_F

    return graph


def er_graph(
    num_nodes: int, edge_prob: float = 0.12, lb: int = 1, ub: int = 10
) -> nx.Graph:
    """
    Erdős–Rényi (ER)モデルでランダムグラフを生成
    - 各ノードにBKB統計属性（平均・分散）を初期化
    - 各エッジに帯域幅(weight)等を初期化
    edge_probは、BAモデルと同程度のエッジ数になるように調整してください。
    """
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)

    # BKBモデル用の属性を初期化（リングバッファ学習）
    initialize_graph_nodes_for_simple_bkb(graph)

    for u, v in graph.edges():
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight
        graph[u][v]["local_min_bandwidth"] = weight
        graph[u][v]["local_max_bandwidth"] = weight
        graph[u][v]["pheromone"] = MIN_F
        graph[u][v]["max_pheromone"] = MAX_F
        graph[u][v]["min_pheromone"] = MIN_F

    return graph


def grid_graph(num_nodes: int, lb: int = 1, ub: int = 10) -> nx.Graph:
    """
    グリッド（格子）ネットワークを生成
    - num_nodesが平方数の場合のみ対応（例: 49, 100）
    - 各ノードにBKB統計属性（平均・分散）を初期化
    - 各エッジに帯域幅(weight)等を初期化
    """
    import math

    side = int(math.sqrt(num_nodes))
    if side * side != num_nodes:
        raise ValueError("num_nodesは平方数（例: 49, 100）である必要があります")
    graph = nx.grid_2d_graph(side, side)
    # ノードをint型に変換（0, 1, ..., num_nodes-1）
    mapping = {(i, j): i * side + j for i in range(side) for j in range(side)}
    graph = nx.relabel_nodes(graph, mapping)

    # BKBモデル用の属性を初期化（リングバッファ学習）
    initialize_graph_nodes_for_simple_bkb(graph)
    for u, v in graph.edges():
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight
        graph[u][v]["local_min_bandwidth"] = weight
        graph[u][v]["local_max_bandwidth"] = weight
        graph[u][v]["pheromone"] = MIN_F
        graph[u][v]["max_pheromone"] = MAX_F
        graph[u][v]["min_pheromone"] = MIN_F
    return graph


def plot_bandwidth_comparison(
    optimal_bandwidth_per_generation: list[int],
    aco_avg_bandwidth_per_generation: list[float],
    sim_number: int,
    start_node: int,
    goal_node: int,
) -> None:
    """
    Generate a graph comparing optimal solution transition and ACO average bottleneck bandwidth

    Args:
        optimal_bandwidth_per_generation: Optimal bottleneck bandwidth for each generation
        aco_avg_bandwidth_per_generation: ACO average bottleneck bandwidth for each generation
        sim_number: Simulation number
        start_node: Start node
        goal_node: Goal node
    """
    plt.figure(figsize=(12, 6))

    generations = list(range(len(optimal_bandwidth_per_generation)))

    # Plot optimal solution transition (black solid line)
    plt.plot(
        generations,
        optimal_bandwidth_per_generation,
        label="Optimal Solution (Modified Dijkstra)",
        color="black",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=3,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.0,
        markevery=50,  # Show markers every 50 generations
    )

    # Plot ACO average bottleneck bandwidth (dark gray dashed line)
    plt.plot(
        generations,
        aco_avg_bandwidth_per_generation,
        label="ACO Average Bandwidth",
        color="dimgray",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=3,
        markerfacecolor="dimgray",
        markeredgecolor="dimgray",
        markeredgewidth=1.0,
        markevery=50,  # Show markers every 50 generations
    )

    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Bottleneck Bandwidth (Mbps)", fontsize=12)
    plt.title(
        f"Optimal vs ACO Bandwidth (Sim {sim_number}, {start_node}->{goal_node})",
        fontsize=14,
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save as SVG
    output_filename_svg = (
        f"./simulation_result/bandwidth_comparison_sim{sim_number}_"
        f"{start_node}to{goal_node}.svg"
    )
    plt.savefig(output_filename_svg, format="svg", bbox_inches="tight")

    # Save as PNG (high resolution)
    output_filename_png = (
        f"./simulation_result/bandwidth_comparison_sim{sim_number}_"
        f"{start_node}to{goal_node}.png"
    )
    plt.savefig(output_filename_png, format="png", dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Graph saved: {output_filename_svg}, {output_filename_png}")


# ------------------ Main Process ------------------
if __name__ == "__main__":  # noqa: C901
    # ===== Initialize log files =====
    import os

    log_filename = "./simulation_result/log_ant_available_bandwidth_ave.csv"
    log_optimal_bandwidth = "./simulation_result/log_optimal_bandwidth.csv"
    log_aco_avg_bandwidth = "./simulation_result/log_aco_avg_bandwidth.csv"

    # ★★★ 詳細分析用ログファイル ★★★
    log_detailed = "./simulation_result/log_detailed_tracking.csv"

    files = [log_filename, log_optimal_bandwidth, log_aco_avg_bandwidth, log_detailed]
    for filename in files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted existing log file '{filename}'")
        with open(filename, "w", newline="") as f:
            if filename == log_detailed:
                # ヘッダー行を書き込み
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "simulation",
                        "generation",
                        "optimal_bw",
                        "goal_short_bkb",
                        "goal_long_bkb",
                        "goal_effective_bkb",
                        "goal_var",
                        "confidence",
                        "tracking_rate_short",
                        "tracking_rate_effective",
                        "success_rate",
                    ]
                )
        print(f"Initialized log file '{filename}'")

    print("\n" + "=" * 70)
    print("🚀 BKB学習設定")
    learning_method = f"リングバッファ学習（サイズ={TIME_WINDOW_SIZE}、直近{TIME_WINDOW_SIZE}個の観測値のみ記憶）"
    print(f"   学習手法: {learning_method}")
    print(f"   ペナルティ係数: {PENALTY_FACTOR}")
    print(f"   帯域更新間隔: {BANDWIDTH_UPDATE_INTERVAL}世代ごと")
    print("=" * 70)
    print("Simulation Settings:")
    print(f"  Ants per generation: {ANT_NUM}")
    print(f"  Number of generations: {GENERATION}")
    print("  Bandwidth variation: Every generation (AR(1) model)")
    print(f"  Number of trials: {SIMULATIONS}")
    print("=" * 70 + "\n")

    # ===== 変動設定の表示 =====
    print_fluctuation_settings()

    for sim in range(SIMULATIONS):
        # ===== Simple fixed start/goal setting =====
        NUM_NODES = 100
        START_NODE = random.randint(0, NUM_NODES - 1)
        GOAL_NODE = random.choice([n for n in range(NUM_NODES) if n != START_NODE])

        print(
            f"\n[Simulation {sim+1}/{SIMULATIONS}] Start: {START_NODE}, Goal: {GOAL_NODE}"
        )

        # Generate graph once per simulation
        # graph = grid_graph(num_nodes=NUM_NODES, lb=1, ub=10)
        # graph = er_graph(num_nodes=NUM_NODES, edge_prob=0.12, lb=1, ub=10)
        graph = ba_graph(num_nodes=NUM_NODES, num_edges=6, lb=1, ub=15)

        set_pheromone_min_max_by_degree_and_width(graph)

        # ★変動エッジを選択 (設定に応じて自動選択)★
        fluctuating_edges = select_fluctuating_edges(graph)

        # ★変動対象エッジのみ AR(1)状態を初期化★
        edge_states = initialize_ar1_states(graph, fluctuating_edges)

        # ★初回の帯域更新も変動対象のみに適用される★
        update_available_bandwidth_ar1(graph, edge_states, 0)

        # Calculate initial optimal solution in dynamic environment (for comparison)
        try:
            initial_optimal = calculate_current_optimal_bottleneck(
                graph, START_NODE, GOAL_NODE
            )
            print(f"  Initial optimal bottleneck bandwidth: {initial_optimal}Mbps")
        except (nx.NetworkXNoPath, Exception):
            print("  Error: No path exists. Skipping...")
            continue

        ant_log: list[int] = []

        # ===== Logs for graph drawing =====
        optimal_bandwidth_per_generation: list[int] = (
            []
        )  # Optimal solution for each generation
        aco_avg_bandwidth_per_generation: list[float] = (
            []
        )  # ACO average bandwidth for each generation

        for generation in range(GENERATION):
            # === Bandwidth variation by AR(1) model (executed every generation) ===
            update_available_bandwidth_ar1(graph, edge_states, generation)

            # === Recalculate optimal solution ===
            current_optimal = calculate_current_optimal_bottleneck(
                graph, START_NODE, GOAL_NODE
            )
            if current_optimal == 0:
                # Skip if no path exists
                continue

            # Record optimal solution
            optimal_bandwidth_per_generation.append(current_optimal)

            # Bandwidth varies every generation (detailed logs displayed every 100 generations)

            # === Ant exploration ===
            ants = [
                Ant(START_NODE, GOAL_NODE, [START_NODE], []) for _ in range(ANT_NUM)
            ]

            # Record bottleneck bandwidth of ants that reached goal in this generation
            generation_bandwidth_log: list[int] = []

            temp_ant_list = list(ants)
            while temp_ant_list:
                ant_next_node_const_epsilon(
                    temp_ant_list,
                    graph,
                    ant_log,
                    current_optimal,
                    generation_bandwidth_log,
                    generation,
                )

            # Calculate average bottleneck bandwidth for this generation
            if generation_bandwidth_log:
                avg_bandwidth = sum(generation_bandwidth_log) / len(
                    generation_bandwidth_log
                )
            else:
                # Record 0 if no ants reached the goal
                avg_bandwidth = 0.0
            aco_avg_bandwidth_per_generation.append(avg_bandwidth)

            # Pheromone evaporation
            # ★★★ 共通モジュールを使用したフェロモン揮発 ★★★
            volatilize_by_width(
                graph,
                volatilization_mode=VOLATILIZATION_MODE,
                base_evaporation_rate=V,
                penalty_factor=PENALTY_FACTOR,
                adaptive_rate_func=None,  # 帯域変動パターンに基づく適応的揮発は未使用
            )
            # BKB値の揮発処理（共通モジュール使用）
            evaporate_bkb_values(graph, BKB_EVAPORATION_RATE, use_int_cast=False)

            # ★★★ 詳細ログ記録（10世代ごと） ★★★
            # リングバッファ学習では統計情報がないため、シンプルなログのみ
            if generation % 10 == 0:
                goal_bkb = graph.nodes[GOAL_NODE].get("best_known_bottleneck", 0)
                tracking_rate = goal_bkb / current_optimal if current_optimal > 0 else 0
                recent_success = (
                    sum(ant_log[-10:]) / min(len(ant_log), 10) if ant_log else 0
                )

                with open(log_detailed, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            sim + 1,
                            generation,
                            current_optimal,
                            goal_bkb,  # リングバッファ学習ではBKB値のみ
                            0.0,  # short_bkb（未使用）
                            0.0,  # long_bkb（未使用）
                            goal_bkb,  # effective_bkb（BKB値を使用）
                            0.0,  # var（未使用）
                            0.0,  # confidence（未使用）
                            tracking_rate,  # tracking_rate
                            tracking_rate,  # tracking_rate_effective（同じ値）
                            recent_success,
                        ]
                    )

            # Progress display (every 100 generations)
            if generation % 100 == 0:
                recent_success_rate = (
                    sum(ant_log[-100:]) / min(len(ant_log), 100) if ant_log else 0
                )
                avg_utilization = sum(edge_states.values()) / len(edge_states)

                # ACO average bandwidth for recent 100 generations
                recent_aco_avg = 0.0
                if len(aco_avg_bandwidth_per_generation) >= 100:
                    recent_aco_avg = sum(aco_avg_bandwidth_per_generation[-100:]) / 100
                elif aco_avg_bandwidth_per_generation:
                    recent_aco_avg = sum(aco_avg_bandwidth_per_generation) / len(
                        aco_avg_bandwidth_per_generation
                    )

                # ゴールノードのBKB値を取得（リングバッファ学習）
                goal_bkb = graph.nodes[GOAL_NODE].get("best_known_bottleneck", 0)
                bkb_display = f"ゴールBKB = {goal_bkb:.1f}Mbps"

                print(
                    f"Gen {generation}: Success rate = {recent_success_rate:.3f}, "
                    f"ACO avg BW = {recent_aco_avg:.1f}Mbps, "
                    f"Current optimal = {current_optimal}Mbps, "
                    f"Avg utilization = {avg_utilization:.3f}, "
                    f"{bkb_display}"
                )

                # Detailed output of optimal solution
                try:
                    optimal_path = max_load_path(graph, START_NODE, GOAL_NODE)
                    print(
                        f"    -> Optimal path: {' -> '.join(map(str, optimal_path[:8]))}..."
                    )
                except nx.NetworkXNoPath:
                    print("    -> Optimal path: No path")

        # --- Save results ---
        # Success rate log
        with open(log_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        # Optimal solution bandwidth log
        with open(log_optimal_bandwidth, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(optimal_bandwidth_per_generation)

        # ACO average bandwidth log
        with open(log_aco_avg_bandwidth, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(aco_avg_bandwidth_per_generation)

        # --- Generate graph ---
        plot_bandwidth_comparison(
            optimal_bandwidth_per_generation,
            aco_avg_bandwidth_per_generation,
            sim + 1,
            START_NODE,
            GOAL_NODE,
        )

        # Display final success rate
        final_success_rate = sum(ant_log) / len(ant_log) if ant_log else 0
        final_aco_avg = (
            sum(aco_avg_bandwidth_per_generation)
            / len(aco_avg_bandwidth_per_generation)
            if aco_avg_bandwidth_per_generation
            else 0
        )
        final_optimal_avg = (
            sum(optimal_bandwidth_per_generation)
            / len(optimal_bandwidth_per_generation)
            if optimal_bandwidth_per_generation
            else 0
        )
        achievement_rate = (
            (final_aco_avg / final_optimal_avg * 100) if final_optimal_avg > 0 else 0
        )

        # 最終BKB値の取得（リングバッファ学習）
        goal_bkb_final = graph.nodes[GOAL_NODE].get("best_known_bottleneck", 0)

        print(
            f"Simulation {sim+1}/{SIMULATIONS} completed - "
            f"Success rate: {final_success_rate:.3f}, "
            f"ACO avg: {final_aco_avg:.1f}Mbps, "
            f"Optimal avg: {final_optimal_avg:.1f}Mbps, "
            f"Achievement: {achievement_rate:.1f}%, "
            f"最終ゴールBKB: {goal_bkb_final:.1f}Mbps"
        )

    print(f"\nAll {SIMULATIONS} simulations completed!")
