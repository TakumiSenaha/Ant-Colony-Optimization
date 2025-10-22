import csv
import math
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import networkx as nx  # type: ignore[import-untyped]

from modified_dijkstra import max_load_path

# ===== シミュレーションパラメータ =====
V = 0.98  # フェロモン揮発量
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

# ===== BKBモデル用パラメータ =====
PENALTY_FACTOR = 0.5  # BKBを下回るエッジへのペナルティ(0.0-1.0)
ACHIEVEMENT_BONUS = 1.5  # BKBを更新した場合の報酬ボーナス係数
BKB_EVAPORATION_RATE = 0.999  # BKB値の揮発率

# ===== 動的帯域変動パラメータ（AR(1)モデル） =====
BANDWIDTH_UPDATE_INTERVAL = 1  # 何世代ごとに帯域を更新するか（1=毎世代）

MEAN_UTILIZATION: float = 0.4  # (根拠: ISPの一般的な運用マージン)
AR_COEFFICIENT: float = 0.95  # (根拠: ネットワークトラフィックの高い自己相関)
NOISE_VARIANCE: float = 0.000975  # (根拠: 上記2値から逆算した値)


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


def volatilize_by_width(graph: nx.Graph) -> None:
    """
    各エッジのフェロモン値を双方向で揮発させる
    - VOLATILIZATION_MODE が 0 の場合: 固定の揮発率を適用
    - VOLATILIZATION_MODE が 1 の場合: エッジのlocal_min/max帯域幅を基準に揮発量を調整
    - VOLATILIZATION_MODE が 2 の場合: エッジの帯域幅の平均/分散を基準に揮発量を計算
    - VOLATILIZATION_MODE が 3 の場合: ノードのbest_known_bottleneck(BKB)に基づきペナルティを適用
    """
    for u, v in graph.edges():
        # u → v の揮発計算
        _apply_volatilization(graph, u, v)
        # v → u の揮発計算
        _apply_volatilization(graph, v, u)

    for node in graph.nodes():
        if "best_known_bottleneck" in graph.nodes[node]:
            graph.nodes[node]["best_known_bottleneck"] *= BKB_EVAPORATION_RATE


def _apply_volatilization(graph: nx.Graph, u: int, v: int) -> None:
    """
    指定された方向のエッジ (u → v) に対して揮発処理を適用
    """
    # 現在のフェロモン値と帯域幅を取得
    current_pheromone = graph[u][v]["pheromone"]
    weight_uv = graph[u][v]["weight"]

    # エッジのローカル最小・最大帯域幅を取得
    local_min_bandwidth = graph[u][v]["local_min_bandwidth"]
    local_max_bandwidth = graph[u][v]["local_max_bandwidth"]

    # 揮発率の計算
    if VOLATILIZATION_MODE == 0:
        # --- 既存の揮発式 ---
        # 最大帯域幅100Mbpsを基準に固定値で揮発率を計算
        rate = V

    # 0.99に設定する方が，最適解既知でないときに如実に良くなる．
    elif VOLATILIZATION_MODE == 1:
        # --- 帯域幅の最小値・最大値を基準に揮発量を調整 ---
        # エッジの帯域幅が、ローカルな最小・最大帯域幅のどの位置にあるかを計算
        if local_max_bandwidth == local_min_bandwidth:
            # 未使用エッジの場合：帯域幅が大きいほど rate が 1 に近づく
            rate = 0.98
        else:
            # 使用済みエッジの場合：帯域幅の相対位置を基準に揮発量を調整
            normalized_position = (weight_uv - local_min_bandwidth) / max(
                1, (local_max_bandwidth - local_min_bandwidth)
            )
            rate = 0.98 * normalized_position

    # FIXME: OverflowError: cannot convert float infinity to integer
    elif VOLATILIZATION_MODE == 2:
        # --- 平均・分散を基準に揮発量を調整 ---
        # 平均帯域幅と標準偏差を計算し、それを基に揮発率を算出
        if local_max_bandwidth == local_min_bandwidth:
            # 未使用エッジの場合：帯域幅が大きいほど rate が 1 に近づく
            avg_bandwidth = weight_uv
            std_dev = 1  # デフォルト値
        else:
            # 使用済みエッジの場合
            avg_bandwidth = 0.5 * (local_min_bandwidth + local_max_bandwidth)
            std_dev = max(abs(local_max_bandwidth - avg_bandwidth), 1)

        # 平均・分散に基づいて揮発率を計算
        gamma = 1.0  # 減衰率の調整パラメータ
        rate = math.exp(-gamma * (avg_bandwidth - weight_uv) / std_dev)

    elif VOLATILIZATION_MODE == 3:
        # --- ノードのBKBに基づきペナルティを適用 ---
        # 基本の残存率を設定
        rate = V

        # 行き先ノードvが知っている最良のボトルネック帯域(BKB)を取得
        bkb_v = graph.nodes[v].get("best_known_bottleneck", 0)

        # このエッジの帯域幅が、行き先ノードのBKBより低い場合、ペナルティを課す
        if weight_uv < bkb_v:
            rate *= PENALTY_FACTOR  # 残存率を下げることで、揮発を促進する

    else:
        raise ValueError("Invalid VOLATILIZATION_MODE. Choose 0, 1, 2 or 3.")

    # フェロモン値を計算して更新
    new_pheromone = max(
        math.floor(current_pheromone * rate), graph[u][v]["min_pheromone"]
    )
    graph[u][v]["pheromone"] = new_pheromone


def calculate_pheromone_increase(bottleneck_bandwidth: int) -> float:
    """
    フェロモン付加量を計算する。
    """
    # ボトルネック帯域が大きいほど、指数的に報酬を増やす
    # ただし、過大にならないよう2乗程度に抑える
    return float(bottleneck_bandwidth * 10)


def initialize_ar1_states(graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    """
    各エッジのAR(1)モデルの初期利用率を設定する
    """
    edge_states = {}
    for u, v in graph.edges():
        # u -> v / v -> u の初期利用率
        util_uv = random.uniform(0.3, 0.5)
        util_vu = random.uniform(0.3, 0.5)
        edge_states[(u, v)] = util_uv
        edge_states[(v, u)] = util_vu

        # 標準的な可用帯域計算: キャパシティ × (1 - 使用率)
        capacity = graph[u][v]["original_weight"]
        avg_util = 0.5 * (util_uv + util_vu)
        initial_available = int(round(capacity * (1.0 - avg_util)))
        # 10Mbps刻みに丸め
        initial_available = ((initial_available + 5) // 10) * 10
        graph[u][v]["weight"] = initial_available
        graph[u][v]["local_min_bandwidth"] = initial_available
        graph[u][v]["local_max_bandwidth"] = initial_available
    return edge_states


def update_available_bandwidth_ar1(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], float], generation: int
) -> bool:
    """
    AR(1)モデルによる帯域変動
    - BANDWIDTH_UPDATE_INTERVAL世代ごとにのみ更新
    """
    # 更新間隔でない世代は変化なし
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), current_utilization in edge_states.items():
        # AR(1)モデル: X(t) = c + φ*X(t-1) + ε(t)
        noise = random.gauss(0, math.sqrt(NOISE_VARIANCE))

        new_utilization = (
            (1 - AR_COEFFICIENT) * MEAN_UTILIZATION  # 平均への回帰
            + AR_COEFFICIENT * current_utilization  # 過去の値への依存
            + noise  # ランダムノイズ
        )

        # 利用率を0.05 - 0.95の範囲にクリップ
        new_utilization = max(0.05, min(0.95, new_utilization))

        # 状態を更新
        edge_states[(u, v)] = new_utilization

        # 標準的な可用帯域計算: キャパシティ × (1 - 使用率)
        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        # 10Mbps刻みに丸め
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        # 変化があったかチェック
        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        # グラフのweight属性を更新
        graph[u][v]["weight"] = available_bandwidth

        # local_min/max_bandwidth も更新
        graph[u][v]["local_min_bandwidth"] = graph[u][v]["weight"]
        graph[u][v]["local_max_bandwidth"] = graph[u][v]["weight"]

    return bandwidth_changed


def calculate_current_optimal_bottleneck(
    graph: nx.Graph, start_node: int, goal_node: int
) -> int:
    """
    現在のネットワーク状態での最適ボトルネック帯域を計算
    """
    try:
        optimal_path = max_load_path(graph, start_node, goal_node)
        optimal_bottleneck = min(
            graph.edges[u, v]["weight"]
            for u, v in zip(optimal_path[:-1], optimal_path[1:])
        )
        return optimal_bottleneck
    except nx.NetworkXNoPath:
        return 0


def update_pheromone(ant: Ant, graph: nx.Graph) -> None:
    """
    Antがゴールに到達したとき、経路上のフェロモンとノードのBKBを更新する。
    ★★★ フェロモンは経路上のエッジに「双方向」で付加する ★★★
    """
    bottleneck_bn = min(ant.width) if ant.width else 0
    if bottleneck_bn == 0:
        return

    # --- 経路上の各エッジにフェロモンを付加 ---
    for i in range(1, len(ant.route)):
        u, v = ant.route[i - 1], ant.route[i]

        # ステップ1：基本のフェロモン増加量を計算
        pheromone_increase = calculate_pheromone_increase(bottleneck_bn)

        # ステップ2：功績ボーナスの判定
        # この経路によって、行き先ノードvのBKBが更新されるか？
        current_bkb_v = graph.nodes[v].get("best_known_bottleneck", 0)
        if bottleneck_bn > current_bkb_v:
            pheromone_increase *= ACHIEVEMENT_BONUS

        # ===== ★★★ フェロモンを双方向に付加 ★★★ =====
        # 順方向 (u -> v) のフェロモンを更新
        max_pheromone_uv = graph.edges[u, v].get("max_pheromone", MAX_F)
        graph.edges[u, v]["pheromone"] = min(
            graph.edges[u, v]["pheromone"] + pheromone_increase,
            max_pheromone_uv,
        )

        # 逆方向 (v -> u) のフェロモンも更新
        max_pheromone_vu = graph.edges[v, u].get("max_pheromone", MAX_F)
        graph.edges[v, u]["pheromone"] = min(
            graph.edges[v, u]["pheromone"] + pheromone_increase,
            max_pheromone_vu,
        )
        # =======================================================

    # --- BKBの更新（フェロモン付加の後に行う）---
    # 経路上の各ノードのBKBを更新
    for node in ant.route:
        current_bkb = graph.nodes[node].get("best_known_bottleneck", 0)
        graph.nodes[node]["best_known_bottleneck"] = max(current_bkb, bottleneck_bn)


# ===== 定数ε-Greedy法 =====
def ant_next_node_const_epsilon(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    current_optimal_bottleneck: int,
    generation_bandwidth_log: list[int],
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
            update_pheromone(ant, graph)
            ant_log.append(1 if bottleneck_bw >= current_optimal_bottleneck else 0)
            generation_bandwidth_log.append(bottleneck_bw)
            ant_list.remove(ant)
        elif len(ant.route) >= TTL:
            ant_log.append(0)
            ant_list.remove(ant)


def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """
    Barabási-Albertモデルでグラフを生成
    - 各ノードに best_known_bottleneck を初期化
    - 各エッジに帯域幅(weight)等を初期化
    """
    graph = nx.barabasi_albert_graph(num_nodes, num_edges)

    # ===== 全てのノードに best_known_bottleneck 属性を初期値 0 で追加 =====
    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0
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
    - 各ノードに best_known_bottleneck を初期化
    - 各エッジに帯域幅(weight)等を初期化
    edge_probは、BAモデルと同程度のエッジ数になるように調整してください。
    """
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)

    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0

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
    - 各ノードに best_known_bottleneck を初期化
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
    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0
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

    for filename in [log_filename, log_optimal_bandwidth, log_aco_avg_bandwidth]:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted existing log file '{filename}'")
        with open(filename, "w", newline="") as f:
            pass  # Create empty file
        print(f"Initialized log file '{filename}'")

    print("\n" + "=" * 60)
    print("Simulation Settings:")
    print(f"  Ants per generation: {ANT_NUM}")
    print(f"  Number of generations: {GENERATION}")
    print("  Bandwidth variation: Every generation (AR(1) model)")
    print(f"  Number of trials: {SIMULATIONS}")
    print("=" * 60 + "\n")

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

        # Initialize AR(1) state
        edge_states = initialize_ar1_states(graph)

        # Apply initial AR(1) bandwidth update (call as generation 0)
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
            volatilize_by_width(graph)

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

                print(
                    f"Gen {generation}: Success rate = {recent_success_rate:.3f}, "
                    f"ACO avg BW = {recent_aco_avg:.1f}Mbps, "
                    f"Current optimal = {current_optimal}Mbps, "
                    f"Avg utilization = {avg_utilization:.3f}"
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

        print(
            f"Simulation {sim+1}/{SIMULATIONS} completed - "
            f"Success rate: {final_success_rate:.3f}, "
            f"ACO avg: {final_aco_avg:.1f}Mbps, "
            f"Optimal avg: {final_optimal_avg:.1f}Mbps, "
            f"Achievement: {achievement_rate:.1f}%"
        )

    print(f"\nAll {SIMULATIONS} simulations completed!")
