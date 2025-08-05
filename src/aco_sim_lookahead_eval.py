import csv
import math
import random
from datetime import datetime

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

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
GENERATION = 500  # 1フェーズあたりの世代数
SIMULATIONS = 100  # シミュレーションの試行回数

# ===== BKBモデル用パラメータ =====
PENALTY_FACTOR = 0.9  # BKBを下回るエッジへのペナルティ (0.0-1.0)
ACHIEVEMENT_BONUS = 1.5  # BKBを更新した場合の報酬ボーナス係数
BKB_EVAPORATION_RATE = 0.999  # BKB値の揮発率


class Ant:
    def __init__(
        self, current: int, destinations: set[int], route: list[int], width: list[int]
    ):
        self.current = current
        self.destinations = destinations
        self.route = route
        self.width = width

    def __repr__(self):
        return f"Ant(current={self.current}, destinations={self.destinations}, route={self.route}, width={self.width})"


def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """Barabási-Albertモデルでグラフを生成し、属性を初期化する"""
    graph = nx.barabasi_albert_graph(num_nodes, num_edges).to_directed()
    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0

    for u, v in graph.edges():
        if not graph.has_edge(
            v, u
        ):  # BAモデルは有向グラフを返すことがあるため双方向を保証
            graph.add_edge(v, u)

        weight = random.randint(lb, ub) * 10
        graph.edges[u, v]["weight"] = weight
        graph.edges[v, u]["weight"] = weight
        graph.edges[u, v]["pheromone"] = MIN_F
        graph.edges[v, u]["pheromone"] = MIN_F

    set_pheromone_min_max_by_degree(graph)
    return graph


def set_pheromone_min_max_by_degree(graph: nx.Graph) -> None:
    """ノードの次数に基づいてフェロモンの最小値・最大値を設定"""
    for u, v in graph.edges():
        degree_u = graph.degree(u)
        degree_v = graph.degree(v)
        graph.edges[u, v]["min_pheromone"] = MIN_F * 3 // degree_u
        graph.edges[v, u]["min_pheromone"] = MIN_F * 3 // degree_v
        graph.edges[u, v]["max_pheromone"] = MAX_F
        graph.edges[v, u]["max_pheromone"] = MAX_F


def _apply_volatilization(graph: nx.Graph, u: int, v: int) -> None:
    """指定されたエッジ(u->v)のフェロモンを揮発させる"""
    rate = V
    bkb_v = graph.nodes[v].get("best_known_bottleneck", 0)
    if graph.edges[u, v]["weight"] < bkb_v:
        rate *= PENALTY_FACTOR

    current_pheromone = graph.edges[u, v]["pheromone"]
    min_pheromone = graph.edges[u, v].get("min_pheromone", MIN_F)
    graph.edges[u, v]["pheromone"] = max(
        math.floor(current_pheromone * rate), min_pheromone
    )


def volatilize(graph: nx.Graph) -> None:
    """全エッジのフェロモンと全ノードのBKBを揮発させる"""
    for u, v in graph.edges():
        _apply_volatilization(graph, u, v)

    for node in graph.nodes():
        if "best_known_bottleneck" in graph.nodes[node]:
            graph.nodes[node]["best_known_bottleneck"] *= BKB_EVAPORATION_RATE


def update_pheromone(ant: Ant, graph: nx.Graph) -> None:
    """アリの経路にフェロモンを付加し、ノードのBKBを更新する"""
    bottleneck_bn = min(ant.width) if ant.width else 0
    if bottleneck_bn == 0:
        return

    # 経路上の各エッジに双方向でフェロモンを付加
    for i in range(1, len(ant.route)):
        u, v = ant.route[i - 1], ant.route[i]
        pheromone_increase = bottleneck_bn * 10

        # 行き先ノードvと、帰り道でのuのBKB更新を評価
        current_bkb_v = graph.nodes[v].get("best_known_bottleneck", 0)
        current_bkb_u = graph.nodes[u].get("best_known_bottleneck", 0)
        if bottleneck_bn > current_bkb_v or bottleneck_bn > current_bkb_u:
            pheromone_increase = int(pheromone_increase * ACHIEVEMENT_BONUS)

        max_pheromone_uv = graph.edges[u, v].get("max_pheromone", MAX_F)
        graph.edges[u, v]["pheromone"] = min(
            graph.edges[u, v]["pheromone"] + pheromone_increase, max_pheromone_uv
        )

        max_pheromone_vu = graph.edges[v, u].get("max_pheromone", MAX_F)
        graph.edges[v, u]["pheromone"] = min(
            graph.edges[v, u]["pheromone"] + pheromone_increase, max_pheromone_vu
        )

    # 経路上の各ノードのBKBを更新
    for node in ant.route:
        current_bkb = graph.nodes[node].get("best_known_bottleneck", 0)
        graph.nodes[node]["best_known_bottleneck"] = max(current_bkb, bottleneck_bn)


def ant_router_lookahead(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    current_optimal_bottleneck: int,
) -> None:
    """「先読み(Lookahead)」機能を持つアリの次ノード選択と状態更新"""
    for ant in reversed(ant_list):
        neighbors = list(graph.neighbors(ant.current))
        candidates = [n for n in neighbors if n not in ant.route]

        if not candidates:
            ant_list.remove(ant)
            ant_log.append(0)
            continue

        if random.random() < EPSILON:
            next_node = random.choice(candidates)
        else:
            pheromones = [graph.edges[ant.current, n]["pheromone"] for n in candidates]
            widths = [graph.edges[ant.current, n]["weight"] for n in candidates]

            # ★★★ 変更点：「匂い」＝移動先ノードのBKBをヒューリスティック情報に追加 ★★★
            next_node_bkbs = [
                graph.nodes[n].get("best_known_bottleneck", 0) for n in candidates
            ]

            weights = [
                (p**ALPHA) * (w**BETA) * (1 + bkb)
                for p, w, bkb in zip(pheromones, widths, next_node_bkbs)
            ]
            # ====================================================================

            if not weights or sum(weights) == 0:
                next_node = random.choice(candidates)
            else:
                next_node = random.choices(candidates, weights=weights, k=1)[0]

        ant.route.append(next_node)
        ant.width.append(graph.edges[ant.current, next_node]["weight"])
        ant.current = next_node

        if ant.current in ant.destinations:
            update_pheromone(ant, graph)
            ant_log.append(1 if min(ant.width) >= current_optimal_bottleneck else 0)
            ant_list.remove(ant)
        elif len(ant.route) >= TTL:
            ant_log.append(0)
            ant_list.remove(ant)


# ------------------ メイン処理 ------------------
if __name__ == "__main__":
    SWITCH_INTERVAL = 200
    NUM_NODES = 100

    for sim in range(SIMULATIONS):
        graph = ba_graph(num_nodes=NUM_NODES, num_edges=3, lb=1, ub=10)
        ant_log: list[int] = []
        optimal_bottleneck_dict = {}

        all_nodes = list(range(NUM_NODES))
        initial_provider_node = random.choice(all_nodes)
        goal_nodes = {initial_provider_node}

        start_node_candidates = [n for n in all_nodes if n != initial_provider_node]
        start_node_list = random.sample(start_node_candidates, 10)

        previous_start = None

        total_generations = len(start_node_list) * SWITCH_INTERVAL
        for generation in range(total_generations):
            phase = generation // SWITCH_INTERVAL

            if generation % SWITCH_INTERVAL == 0:
                if previous_start is not None:
                    print(
                        f"キャッシュ追加: ノード {previous_start} をゴール群に追加します。"
                    )
                    goal_nodes.add(previous_start)

                current_start = start_node_list[phase]
                previous_start = current_start

                if current_start in goal_nodes:
                    optimal_bottleneck_dict[current_start] = -1
                    print(
                        f"\n--- 世代 {generation}: スタート {current_start} は既にゴールのためスキップ ---"
                    )
                    continue

                print(
                    f"\n--- 世代 {generation}: スタート {current_start}, ゴール群 {goal_nodes} ---"
                )

                for node in graph.nodes():
                    graph.nodes[node]["best_known_bottleneck"] = 0

                best_bottleneck_for_phase = 0
                for goal in goal_nodes:
                    if current_start == goal:
                        continue
                    try:
                        path = max_load_path(graph, current_start, goal)
                        bottleneck = min(
                            graph.edges[u, v]["weight"]
                            for u, v in zip(path[:-1], path[1:])
                        )
                        best_bottleneck_for_phase = max(
                            best_bottleneck_for_phase, bottleneck
                        )
                    except nx.NetworkXNoPath:
                        continue

                optimal_bottleneck_dict[current_start] = best_bottleneck_for_phase
                print(f"現在の最適ボトルネック: {best_bottleneck_for_phase}")

            current_start = start_node_list[phase]
            current_optimal_bottleneck = optimal_bottleneck_dict.get(current_start, -1)
            if current_optimal_bottleneck <= 0:
                ant_log.extend([0] * ANT_NUM)  # ログの長さを合わせるためのパディング
                continue

            ants = [
                Ant(current_start, goal_nodes, [current_start], [])
                for _ in range(ANT_NUM)
            ]

            temp_ant_list = list(ants)
            while temp_ant_list:
                ant_router_lookahead(
                    temp_ant_list, graph, ant_log, current_optimal_bottleneck
                )

            volatilize(graph)

        # ログの長さを揃えるための最終パディング
        expected_len = total_generations * ANT_NUM
        if len(ant_log) < expected_len:
            ant_log.extend([0] * (expected_len - len(ant_log)))

        with open(
            "./simulation_result/log_lookahead_caching.csv", "a", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        print(f"✅ シミュレーション {sim+1} 回目完了")
