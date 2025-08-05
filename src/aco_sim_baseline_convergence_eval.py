import csv
import math
import os
import random
from datetime import datetime

import networkx as nx

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
        if not graph.has_edge(v, u):
            graph.add_edge(v, u)

        weight = random.randint(lb, ub) * 10
        graph.edges[u, v]["weight"] = weight
        graph.edges[v, u]["weight"] = weight
        graph.edges[u, v]["pheromone"] = MIN_F
        graph.edges[v, u]["pheromone"] = MIN_F
        graph.edges[u, v]["max_pheromone"] = MAX_F
        graph.edges[v, u]["max_pheromone"] = MAX_F
        graph.edges[u, v]["min_pheromone"] = MIN_F
        graph.edges[v, u]["min_pheromone"] = MIN_F

    return graph


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

    for i in range(1, len(ant.route)):
        u, v = ant.route[i - 1], ant.route[i]
        pheromone_increase = bottleneck_bn * 10

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
            next_node_bkbs = [
                graph.nodes[n].get("best_known_bottleneck", 0) for n in candidates
            ]

            weights = [
                (p**ALPHA) * (w**BETA) * (1 + bkb)
                for p, w, bkb in zip(pheromones, widths, next_node_bkbs)
            ]

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


def ant_next_node_const_epsilon(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    current_optimal_bottleneck: int,
) -> None:
    """
    固定パラメータ(α, β, ε)を用いた、最もシンプルなε-Greedy法で次のノードを決定する。
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
        if ant.current in ant.destinations:
            update_pheromone(ant, graph)
            ant_log.append(1 if min(ant.width) >= current_optimal_bottleneck else 0)
            ant_list.remove(ant)
        elif len(ant.route) >= TTL:
            ant_log.append(0)
            ant_list.remove(ant)


# ------------------ ★★★ メイン処理を大幅に単純化 ★★★ ------------------
if __name__ == "__main__":
    NUM_NODES = 100

    # ログファイル名を指定し、シミュレーション開始前にクリア
    output_csv_path = "./simulation_result/log_baseline.csv"
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    for sim in range(SIMULATIONS):
        # グラフ生成
        graph = ba_graph(num_nodes=NUM_NODES, num_edges=3, lb=1, ub=10)
        ant_log: list[int] = []

        # シミュレーションごとにスタート・ゴールを一度だけランダムに決定
        start_node, goal_node = random.sample(range(NUM_NODES), 2)
        goal_nodes = {goal_node}  # ゴールは単一だが、Antクラスに合わせてsetで渡す

        # 最適解を一度だけ計算
        try:
            path = max_load_path(graph, start_node, goal_node)
            optimal_bottleneck = min(
                graph.edges[u, v]["weight"] for u, v in zip(path[:-1], path[1:])
            )
            print(
                f"--- シミュレーション {sim+1}: スタート {start_node}, ゴール {goal_node} ---"
            )
            print(f"最適ボトルネック: {optimal_bottleneck}")
        except nx.NetworkXNoPath:
            print(f"⚠️ 経路が存在しません。シミュレーション {sim+1} をスキップします。")
            continue

        # 固定の世代数だけループ
        for generation in range(GENERATION):
            ants = [
                Ant(start_node, goal_nodes, [start_node], []) for _ in range(ANT_NUM)
            ]

            temp_ant_list = list(ants)
            while temp_ant_list:
                # ベースライン手法: ant_router_lookahead ではなく ant_next_node_const_epsilon を使う
                ant_next_node_const_epsilon(
                    temp_ant_list, graph, ant_log, optimal_bottleneck
                )

            volatilize(graph)

        # 1シミュレーションの結果を1行として追記
        with open(output_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        print(f"✅ シミュレーション {sim+1} 回目完了")
