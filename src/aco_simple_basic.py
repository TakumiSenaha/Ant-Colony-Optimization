import csv
import math
import random

import networkx as nx

from modified_dijkstra import max_load_path

# ===== シミュレーションパラメータ =====
V = 0.98  # フェロモン揮発量（固定）
MIN_F = 100  # フェロモン最小値
MAX_F = 1000000000  # フェロモン最大値
TTL = 100  # AntのTime to Live

# ===== ACOパラメータ =====
ALPHA = 1.0  # フェロモンの影響度
BETA = 1.0  # ヒューリスティック情報(帯域幅)の影響度
ANT_NUM = 10  # 世代ごとに探索するアリの数
GENERATION = 1000  # 総世代数
SIMULATIONS = 100  # シミュレーションの試行回数

# ===== 選択モード =====
USE_PHEROMONE_ONLY = False  # True: フェロモンのみ使用, False: フェロモン+帯域幅使用


class Ant:
    def __init__(
        self,
        current: int,
        destinations: set[int],
        route: list[int],
        width: list[int],
    ):
        self.current = current
        self.destinations = destinations
        self.route = route
        self.width = width

    def __repr__(self):
        return (
            f"Ant(current={self.current}, destinations={self.destinations}, "
            f"route={self.route}, width={self.width})"
        )


def simple_pheromone_update(ant: Ant, graph: nx.Graph) -> None:
    """
    最もシンプルなフェロモン更新：
    - ボトルネック帯域値に比例してフェロモンを付加
    - 複雑な機能（BKB、功績ボーナス等）は一切使用しない
    """
    bottleneck_bandwidth = min(ant.width) if ant.width else 0
    if bottleneck_bandwidth == 0:
        return

    # シンプルにボトルネック帯域値に比例したフェロモン量を付加
    pheromone_increase = float(bottleneck_bandwidth * 10)

    # 経路上の各エッジにフェロモンを付加（双方向）
    for i in range(1, len(ant.route)):
        u, v = ant.route[i - 1], ant.route[i]

        # 順方向 (u -> v) のフェロモンを更新
        graph.edges[u, v]["pheromone"] = min(
            graph.edges[u, v]["pheromone"] + pheromone_increase,
            MAX_F,
        )

        # 逆方向 (v -> u) のフェロモンも更新
        graph.edges[v, u]["pheromone"] = min(
            graph.edges[v, u]["pheromone"] + pheromone_increase,
            MAX_F,
        )


def simple_volatilize(graph: nx.Graph) -> None:
    """
    最もシンプルなフェロモン揮発：
    - 全エッジに固定率Vを適用
    - 複雑な調整は一切行わない
    """
    for u, v in graph.edges():
        # u → v の揮発
        current_pheromone_uv = graph[u][v]["pheromone"]
        new_pheromone_uv = max(math.floor(current_pheromone_uv * V), MIN_F)
        graph[u][v]["pheromone"] = new_pheromone_uv

        # v → u の揮発
        current_pheromone_vu = graph[v][u]["pheromone"]
        new_pheromone_vu = max(math.floor(current_pheromone_vu * V), MIN_F)
        graph[v][u]["pheromone"] = new_pheromone_vu


def simple_ant_next_node(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    current_optimal_bottleneck: int,
) -> None:
    """
    最もシンプルなACO経路選択：
    - ε-Greedyなし、完全に確率的な選択のみ
    - USE_PHEROMONE_ONLY = True: フェロモンのみで選択
    - USE_PHEROMONE_ONLY = False: フェロモン + 帯域幅で選択
    """
    for ant in reversed(ant_list):
        neighbors = list(graph.neighbors(ant.current))
        candidates = [n for n in neighbors if n not in ant.route]

        if not candidates:
            ant_list.remove(ant)
            ant_log.append(0)
            continue

        # ===== 完全確率的選択（ε-Greedyなし）=====
        pheromones = [graph[ant.current][n]["pheromone"] for n in candidates]

        if USE_PHEROMONE_ONLY:
            # ★★★ フェロモンのみで選択 ★★★
            weights = [p**ALPHA for p in pheromones]
        else:
            # ★★★ フェロモン + 帯域幅で選択 ★★★
            widths = [graph[ant.current][n]["weight"] for n in candidates]
            weight_pheromone = [p**ALPHA for p in pheromones]
            weight_width = [w**BETA for w in widths]
            weights = [p * w for p, w in zip(weight_pheromone, weight_width)]

        # 重みが全て0の場合のフォールバック
        if not weights or sum(weights) == 0:
            next_node = random.choice(candidates)
        else:
            next_node = random.choices(candidates, weights=weights, k=1)[0]

        # アリの状態更新
        next_edge_bandwidth = graph[ant.current][next_node]["weight"]
        ant.route.append(next_node)
        ant.width.append(next_edge_bandwidth)
        ant.current = next_node

        # ゴール判定
        if ant.current in ant.destinations:
            simple_pheromone_update(ant, graph)
            ant_log.append(1 if min(ant.width) >= current_optimal_bottleneck else 0)
            ant_list.remove(ant)
        elif len(ant.route) >= TTL:
            ant_log.append(0)
            ant_list.remove(ant)


def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """
    Barabási-Albertモデルでグラフを生成
    - シンプルな初期化のみ（複雑な属性は追加しない）
    """
    graph = nx.barabasi_albert_graph(num_nodes, num_edges)

    for u, v in graph.edges():
        # リンクの帯域幅(weight)をランダムに設定
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight

        # フェロモン値を初期化（シンプルに固定値）
        graph[u][v]["pheromone"] = MIN_F

    return graph


def er_graph(
    num_nodes: int, edge_prob: float = 0.12, lb: int = 1, ub: int = 10
) -> nx.Graph:
    """
    Erdős–Rényi (ER)モデルでランダムグラフを生成
    """
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)

    for u, v in graph.edges():
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight
        graph[u][v]["pheromone"] = MIN_F

    return graph


def grid_graph(num_nodes: int, lb: int = 1, ub: int = 10) -> nx.Graph:
    """
    グリッド（格子）ネットワークを生成
    """
    import math

    side = int(math.sqrt(num_nodes))
    if side * side != num_nodes:
        raise ValueError("num_nodesは平方数（例: 49, 100）である必要があります")
    graph = nx.grid_2d_graph(side, side)
    # ノードをint型に変換（0, 1, ..., num_nodes-1）
    mapping = {(i, j): i * side + j for i in range(side) for j in range(side)}
    graph = nx.relabel_nodes(graph, mapping)

    for u, v in graph.edges():
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight
        graph[u][v]["pheromone"] = MIN_F

    return graph


# ------------------ メイン処理 ------------------
if __name__ == "__main__":
    # ===== シンプルな固定スタート・ゴール設定 =====
    NUM_NODES = 100
    START_NODE = random.randint(0, NUM_NODES - 1)
    GOAL_NODE = random.choice([n for n in range(NUM_NODES) if n != START_NODE])

    selection_mode = "フェロモンのみ" if USE_PHEROMONE_ONLY else "フェロモン+帯域幅"
    print(f"シンプルACOシミュレーション開始 (選択モード: {selection_mode})")
    print(f"スタートノード: {START_NODE}, ゴールノード: {GOAL_NODE}")
    print(f"実行予定: {SIMULATIONS}回のシミュレーション × {GENERATION}世代")

    for sim in range(SIMULATIONS):
        # グラフ生成（元のコードと同じ構造を維持）
        graph = ba_graph(num_nodes=NUM_NODES, num_edges=6, lb=1, ub=10)

        ant_log: list[int] = []

        # 最適解の計算（比較用）
        try:
            optimal_path = max_load_path(graph, START_NODE, GOAL_NODE)
            optimal_bottleneck = min(
                graph.edges[u, v]["weight"]
                for u, v in zip(optimal_path[:-1], optimal_path[1:])
            )
            print(f"最適ボトルネック帯域: {optimal_bottleneck}")
        except nx.NetworkXNoPath:
            print("経路が存在しません")
            continue

        # シンプルなACOシミュレーション
        for generation in range(GENERATION):
            # アリの初期化（シンプルに固定スタート・ゴール）
            ants = [
                Ant(START_NODE, {GOAL_NODE}, [START_NODE], []) for _ in range(ANT_NUM)
            ]

            temp_ant_list = list(ants)
            while temp_ant_list:
                simple_ant_next_node(temp_ant_list, graph, ant_log, optimal_bottleneck)

            # シンプルなフェロモン揮発
            simple_volatilize(graph)

            # 進捗表示
            if generation % 100 == 0:
                recent_success_rate = (
                    sum(ant_log[-100:]) / min(len(ant_log), 100) if ant_log else 0
                )
                print(
                    f"世代 {generation}: 最近100回の成功率 = {recent_success_rate:.3f}"
                )

        # 結果の保存（選択モードに応じてファイル名を変更）
        filename = (
            "./simulation_result/log_ant_pheromone_only.csv"
            if USE_PHEROMONE_ONLY
            else "./simulation_result/log_ant_simple_basic.csv"
        )
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        # 最終成功率の表示
        final_success_rate = sum(ant_log) / len(ant_log) if ant_log else 0
        print(
            f"✅ シミュレーション {sim+1}/{SIMULATIONS} 完了 - 成功率: {final_success_rate:.3f}"
        )

    print(f"\n🎉 全{SIMULATIONS}回のシミュレーション完了！")
