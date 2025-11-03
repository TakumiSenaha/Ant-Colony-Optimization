import csv
import random

import networkx as nx

from bkb_learning import evaporate_bkb_values
from modified_dijkstra import max_load_path
from pheromone_update import update_pheromone, volatilize_by_width

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
PENALTY_FACTOR = 0.5  # BKBを下回るエッジへのペナルティ(0.0-1.0)
ACHIEVEMENT_BONUS = 1.5  # BKBを更新した場合の報酬ボーナス係数
BKB_EVAPORATION_RATE = 0.999  # BKB値の揮発率

# ===== キャッシュ管理パラメータ =====
# キャッシュの生存期間は総世代数と同じに設定(キャッシュは消滅しないシナリオ)
CACHE_LIFETIME = GENERATION


class Ant:
    def __init__(
        self, current: int, destinations: set[int], route: list[int], width: list[int]
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


def set_pheromone_min_max_by_degree_and_width(graph: nx.Graph) -> None:
    """ノードの隣接数と帯域幅に基づいて
    フェロモンの最小値と最大値を双方向に設定"""
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

# ===== 新しいパラメータ（功績ボーナス）=====
ACHIEVEMENT_BONUS = 1.5  # BKBを更新した場合のフェロモン増加ボーナス係数


# BKB更新関数のラッパー（単純なmax手法）
def _bkb_update_simple_max(
    graph: nx.Graph, node: int, bottleneck: float, generation: int
) -> None:
    """単純なmax手法でBKBを更新"""
    current_bkb = graph.nodes[node].get("best_known_bottleneck", 0)
    graph.nodes[node]["best_known_bottleneck"] = max(current_bkb, bottleneck)


# ===== 定数ε-Greedy法 =====
def ant_next_node_const_epsilon(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    current_optimal_bottleneck: int,
    generation: int,
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
            # ★★★ 共通モジュールを使用したフェロモン更新 ★★★
            update_pheromone(
                ant,
                graph,
                generation,
                max_pheromone=MAX_F,
                achievement_bonus=ACHIEVEMENT_BONUS,
                bkb_update_func=_bkb_update_simple_max,
                pheromone_increase_func=None,  # シンプル版を使用
                observe_bandwidth_func=None,  # 帯域監視は未使用
            )
            ant_log.append(1 if min(ant.width) >= current_optimal_bottleneck else 0)
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


# ------------------ メイン処理 ------------------
if __name__ == "__main__":
    # ===== スタートノード切り替えのための設定 =====
    SWITCH_INTERVAL = 100  # スタートノード切り替え間隔
    NUM_NODES = 100
    START_NODE_LIST = random.sample(range(NUM_NODES), 10)
    GOAL_NODE = random.choice([n for n in range(NUM_NODES) if n not in START_NODE_LIST])
    # ==========================================

    # ===== ログファイルの初期化 =====
    log_filename = "./simulation_result/log_ant.csv"
    with open(log_filename, "w", newline="") as f:
        pass  # 空のファイルを作成
    print(f"ログファイル '{log_filename}' を初期化しました。")

    for sim in range(100):
        # グラフはシミュレーションごとに一度だけ生成
        # graph = grid_graph(num_nodes=NUM_NODES, lb=1, ub=10)
        # graph = er_graph(num_nodes=NUM_NODES, edge_prob=0.12, lb=1, ub=10)
        graph = ba_graph(num_nodes=NUM_NODES, num_edges=6, lb=1, ub=10)

        set_pheromone_min_max_by_degree_and_width(graph)

        ant_log: list[int] = []

        # スタートノードごとに最適経路・ボトルネック値をキャッシュ
        optimal_bottleneck_dict = {}

        # ===== ★★★ 動的なゴール管理（キャッシュ消滅機能付き）★★★ =====
        all_nodes = list(range(NUM_NODES))
        initial_provider_node = random.choice(all_nodes)
        goal_nodes = {initial_provider_node}  # setでゴールを管理

        # キャッシュの生存期間を管理する辞書 {node_id: added_generation}
        cache_generation_dict = {initial_provider_node: 0}  # 初期ゴールは世代0で追加

        start_node_candidates = [n for n in all_nodes if n != initial_provider_node]
        start_node_list = random.sample(start_node_candidates, 10)

        previous_start = None

        for generation in range(GENERATION):
            # ===== スタートノードの決定 =====
            phase = generation // SWITCH_INTERVAL
            if phase >= len(start_node_list):
                break

            # ===== スタート地点切り替えと動的ゴール追加・削除処理 =====
            if generation % SWITCH_INTERVAL == 0:
                # 新しいキャッシュを追加する前に、古いキャッシュを削除
                if generation > 0:  # 最初の世代では削除しない
                    expired_caches = []
                    for cache_node, added_gen in cache_generation_dict.items():
                        if generation - added_gen >= CACHE_LIFETIME:
                            expired_caches.append(cache_node)

                    # 期限切れのキャッシュを削除（初期ゴールは削除しない）
                    for cache_node in expired_caches:
                        if cache_node != initial_provider_node:
                            goal_nodes.discard(cache_node)
                            del cache_generation_dict[cache_node]
                            print(
                                f"キャッシュ削除: ノード {cache_node} (世代 {generation} で期限切れ)"
                            )

                # 新しいキャッシュを追加
                if previous_start is not None:
                    print(
                        f"キャッシュ追加: ノード {previous_start} をゴール群に追加します。(世代 {generation})"
                    )
                    goal_nodes.add(previous_start)
                    cache_generation_dict[previous_start] = generation  # 追加世代を記録

                current_start = start_node_list[phase]
                if current_start in goal_nodes:
                    print(
                        f"警告: スタートノード {current_start} は既にゴールです。このフェーズをスキップします。"
                    )
                    optimal_bottleneck_dict[current_start] = -1
                    previous_start = current_start
                    continue

                previous_start = current_start

                print(
                    f"\n--- 世代 {generation}: スタート {current_start}, ゴール群 {goal_nodes} ---"
                )
                print(f"現在のキャッシュ状態: {cache_generation_dict}")
                print(f"アクティブなキャッシュ数: {len(goal_nodes)}")

                for node in graph.nodes():
                    graph.nodes[node]["best_known_bottleneck"] = 0

                # ★★★ 最適解の再定義：複数ゴールの中から最良のものを探す ★★★
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
                continue

            ants = [
                Ant(current_start, goal_nodes, [current_start], [])
                for _ in range(ANT_NUM)
            ]

            temp_ant_list = list(ants)
            while temp_ant_list:
                ant_next_node_const_epsilon(
                    temp_ant_list,
                    graph,
                    ant_log,
                    current_optimal_bottleneck,
                    generation,
                )

            # フェロモンの揮発
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

        # --- 結果の保存 ---
        with open("./simulation_result/log_ant.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        print(f"✅ シミュレーション {sim+1} 回目完了")
