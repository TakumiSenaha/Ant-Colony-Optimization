import csv
import math
import random

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
GENERATION = 500  # 総世代数
SIMULATIONS = 100  # シミュレーションの試行回数

# ===== BKBモデル用パラメータ =====
PENALTY_FACTOR = 0.5  # BKBを下回るエッジへのペナルティ(0.0-1.0)
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


def set_pheromone_min_max_by_degree_and_width(graph: nx.Graph) -> None:
    """ノードの隣接数と帯域幅に基づいてフェロモンの最小値と最大値を双方向に設定"""
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

    # --- ログを出力 ---
    # print(f"Edge ({u} → {v})")
    # print(f"  計算されたレート: {rate:.4f}")
    # print(f"  weight (エッジ帯域幅): {weight_uv}")
    # print(f"  local_min_bandwidth: {local_min_bandwidth}")
    # print(f"  local_max_bandwidth: {local_max_bandwidth}")
    # print(f"  新しいフェロモン値: {current_pheromone - new_pheromone}\n")


def calculate_pheromone_increase(bottleneck_bandwidth: int) -> float:
    """
    フェロモン付加量を計算する。
    """
    # ボトルネック帯域が大きいほど、指数的に報酬を増やす
    # ただし、過大にならないよう2乗程度に抑える
    return float(bottleneck_bandwidth * 10)


# ===== 新しいパラメータ（功績ボーナス）=====
ACHIEVEMENT_BONUS = 1.5  # BKBを更新した場合のフェロモン増加ボーナス係数


def update_pheromone(ant: Ant, graph: nx.Graph) -> None:
    """
    Antがゴールに到達したとき、経路上のフェロモンとノードのBKBを更新する。
    BKBを更新した経路には功績ボーナスを与える。
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

        # フェロモンを更新
        graph[u][v]["pheromone"] = min(
            graph[u][v]["pheromone"] + pheromone_increase,
            graph[u][v].get("max_pheromone", MAX_F),
        )

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


# ------------------ メイン処理 ------------------
if __name__ == "__main__":
    # ===== スタートノード切り替えのための設定 =====
    SWITCH_INTERVAL = 100  # スタートノード切り替え間隔
    NUM_NODES = 100
    START_NODE_LIST = random.sample(range(NUM_NODES), 6)
    GOAL_NODE = random.choice([n for n in range(NUM_NODES) if n not in START_NODE_LIST])
    # ==========================================

    for sim in range(SIMULATIONS):
        # グラフはシミュレーションごとに一度だけ生成
        graph = ba_graph(num_nodes=NUM_NODES, num_edges=3, lb=1, ub=10)
        set_pheromone_min_max_by_degree_and_width(graph)

        ant_log: list[int] = []

        # スタートノードごとに最適経路・ボトルネック値をキャッシュ
        optimal_bottleneck_dict = {}

        # ===== ★★★ 動的なゴール管理 ★★★ =====
        all_nodes = list(range(NUM_NODES))
        initial_provider_node = random.choice(all_nodes)
        goal_nodes = {initial_provider_node}  # setでゴールを管理

        start_node_candidates = [n for n in all_nodes if n != initial_provider_node]
        start_node_list = random.sample(start_node_candidates, 6)

        previous_start = None

        for generation in range(GENERATION):
            # ===== スタートノードの決定 =====
            phase = generation // SWITCH_INTERVAL
            if phase >= len(start_node_list):
                break

            # ===== スタート地点切り替えと動的ゴール追加処理 =====
            if generation % SWITCH_INTERVAL == 0:
                if previous_start is not None:
                    print(
                        f"キャッシュ追加: ノード {previous_start} をゴール群に追加します。"
                    )
                    goal_nodes.add(previous_start)

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
                    temp_ant_list, graph, ant_log, current_optimal_bottleneck
                )

            # フェロモンの揮発
            volatilize_by_width(graph)

        # --- 結果の保存 ---
        with open("./simulation_result/log_ant.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        print(f"✅ シミュレーション {sim+1} 回目完了")
