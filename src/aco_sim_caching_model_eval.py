import csv
import math
import random
import sys
from datetime import datetime

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from modified_dijkstra import max_load_path

V = 0.98  # フェロモン揮発量
MIN_F = 100  # フェロモン最小値
MAX_F = 1000000000  # フェロモン最大値
TTL = 100  # AntのTime to Live
W = 1000  # 帯域幅初期値
ALPHA = 1.0  # フェロモンの影響度（固定）
BETA = 1  # 経路選択の際のヒューリスティック値に対する重み(累乗)
EPSILON = 0.1  # ランダムに行動する固定確率 (例: 10%)

ANT_NUM = 10  # 一回で放つAntの数
GENERATION = 1000  # ant，interestを放つ回数(世代)
SIMULATIONS = 100


class Ant:
    def __init__(
        self, current: int, destination: int, route: list[int], width: list[int]
    ):
        self.current = current  # 現在のノード
        self.destination = destination  # コンテンツ保持ノード
        self.route = route  # 辿ってきた経路の配列
        self.width = width  # 辿ってきた経路の帯域の配列

    def __repr__(self):
        return f"Ant(current={self.current}, destination={self.destination}, route={self.route}, width={self.width})"


class Interest:
    def __init__(self, current: int, destination: int, route: list[int], minwidth: int):
        self.current = current  # 現在のノード
        self.destination = destination  # コンテンツ保持ノード
        self.route = route  # 辿ってきた経路の配列
        self.minwidth = minwidth  # 辿ってきた経路の最小帯域

    def __repr__(self):
        return f"Interest(current={self.current}, destination={self.destination}, route={self.route}, minwidth={self.minwidth})"


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


# ===================== 揮発式の切り替えオプション =====================
# VOLATILIZATION_MODE:
# 0: 既存の揮発式（固定値を基準に帯域幅で揮発量を調整）
# 1: 帯域幅の最小値・最大値を基準に揮発量を動的に調整
# 2: 帯域幅の平均・分散を基準に揮発量を計算
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

    bkb_evaporation_rate = 0.999  # BKBを維持する割合
    for node in graph.nodes():
        if "best_known_bottleneck" in graph.nodes[node]:
            graph.nodes[node]["best_known_bottleneck"] *= bkb_evaporation_rate


PENALTY_FACTOR = 0.5  # BKBを下回るエッジへのペナルティ（残存率をさらに下げる）


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


def ant_next_node(
    ant_list: list[Ant], graph: nx.Graph, ant_log: list[int], optimal_bottleneck: int
) -> None:
    """
    Antの次の移動先を決定し、移動を実行
    """
    for ant in reversed(ant_list):
        neighbors = list(graph.neighbors(ant.current))
        # 戻ることは基本的に許されていない
        candidates = [n for n in neighbors if n not in ant.route]

        # 候補先がないなら削除（戻ることしか出来なくなったら探索失敗）
        if not candidates:
            ant_list.remove(ant)
            ant_log.append(0)
            print(f"Ant Can't Find Route! → {ant.route}")
        else:
            # フェロモン値と帯域幅を取得
            pheromones = [graph[ant.current][n]["pheromone"] for n in candidates]
            widths = [graph[ant.current][n]["weight"] for n in candidates]

            # 帯域幅に基づいた重み付け
            weight_width = [w**BETA for w in widths]
            weights = [p * w for p, w in zip(pheromones, weight_width)]

            # フェロモン値に基づいて次のノードを選択
            # Option: 帯域幅を考慮しない場合以下の行のコメントアウトを外す
            # weights = pheromones  # フェロモン値のみを考慮

            # 重みに基づいて次のノードを選択
            next_node = random.choices(candidates, weights=weights, k=1)[0]

            # ---antのルートと帯域幅を更新---
            # 次のリンクの帯域幅を取得
            next_edge_bandwidth = graph[ant.current][next_node]["weight"]
            ant.route.append(next_node)
            ant.width.append(next_edge_bandwidth)

            # 次のノードに移動
            ant.current = next_node

            # 目的ノードに到達した場合、フェロモンを更新してリストから削除
            if ant.current == ant.destination:
                update_pheromone(ant, graph)
                # 2値記録: 最適ボトルネック値と一致なら1, そうでなければ0
                ant_log.append(1 if min(ant.width) == optimal_bottleneck else 0)
                ant_list.remove(ant)
                print(f"Ant Goal! → {ant.route} : {min(ant.width)}")
            # TTL（生存時間）を超えた場合もリストから削除
            elif len(ant.route) == TTL:
                ant_log.append(0)
                ant_list.remove(ant)
                print(f"Ant TTL! → {ant.route}")


def ant_next_node_aware_generation(
    ant_list: list[Ant], graph: nx.Graph, ant_log: list[int], generation: int
) -> None:
    """Antの次の移動先を決定し、移動を実行"""
    alpha = 1 + (generation / GENERATION) * 5  # フェロモンの影響を増加（1 → 6）
    beta = BETA  # 必要ならBETAも世代で変化させられる

    for ant in reversed(ant_list):
        neighbors = list(graph.neighbors(ant.current))
        candidates = [n for n in neighbors if n not in ant.route]

        if not candidates:  # 候補がない場合、探索を終了
            ant_list.remove(ant)
            ant_log.append(0)
            print(f"Ant Can't Find Route! → {ant.route}")
        else:
            # フェロモン値と帯域幅を取得
            pheromones = [graph[ant.current][n]["pheromone"] for n in candidates]
            widths = [graph[ant.current][n]["weight"] for n in candidates]

            # フェロモンと帯域幅の影響を調整
            weight_pheromone = [
                p**alpha for p in pheromones
            ]  # フェロモンの影響を世代で増加
            weight_width = [w**beta for w in widths]
            weights = [p * w for p, w in zip(weight_pheromone, weight_width)]

            # 次のノードを選択
            next_node = random.choices(candidates, weights=weights, k=1)[0]

            # Antの状態を更新
            ant.route.append(next_node)
            ant.width.append(graph[ant.current][next_node]["weight"])
            ant.current = next_node

            # ゴールに到達した場合
            if ant.current == ant.destination:
                update_pheromone(ant, graph)
                ant_log.append(min(ant.width))
                ant_list.remove(ant)
                print(f"Ant Goal! → {ant.route} : {min(ant.width)}")
            elif len(ant.route) == TTL:  # TTLを超えた場合
                ant_log.append(0)
                ant_list.remove(ant)
                print(f"Ant TTL! → {ant.route}")


# ===== 定数ε-Greedy法 =====
def ant_next_node_const_epsilon(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    optimal_bottleneck: int,
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
        if ant.current == ant.destination:
            update_pheromone(ant, graph)
            ant_log.append(1 if min(ant.width) == optimal_bottleneck else 0)
            ant_list.remove(ant)
        elif len(ant.route) >= TTL:
            ant_log.append(0)
            ant_list.remove(ant)


def interest_next_node(
    interest_list: list[Interest], graph: nx.Graph, interest_log: list[int]
) -> None:
    """Interestの次の移動先を決定し、移動を実行"""
    for interest in reversed(interest_list):
        neighbors = list(graph.neighbors(interest.current))
        candidates = [n for n in neighbors if n not in interest.route]

        # 候補先がないなら削除
        if not candidates:
            interest_list.remove(interest)
            interest_log.append(0)
            print(f"Interest Can't Find Route! → {interest.route}")
        else:
            # 候補先の帯域幅を取得（双方向対応）
            widths = []
            for n in candidates:
                # 双方向リンクの帯域幅を確認
                if "weight" in graph[interest.current][n]:
                    widths.append(graph[interest.current][n]["weight"])
                elif "weight" in graph[n][interest.current]:  # 逆方向も確認
                    widths.append(graph[n][interest.current]["weight"])
                else:
                    print(f"Warning: No weight data between {interest.current} and {n}")
                    widths.append(0)  # デフォルトで 0 を追加

            # 最大の帯域幅を持つノードを選択
            next_node = candidates[widths.index(max(widths))]

            # interestのルートを更新
            interest.route.append(next_node)
            interest.current = next_node

            # 帯域幅の最小値を更新（双方向確認）
            if "weight" in graph[interest.route[-2]][interest.route[-1]]:
                interest.minwidth = min(
                    interest.minwidth,
                    graph[interest.route[-2]][interest.route[-1]]["weight"],
                )
            elif "weight" in graph[interest.route[-1]][interest.route[-2]]:
                interest.minwidth = min(
                    interest.minwidth,
                    graph[interest.route[-1]][interest.route[-2]]["weight"],
                )

            # 目的ノードに到達した場合
            if interest.current == interest.destination:
                interest_log.append(interest.minwidth)
                interest_list.remove(interest)
                print(f"Interest Goal! → {interest.route} : {interest.minwidth}")

            # TTL（生存時間）を超えた場合
            elif len(interest.route) == TTL:
                interest_log.append(0)
                interest_list.remove(interest)
                print(f"Interest TTL! → {interest.route}")


def load_graph(file_name: str) -> nx.Graph:
    """保存されたエッジリスト形式のグラフを読み込む"""
    graph = nx.read_edgelist(file_name, data=[("weight", float)], nodetype=int)

    # ===== 全てのノードに best_known_bottleneck 属性を初期値 0 で追加 =====
    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0
    # =======================================================================

    # 読み込んだグラフのエッジに初期フェロモン値を追加
    for u, v in graph.edges():
        graph[u][v]["pheromone"] = MIN_F
        graph[u][v]["local_min_bandwidth"] = graph[u][v]["weight"]
        graph[u][v]["local_max_bandwidth"] = graph[u][v]["weight"]

    return graph


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


def make_graph_bidirectional(graph: nx.Graph) -> nx.DiGraph:
    """
    無向グラフを双方向グラフに変換し、双方向のエッジに明示的に属性を設定
    """
    directed_G = nx.DiGraph()

    for u, v, data in graph.edges(data=True):
        weight = data["weight"]
        local_min_bandwidth = data["local_min_bandwidth"]
        local_max_bandwidth = data["local_max_bandwidth"]
        pheromone = data["pheromone"]

        # 双方向エッジを作成
        directed_G.add_edge(
            u,
            v,
            weight=weight,
            pheromone=pheromone,
            local_min_bandwidth=local_min_bandwidth,
            local_max_bandwidth=local_max_bandwidth,
        )
        directed_G.add_edge(
            v,
            u,
            weight=weight,
            pheromone=pheromone,
            local_min_bandwidth=local_min_bandwidth,
            local_max_bandwidth=local_max_bandwidth,
        )

    return directed_G


def set_optimal_path(
    graph: nx.Graph,
    start: int,
    goal: int,
    min_pheromone: int = 100,
    max_hops: int = 5,
    max_attempts: int = 100,
    max_weight: int = 100,
) -> nx.Graph:
    """
    指定されたスタートノードとゴールノードの間に最適経路を設定する。
    - 経路が見つかるまで最大 max_attempts 回試行。
    - 経路が見つからない場合はネットワークを再生成する。

    - start: スタートノード
    - goal: ゴールノード
    - min_pheromone: 最適経路のエッジに設定する初期フェロモン値
    - max_hops: ランダム経路の最大ホップ数
    - max_attempts: 試行回数の上限
    """
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}: Setting optimal path from {start} to {goal}...")

        # ランダム経路の設定
        current_node = start
        path = [current_node]
        visited = set(path)

        # 失敗するごとにランダム経路の最大ホップ数を設定を増加して緩和する．
        max_hops = max_hops + attempt

        for _ in range(max_hops):
            neighbors = list(graph.neighbors(current_node))
            # 訪問済みノードを除外
            neighbors = [n for n in neighbors if n not in visited]

            if not neighbors:
                print(
                    f"No further neighbors from node {current_node}. Stopping path extension."
                )
                break

            # 次のノードをランダムに選択
            next_node = random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)

            # ゴールノードに到達したら終了
            if next_node == goal:
                break

            current_node = next_node

        # 経路がゴールに到達している場合、帯域幅を設定して終了
        if path[-1] == goal:
            print(f"Random path from {start} to {goal}: {path}")
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                graph[u][v]["weight"] = max_weight
                graph[v][u]["weight"] = max_weight
                graph[u][v]["pheromone"] = min_pheromone
                graph[v][u]["pheromone"] = min_pheromone
                graph[u][v]["local_min_bandwidth"] = max_weight
                graph[v][u]["local_min_bandwidth"] = max_weight
                graph[u][v]["local_max_bandwidth"] = max_weight
                graph[v][u]["local_max_bandwidth"] = max_weight
                print(f"Set optimal path edge ({u} → {v}) to weight=100.")
            return graph

        print(f"Path from {start} to {goal} did not reach goal. Retrying...")

    # 最大試行回数を超えた場合
    print(
        f"Failed to find a valid path from {start} to {goal} after {max_attempts} attempts."
    )
    return 0


def add_optimal_path(
    graph: nx.Graph,
    start: int,
    goal: int,
    min_pheromone: int = 100,
    num_intermediate_nodes: int = 5,
) -> nx.Graph:
    """
    最適経路を設定し、帯域幅を100に固定。

    - start: スタートノード
    - goal: ゴールノード
    - min_pheromone: 最適経路のエッジに設定する初期フェロモン値
    - num_intermediate_nodes: 経由する中間ノードの数: ホップ数はnum_intermediate_nodes+1
    """
    num_nodes = len(graph.nodes())
    if num_intermediate_nodes >= num_nodes - 2:
        raise ValueError("中間ノードの数が多すぎます。")

    # スタートノードとゴールノード以外のノードをランダムに選択
    intermediate_nodes = random.sample(
        [i for i in range(num_nodes) if i not in {start, goal}], num_intermediate_nodes
    )

    # 経路のノードを結合
    full_path = [start] + intermediate_nodes + [goal]

    print(f"Generated long path: {full_path}")

    # 経路に基づきエッジを設定
    for u, v in zip(full_path[:-1], full_path[1:]):
        graph.add_edge(
            u,
            v,
            weight=100,
            pheromone=min_pheromone,
            local_min_bandwidth=100,
            local_max_bandwidth=100,
        )
        graph.add_edge(
            v,
            u,
            weight=100,
            pheromone=min_pheromone,
            local_min_bandwidth=100,
            local_max_bandwidth=100,
        )
        print(f"Set optimal path edge ({u} → {v}) to weight=100.")

    return graph


def save_graph(graph: nx.Graph):
    """グラフをファイルに保存"""
    file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".edgelist"
    nx.write_edgelist(graph, file_name, data=["pheromone", "weight"])
    return file_name


def save_graph_without_pheromone(graph: nx.Graph, file_name: str) -> None:
    """
    NetworkX グラフをエッジリスト形式で保存
    """
    with open(file_name, "w") as f:
        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 0)

            f.write(f"{u} {v} {weight}\n")

    print(f"グラフを保存しました: {file_name}")


def save_graph_with_pheromone(graph: nx.Graph, file_name: str) -> None:
    """
    NetworkX グラフをエッジリスト形式でフェロモン情報付きで保存
    フォーマット: source target weight pheromone local_min_bandwidth local_max_bandwidth
    """
    with open(file_name, "w") as f:
        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 0)
            local_min_bandwidth = data.get("local_min_bandwidth")
            local_max_bandwidth = data.get("local_max_bandwidth")
            pheromone = data.get("pheromone")

            f.write(
                f"{u} {v} {weight} {pheromone} {local_min_bandwidth} {local_max_bandwidth}\n"
            )

    print(f"グラフを保存しました: {file_name}")


def visualize_graph(graph: nx.Graph, filename="network_graph.pdf"):
    """グラフをPDFに保存し、エッジの太さを帯域幅に基づいて設定"""
    A = to_agraph(graph)
    for u, v in graph.edges():
        edge = A.get_edge(u, v)
        width = graph[u][v]["weight"]
        edge.attr["penwidth"] = str(width / 20)

    A.layout("fdp")
    A.draw(filename, format="pdf")


# ------------------ メイン処理 ------------------
if __name__ == "__main__":
    # ===== スタートノード切り替えのための設定 =====
    SWITCH_INTERVAL = 200  # スタートノード切り替え間隔
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

        for generation in range(GENERATION):
            # ===== スタートノードの決定 =====
            phase = generation // SWITCH_INTERVAL
            current_start = START_NODE_LIST[phase % len(START_NODE_LIST)]

            # ===== スタートノード切り替え時の初期化処理 =====
            if generation % SWITCH_INTERVAL == 0:
                print(
                    f"\n--- 世代 {generation}: スタートノードを {current_start} に変更 ---"
                )

                # BKB（Best Known Bottleneck）を全ノードでリセット
                print("全ノードのBKBをリセットします。")
                for node in graph.nodes():
                    graph.nodes[node]["best_known_bottleneck"] = 0

                # 新しいスタート/ゴールペアに対する最適解を計算（リトライ処理付き）
                # このstart_nodeは、リトライの結果、変更される可能性がある
                start_node_for_calc = current_start
                retry_count = 0
                used_starts = {current_start}  # 既に使用したスタートノードを記録

                while True:
                    try:
                        optimal_path = max_load_path(
                            graph, start_node_for_calc, GOAL_NODE
                        )
                        optimal_bottleneck = min(
                            graph[optimal_path[i]][optimal_path[i + 1]]["weight"]
                            for i in range(len(optimal_path) - 1)
                        )
                        optimal_bottleneck_dict[current_start] = optimal_bottleneck
                        print(
                            f"最適経路: {optimal_path}, ボトルネック帯域幅: {optimal_bottleneck}"
                        )
                        break  # 成功したらループを抜ける

                    except nx.NetworkXNoPath:
                        retry_count += 1
                        print(
                            f"⚠️ {start_node_for_calc} から {GOAL_NODE} へのパスが存在しません。スタートノードを再設定します。"
                        )

                        # まだ選ばれていないノードからランダムに候補を選択
                        candidates = [
                            n
                            for n in range(NUM_NODES)
                            if n not in used_starts and n != GOAL_NODE
                        ]

                        if not candidates or retry_count > 10:
                            print(
                                "有効なスタートノードが見つかりませんでした。このフェーズをスキップします。"
                            )
                            optimal_bottleneck_dict[current_start] = 0  # 失敗を記録
                            break  # ループを抜ける

                        start_node_for_calc = random.choice(candidates)
                        used_starts.add(start_node_for_calc)
                        # 重要：元のリストも更新して、今後の世代で正しいノードを参照できるようにする
                        START_NODE_LIST[phase % len(START_NODE_LIST)] = (
                            start_node_for_calc
                        )
                        current_start = start_node_for_calc

            # 現在の世代で使用するスタートノードと最適解を取得
            optimal_bottleneck = optimal_bottleneck_dict.get(current_start, 0)
            if optimal_bottleneck == 0:
                continue  # このスタートノードからは到達不能なので探索をスキップ

            # ===== アリの生成と探索 =====
            ants = [
                Ant(current_start, GOAL_NODE, [current_start], [])
                for _ in range(ANT_NUM)
            ]

            temp_ant_list = list(ants)
            while temp_ant_list:
                ant_next_node_const_epsilon(
                    temp_ant_list, graph, ant_log, optimal_bottleneck
                )

            # フェロモンの揮発
            volatilize_by_width(graph)

        # --- 結果の保存 ---
        with open("./simulation_result/log_ant.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        print(f"✅ シミュレーション {sim+1} 回目完了")
