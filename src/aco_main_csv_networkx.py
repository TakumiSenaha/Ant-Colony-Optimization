import csv
import math
import random
import sys
from datetime import datetime

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

V = 0.99  # フェロモン揮発量
MIN_F = 100  # フェロモン最小値
MAX_F = 1000000000  # フェロモン最大値
TTL = 100  # AntのTime to Live
W = 1000  # 帯域幅初期値
BETA = 1  # 経路選択の際のヒューリスティック値に対する重み(累乗)

ANT_NUM = 1  # 一回で放つAntの数
GENERATION = 500  # ant，interestを放つ回数(世代)
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
VOLATILIZATION_MODE = 1


def volatilize_by_width(graph: nx.Graph) -> None:
    """
    各エッジのフェロモン値を双方向で揮発させる
    - VOLATILIZATION_MODE が 0 の場合: 既存の揮発式を使用
    - VOLATILIZATION_MODE が 1 の場合: 帯域幅の最小値・最大値を基準に揮発量を調整
    - VOLATILIZATION_MODE が 2 の場合: 平均・分散を基準に揮発量を計算
    """
    for u, v in graph.edges():
        # u → v の揮発計算
        _apply_volatilization(graph, u, v)
        # v → u の揮発計算
        _apply_volatilization(graph, v, u)


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
        rate = V * (0.8 ** ((100 - weight_uv) / 10))

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

    else:
        raise ValueError("Invalid VOLATILIZATION_MODE. Choose 0, 1, or 2.")

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


def calculate_pheromone_increase(
    bottleneck_bandwidth: int,
    local_min_bandwidth: int,
    local_max_bandwidth: int,
    fixed_min_value: int = 10,
) -> float:
    """
    フェロモン付加量を計算する
    - local_min_bandwidth と local_max_bandwidth に応じて調整
    """
    if bottleneck_bandwidth == local_max_bandwidth == local_min_bandwidth:
        pheromone_increase = bottleneck_bandwidth

    elif bottleneck_bandwidth == local_max_bandwidth:
        pheromone_increase = bottleneck_bandwidth**3
        # if bottleneck_bandwidth != 100:
        #     # エラーを吐く
        #     print(bottleneck_bandwidth, local_min_bandwidth, local_max_bandwidth)
        #     raise RuntimeError(
        #         f"Error: bottleneck_bandwidth ({bottleneck_bandwidth}) != 100"
        #     )
    elif local_min_bandwidth < local_max_bandwidth:
        # local_min_bandwidth を引く
        pheromone_increase = (bottleneck_bandwidth - local_min_bandwidth) * 10
    else:
        pheromone_increase = bottleneck_bandwidth

    return max(0, pheromone_increase)  # 負の値を防ぐために max(0) を適用


def update_pheromone(ant: Ant, graph: nx.Graph) -> None:
    """
    Antがゴールに到達したとき、通過した経路のフェロモン値と帯域幅情報を更新する
    - フェロモン値はボトルネック帯域幅 (min(ant.width)) に基づき加算
    - エッジごとの既知の最小・最大帯域幅 (max(ant.width), local_max_bandwidth) を更新
    """
    for i in range(1, len(ant.route)):
        u, v = ant.route[i - 1], ant.route[i]
        # pheromone_increase = min(ant.width) ** 2
        # pheromone_increase = math.exp(min(ant.width) / 10)
        # u→v のフェロモンを更新（通った方向のみ）

        # エッジが知り得た最小帯域幅を更新（ローカル情報が小さければ更新）
        graph[u][v]["local_min_bandwidth"] = min(
            graph[u][v]["local_min_bandwidth"],
            min(ant.width),
        )

        # エッジが知り得た最大帯域幅を更新（ローカル情報が大きければ更新）
        graph[u][v]["local_max_bandwidth"] = max(
            graph[u][v]["local_max_bandwidth"], max(ant.width)
        )

        pheromone_increase = calculate_pheromone_increase(
            bottleneck_bandwidth=min(ant.width),
            local_min_bandwidth=graph[u][v]["local_min_bandwidth"],
            local_max_bandwidth=graph[u][v]["local_max_bandwidth"],
        )

        # pheromone_increase = math.exp(min(ant.width) / 10)
        # pheromone_increase = min(ant.width) * 10

        graph[u][v]["pheromone"] = min(
            graph[u][v]["pheromone"] + pheromone_increase, graph[u][v]["max_pheromone"]
        )

        # print(f"Update Pheromone: {u} → {v} : {graph[u][v]['pheromone']}")
        # print(
        #     f"Update Bandwidth: {u} → {v} : {graph[u][v]['local_min_bandwidth']} : {graph[u][v]['local_max_bandwidth']}"
        # )


def ant_next_node(ant_list: list[Ant], graph: nx.Graph, ant_log: list[int]) -> None:
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
                ant_log.append(min(ant.width))
                ant_list.remove(ant)
                print(f"Ant Goal! → {ant.route} : {min(ant.width)}")

                bottleneck_bandwidth = min(ant.width)
                if bottleneck_bandwidth == 100 or bottleneck_bandwidth == 90:
                    print(
                        f"ボトルネック帯域が{bottleneck_bandwidth}の経路: {ant.route}"
                    )

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
    # 読み込んだグラフのエッジに初期フェロモン値を追加
    for u, v in graph.edges():
        graph[u][v]["pheromone"] = MIN_F
        graph[u][v]["local_min_bandwidth"] = graph[u][v]["weight"]
        graph[u][v]["local_max_bandwidth"] = graph[u][v]["weight"]

    return graph


def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """
    Barabási-Albertモデルでグラフを生成
    - 各エッジに帯域幅(weight)をランダムに設定
    - 各エッジに local_min_bandwidth と local_max_bandwidth を初期化
    """
    graph = nx.barabasi_albert_graph(num_nodes, num_edges)
    for u, v in graph.edges():
        # リンクの帯域幅(weight)をランダムに設定
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight

        # 各エッジが知り得ている最小・最大帯域幅を初期化
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
    max_hops: int = 2,
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


# Main処理
if __name__ == "__main__":
    for sim in range(SIMULATIONS):
        num_nodes = 100  # ノードの数
        num_edges = 3  # 新しいノードが既存ノードに接続する数

        # シミュレーションで使用する開始ノードと終了ノード
        START_NODE: int
        GOAL_NODE: int

        use_existing_graph = False  # 既存のグラフを使用するかどうか

        if not use_existing_graph:
            # BAモデルでグラフを生成
            graph: nx.Graph = ba_graph(num_nodes, num_edges)
            # グラフを双方向に変換
            graph = make_graph_bidirectional(graph)

            # シミュレーションで使用する開始ノードと終了ノードを決定
            while True:
                START_NODE = random.randint(0, num_nodes - 1)
                GOAL_NODE = random.randint(0, num_nodes - 1)
                if START_NODE != GOAL_NODE:
                    break

            # # 最適経路を追加し、その経路の帯域をすべて100に設定
            # graph = add_optimal_path(
            #     graph,
            #     START_NODE,
            #     GOAL_NODE,
            #     min_pheromone=MIN_F,
            #     num_intermediate_nodes=6,
            # )
            graph = set_optimal_path(graph, START_NODE, GOAL_NODE)
            # graph = set_optimal_path(graph, next_start_node, GOAL_NODE, min_pheromone=MIN_F)
            while graph == 0:
                graph = ba_graph(num_nodes, num_edges)
                graph = make_graph_bidirectional(graph)
                graph = set_optimal_path(graph, START_NODE, GOAL_NODE)

        else:
            graph = load_graph("ba_model_graph")
            # グラフを双方向に変換
            graph = make_graph_bidirectional(graph)

            START_NODE = 30
            GOAL_NODE = 32

        # next_start_node = random.randint(0, num_nodes - 1)

        # ノードの隣接数と帯域幅に基づいてフェロモンの最小値・最大値を設定
        set_pheromone_min_max_by_degree_and_width(graph)

        # AntとInterestオブジェクト格納リスト
        ant_list: list[Ant] = []
        interest_list: list[Interest] = []

        # ログのリスト
        ant_log: list[int] = []
        interest_log: list[int] = []

        for generation in range(GENERATION):
            print(f"Simulation {sim+1}, Generation {generation+1}")

            # Antを配置
            ant_list.extend(
                [Ant(START_NODE, GOAL_NODE, [START_NODE], []) for _ in range(ANT_NUM)]
            )

            # Antによる探索
            for _ in range(TTL):
                ant_next_node(ant_list, graph, ant_log)

            # フェロモンの揮発
            volatilize_by_width(graph)

            # Interestによる評価
            # Interestを配置
            interest_list.append(Interest(START_NODE, GOAL_NODE, [START_NODE], W))

            # Interestの移動
            for _ in range(TTL):
                interest_next_node(interest_list, graph, interest_log)

        # save_graph_without_pheromone(graph, "ba_model_graph")
        save_graph_with_pheromone(graph, "ba_model_graph_with_pheromone")

        # 各シミュレーションのログをCSVに保存
        with open("./simulation_result/log_ant.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        with open("./simulation_result/log_interest.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(interest_log)

    # 最終的なグラフの視覚化
    # visualize_graph(graph, "network_graph.pdf")
    print("Simulations completed.")
    print(START_NODE, GOAL_NODE)
