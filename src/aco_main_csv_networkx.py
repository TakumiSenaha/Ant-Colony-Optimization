import csv
import math
import random
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

        graph[u][v]["max_pheromone"] = min(MAX_F, width_u_to_v**3)
        graph[v][u]["max_pheromone"] = min(MAX_F, width_v_to_u**3)


def volatilize_by_width(graph: nx.Graph) -> None:
    """各エッジのフェロモン値を双方向で別々に揮発させる処理"""
    for u, v in graph.edges():
        # uからv、vからuの帯域幅を取得
        width_u_to_v = graph[u][v]["weight"]
        width_v_to_u = graph[v][u]["weight"]

        # フェロモン揮発率の計算
        rate_u_to_v = V * (0.8 ** ((100 - width_u_to_v) / 10))
        rate_v_to_u = V * (0.8 ** ((100 - width_v_to_u) / 10))

        # u→v のフェロモン揮発
        new_pheromone_u_to_v = max(
            math.floor(graph[u][v]["pheromone"] * rate_u_to_v),
            graph[u][v]["min_pheromone"],
        )
        graph[u][v]["pheromone"] = new_pheromone_u_to_v

        # v→u のフェロモン揮発
        new_pheromone_v_to_u = max(
            math.floor(graph[v][u]["pheromone"] * rate_v_to_u),
            graph[v][u]["min_pheromone"],
        )
        graph[v][u]["pheromone"] = new_pheromone_v_to_u


def update_pheromone(ant: Ant, graph: nx.Graph) -> None:
    """Antが通過した経路にフェロモンを双方向に加算"""
    for i in range(1, len(ant.route)):
        u, v = ant.route[i - 1], ant.route[i]
        pheromone_increase = min(ant.width)

        # u→v のフェロモンを更新
        graph[u][v]["pheromone"] = min(
            graph[u][v]["pheromone"] + pheromone_increase, graph[u][v]["max_pheromone"]
        )

        # v→u のフェロモンを別々に更新
        graph[v][u]["pheromone"] = min(
            graph[v][u]["pheromone"] + pheromone_increase, graph[v][u]["max_pheromone"]
        )


def ant_next_node(ant_list: list[Ant], graph: nx.Graph, ant_log: list[int]) -> None:
    """Antの次の移動先を決定し、移動を実行"""
    for ant in reversed(ant_list):
        neighbors = list(graph.neighbors(ant.current))
        candidates = [n for n in neighbors if n not in ant.route]

        # 候補先がないなら削除
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

            # 重みに基づいて次のノードを選択
            next_node = random.choices(candidates, weights=weights, k=1)[0]

            # antのルートと帯域幅を更新
            ant.route.append(next_node)
            ant.width.append(graph[ant.current][next_node]["weight"])
            ant.current = next_node

            # 目的ノードに到達した場合、フェロモンを更新してリストから削除
            if ant.current == ant.destination:
                update_pheromone(ant, graph)
                ant_log.append(min(ant.width))
                ant_list.remove(ant)
                print(f"Ant Goal! → {ant.route} : {min(ant.width)}")

            # TTL（生存時間）を超えた場合もリストから削除
            elif len(ant.route) == TTL:
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


def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """Barabási-Albertモデルでグラフを生成"""
    return nx.barabasi_albert_graph(num_nodes, num_edges)


def make_graph_bidirectional(graph: nx.Graph, lb: int = 1, ub: int = 10) -> nx.Graph:
    """無向グラフを双方向グラフに変換し、双方向の同じ帯域幅とフェロモンを設定"""
    directed_G = nx.DiGraph()

    for u, v in graph.edges():
        weight = random.randint(lb, ub) * 10
        directed_G.add_edge(u, v, weight=weight, pheromone=MIN_F)
        directed_G.add_edge(v, u, weight=weight, pheromone=MIN_F)

    return directed_G


def set_optimal_path(
    graph: nx.Graph, start: int, goal: int, min_pheromone: int = 100
) -> nx.Graph:
    """最適経路を設定し、帯域幅を100に固定"""
    num_nodes = len(graph.nodes())
    node_list = [i for i in range(num_nodes) if i not in {start, goal}]
    node1, node2, node3 = random.sample(node_list, 3)

    for u, v in [(start, node1), (node1, node2), (node2, node3), (node3, goal)]:
        graph.add_edge(u, v, weight=100, pheromone=min_pheromone)
        graph.add_edge(v, u, weight=100, pheromone=min_pheromone)

    return graph


def save_graph(graph: nx.Graph):
    """グラフをファイルに保存"""
    file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".edgelist"
    nx.write_edgelist(graph, file_name, data=["pheromone", "weight"])
    return file_name


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

        # BAモデルでグラフを生成
        graph: nx.Graph = ba_graph(num_nodes, num_edges)

        # グラフを双方向に変換
        graph = make_graph_bidirectional(graph)

        # シミュレーションで使用する開始ノードと終了ノードを決定
        START_NODE: int = random.randint(0, num_nodes - 1)
        GOAL_NODE: int = random.randint(0, num_nodes - 1)

        # 最適経路を追加し、その経路の帯域をすべて100に設定
        graph = set_optimal_path(graph, START_NODE, GOAL_NODE, min_pheromone=MIN_F)

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

        # 各シミュレーションのログをCSVに保存
        with open("ant_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        with open("interest_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(interest_log)

    # 最終的なグラフの視覚化
    visualize_graph(graph, "network_graph.pdf")
    print("Simulations completed.")
