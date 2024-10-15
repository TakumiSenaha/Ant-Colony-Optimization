import csv
import math
import random
from datetime import datetime

import networkx as nx

V = 0.99  # フェロモン揮発量
MIN_F = 100  # フェロモン最小値
MAX_F = 1000000000  # フェロモン最大値
TTL = 100  # AntのTime to Live
W = 1000  # 帯域幅初期値
BETA = 1  # 経路選択の際のヒューリスティック値に対する重み(累乗)

ANT_NUM = 1  # 一回で放つAntの数
GENERATION = 1000  # ant，interestを放つ回数(世代)


class Ant:
    def __init__(
        self, current: int, destination: int, route: list[int], width: list[int]
    ):
        self.current = current  # 現在のノード
        self.destination = destination  # コンテンツ保持ノード
        self.route = route  # 辿ってきた経路の配列
        self.width = width  # 辿ってきた経路の帯域の配列


class Interest:
    def __init__(self, current: int, destination: int, route: list[int], minwidth: int):
        self.current = current  # 現在のノード
        self.destination = destination  # コンテンツ保持ノード
        self.route = route  # 辿ってきた経路の配列
        self.minwidth = minwidth  # 辿ってきた経路の最小帯域


def volatilize_by_width(graph: nx.Graph) -> None:
    """各エッジのフェロモン値を帯域の大きさによって定数倍して揮発させる処理"""
    for u, v in graph.edges():
        width = graph[u][v]["weight"]  # 帯域幅
        pheromone = graph[u][v]["pheromone"]  # フェロモン量

        # 帯域幅に基づいた揮発レートの計算
        rate = 0.99 * (0.8 ** ((100 - width) / 10))
        new_pheromone = math.floor(pheromone * rate)

        # フェロモン量が最小値より小さくならないように調整
        graph[u][v]["pheromone"] = max(new_pheromone, MIN_F)


def update_pheromone(ant: Ant, graph: nx.Graph) -> None:
    """Antが通過した経路にフェロモンを加算"""
    for i in range(1, len(ant.route)):
        u = ant.route[i - 1]
        v = ant.route[i]
        graph[u][v]["pheromone"] += min(ant.width)
        if graph[u][v]["pheromone"] > MAX_F:
            graph[u][v]["pheromone"] = MAX_F


def ant_next_node(ant_list: list[Ant], graph: nx.Graph, ant_log: list[int]) -> None:
    """Antが次のノードを選択し移動"""
    for ant in reversed(ant_list):
        neighbors = list(graph.neighbors(ant.current))
        candidates = [n for n in neighbors if n not in ant.route]

        if not candidates:
            ant_list.remove(ant)
            print(f"Ant Can't Find Route! → {ant.route}")
        else:
            pheromones = [graph[ant.current][n]["pheromone"] for n in candidates]
            widths = [graph[ant.current][n]["weight"] for n in candidates]
            weight_width = [w**BETA for w in widths]
            weights = [p * w for p, w in zip(pheromones, weight_width)]

            next_node = random.choices(candidates, weights=weights, k=1)[0]

            # エッジが存在するかのチェックと双方向アクセスを追加
            if graph.has_edge(ant.current, next_node):
                ant.current = next_node
                ant.route.append(next_node)

                # エッジ属性にアクセスする際に、双方向対応にする
                if "weight" in graph[ant.route[-2]][ant.route[-1]]:
                    ant.width.append(graph[ant.route[-2]][ant.route[-1]]["weight"])
                elif "weight" in graph[ant.route[-1]][ant.route[-2]]:  # 逆方向も確認
                    ant.width.append(graph[ant.route[-1]][ant.route[-2]]["weight"])
                else:
                    print(
                        f"Warning: No weight data between {ant.route[-2]} and {ant.route[-1]}"
                    )
            else:
                print(f"Error: No edge between {ant.current} and {next_node}")

            if ant.current == ant.destination:
                update_pheromone(ant, graph)
                ant_log.append(min(ant.width))
                ant_list.remove(ant)
                print(f"Ant Goal! → {ant.route} : {min(ant.width)}")
            elif len(ant.route) == TTL:
                ant_log.append(0)
                ant_list.remove(ant)
                print(f"Ant TTL! → {ant.route}")


def interest_next_node(
    interest_list: list[Interest], graph: nx.Graph, interest_log: list[int]
) -> None:
    """Interestが次のノードを選択し移動"""
    for interest in reversed(interest_list):
        neighbors = list(graph.neighbors(interest.current))
        candidates = [n for n in neighbors if n not in interest.route]

        if not candidates:
            interest_list.remove(interest)
            interest_log.append(0)
            print(f"Interest Can't Find Route! → {interest.route}")
        else:
            # 各候補ノードに対して帯域幅を取得（双方向対応）
            widths = []
            for n in candidates:
                if "weight" in graph[interest.current][n]:
                    widths.append(graph[interest.current][n]["weight"])
                elif "weight" in graph[n][interest.current]:  # 逆方向も確認
                    widths.append(graph[n][interest.current]["weight"])
                else:
                    print(f"Warning: No weight data between {interest.current} and {n}")
                    widths.append(0)  # デフォルトで 0 を追加

            # 最大の帯域幅を持つノードを選択
            next_node = candidates[widths.index(max(widths))]

            interest.current = next_node
            interest.route.append(next_node)

            # 帯域幅の最小値を更新（双方向確認）
            if "weight" in graph[interest.route[-2]][interest.route[-1]]:
                interest.minwidth = min(
                    interest.minwidth,
                    graph[interest.route[-2]][interest.route[-1]]["weight"],
                )
            elif (
                "weight" in graph[interest.route[-1]][interest.route[-2]]
            ):  # 逆方向も確認
                interest.minwidth = min(
                    interest.minwidth,
                    graph[interest.route[-1]][interest.route[-2]]["weight"],
                )

            # 目的ノードに到達した場合
            if interest.current == interest.destination:
                interest_log.append(interest.minwidth)
                interest_list.remove(interest)
                print(f"Interest Goal! → {interest.route} : {interest.minwidth}")
            elif len(interest.route) == TTL:
                interest_log.append(0)
                interest_list.remove(interest)
                print(f"Interest TTL! → {interest.route}")


def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """NetworkXを用いてBAグラフを作成"""
    G = nx.barabasi_albert_graph(num_nodes, num_edges)
    for u, v in G.edges():
        G[u][v]["pheromone"] = MIN_F  # フェロモン初期値
        G[u][v]["weight"] = random.randint(lb, ub) * 10  # 帯域幅
    return G


def save_graph(graph: nx.Graph):
    """グラフをファイルに保存"""
    file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".edgelist"
    nx.write_edgelist(graph, file_name, data=["pheromone", "weight"])
    return file_name


# Main処理
if __name__ == "__main__":
    num_nodes = 100
    num_edges = 3
    graph = ba_graph(num_nodes, num_edges)

    START_NODE = random.randint(0, num_nodes - 1)
    GOAL_NODE = random.randint(0, num_nodes - 1)

    ant_list = [Ant(START_NODE, GOAL_NODE, [START_NODE], [])]
    interest_list = [Interest(START_NODE, GOAL_NODE, [START_NODE], W)]
    ant_log = []
    interest_log = []

    for generation in range(GENERATION):
        print(f"ant_log {ant_log}")

        # Antによる探索
        ant_next_node(ant_list, graph, ant_log)

        # Interestによる評価
        interest_next_node(interest_list, graph, interest_log)

        # フェロモンの揮発
        volatilize_by_width(graph)

    print("Simulation finished.")

    # 結果の保存
    with open("ant_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ant_log)

    with open("interest_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(interest_log)

    save_graph(graph)
