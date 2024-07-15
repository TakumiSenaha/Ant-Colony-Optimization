import math
import random

import imageio
import matplotlib.pyplot as plt
import networkx as nx

# 定数の設定
V = 0.98  # フェロモン揮発量
MIN_F = 100  # フェロモン最小値
MAX_F = 1000000  # フェロモン最大値
TTL = 100  # 蟻のTime to Live
W = 1000  # 帯域幅初期値
ALPHA = 1.0  # フェロモンの重み
BETA = 2.0  # ヒューリスティックの重み
ANT_NUM = 1  # 一回で放つ蟻の数を1に変更
GENERATION = 10  # 蟻を放つ回数(世代)

# ランダムシードの設定
random.seed(42)


# クラス定義
class Node:
    def __init__(self, connections=None, pheromones=None, widths=None):
        if connections is None:
            connections = []
        if pheromones is None:
            pheromones = []
        if widths is None:
            widths = []
        self.connections = connections
        self.pheromones = pheromones
        self.widths = widths
        self.min_pheromone = MIN_F
        self.max_pheromone = [MAX_F for _ in pheromones]


class Ant:
    def __init__(self, start_node, goal_node):
        self.current_node = start_node
        self.goal_node = goal_node
        self.route = [start_node]
        self.complete = False


# ネットワークの設定
def create_manual_network():
    node_list = [Node() for _ in range(10)]

    # ノード間の接続と帯域幅を手動で定義
    connect_node_twoway(node_list, 0, 1, 50, 50)
    connect_node_twoway(node_list, 1, 2, 40, 40)
    connect_node_twoway(node_list, 2, 3, 10, 10)
    connect_node_twoway(node_list, 3, 4, 40, 40)
    connect_node_twoway(node_list, 4, 5, 10, 10)
    connect_node_twoway(node_list, 5, 6, 50, 50)
    connect_node_twoway(node_list, 6, 7, 20, 20)
    connect_node_twoway(node_list, 7, 8, 30, 30)
    connect_node_twoway(node_list, 8, 9, 40, 40)
    connect_node_twoway(node_list, 0, 2, 15, 15)  # 遠回りになる接続を追加
    connect_node_twoway(node_list, 2, 5, 25, 25)  # 遠回りになる接続を追加
    connect_node_twoway(node_list, 1, 3, 30, 30)
    connect_node_twoway(node_list, 3, 5, 60, 60)
    connect_node_twoway(node_list, 5, 7, 25, 25)
    connect_node_twoway(node_list, 7, 9, 35, 35)
    connect_node_twoway(node_list, 0, 4, 10, 10)  # 直接接続を避けるために接続を変更
    connect_node_twoway(node_list, 4, 9, 45, 45)  # 直接接続を避けるために接続を変更

    return node_list


def connect_node_twoway(node_list, index_a, index_b, width_a2b, width_b2a):
    node_list[index_a].connections.append(index_b)
    node_list[index_a].pheromones.append(MIN_F)
    node_list[index_a].widths.append(width_a2b)

    node_list[index_b].connections.append(index_a)
    node_list[index_b].pheromones.append(MIN_F)
    node_list[index_b].widths.append(width_b2a)

    # フェロモンの最大値も設定する
    node_list[index_a].max_pheromone.append(MAX_F)
    node_list[index_b].max_pheromone.append(MAX_F)


# フェロモンの揮発
def volatilize(node_list):
    for node in node_list:
        for i in range(len(node.pheromones)):
            new_pheromone = math.floor(node.pheromones[i] * V)
            if new_pheromone <= node.min_pheromone:
                node.pheromones[i] = node.min_pheromone
            else:
                node.pheromones[i] = new_pheromone


# フェロモンの更新
def update_pheromone(ant, node_list):
    for i in range(1, len(ant.route)):
        before_node = node_list[ant.route[i - 1]]
        after_node = node_list[ant.route[i]]
        index = before_node.connections.index(ant.route[i])
        before_node.pheromones[index] += min(ant.route) * 10
        if before_node.pheromones[index] > before_node.max_pheromone[index]:
            before_node.pheromones[index] = before_node.max_pheromone[index]


# 蟻の次のノードを決定
def ant_next_node(ant_list, node_list, current_generation):
    for ant in reversed(ant_list):
        current_node = node_list[ant.current_node]
        if ant.current_node == ant.goal_node:
            ant.complete = True
            continue

        connections = current_node.connections
        pheromones = current_node.pheromones
        probabilities = calculate_probabilities(pheromones, ALPHA)

        next_node_index = random.choices(
            range(len(connections)), weights=probabilities
        )[0]
        next_node = connections[next_node_index]

        ant.route.append(next_node)
        ant.current_node = next_node

        if next_node == ant.goal_node:
            update_pheromone(ant, node_list)
            ant.complete = True


# フェロモン値に基づいて各候補ノードの選択確率を計算
def calculate_probabilities(pheromones, alpha):
    total = sum(pher**alpha for pher in pheromones)
    probabilities = [(pher**alpha) / total for pher in pheromones]
    return probabilities


# ネットワークの描画（帯域幅のみ表示）
def draw_network_bandwidth(node_list, title=""):
    G = nx.Graph()
    for i, node in enumerate(node_list):
        for conn, width in zip(node.connections, node.widths):
            G.add_edge(i, conn, weight=width)
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        node_color="lightblue",
        font_size=15,
        width=[d["weight"] / 10 for (u, v, d) in G.edges(data=True)],
        edge_color="black",
    )
    plt.title(title)
    plt.show()


# ネットワークの描画（フェロモン濃度を表示）
def draw_network(node_list, title=""):
    G = nx.Graph()
    for i, node in enumerate(node_list):
        for conn, width, pheromone in zip(
            node.connections, node.widths, node.pheromones
        ):
            G.add_edge(i, conn, weight=width, pheromone=pheromone)
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    widths = [d["weight"] / 10 for (u, v, d) in edges]
    pheromones = [d["pheromone"] for (u, v, d) in edges]
    max_pheromone = max(pheromones) if pheromones else 1
    colors = [plt.cm.Reds(pheromone / max_pheromone) for pheromone in pheromones]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        node_color="lightblue",
        font_size=15,
        width=widths,
        edge_color=colors,
    )
    plt.title(title)
    plt.show()


# 動画の生成
def create_gif(node_list, all_routes, filename="aco_search.gif"):
    images = []
    for i, routes in enumerate(all_routes):
        plt.figure()
        draw_network(node_list, title=f"Generation {i+1}")
        images.append(imageio.imread(f"frame_{i}.png"))
    imageio.mimsave(filename, images, duration=0.5)


# メイン処理
def main():
    node_list = create_manual_network()

    # 初期ネットワークの表示
    draw_network_bandwidth(node_list, title="Initial Network with Bandwidth")

    ants = [Ant(0, 9) for _ in range(10)]

    all_routes = []

    for gen in range(GENERATION):
        print(f"Generation {gen + 1}")
        ant_next_node(ants, node_list, gen)
        routes = [ant.route for ant in ants if ant.complete]
        all_routes.append(routes)
        for ant in ants:
            if ant.complete:
                print(f"Ant completed the route: {ant.route}")
            else:
                print(f"Ant current route: {ant.route}")
        volatilize(node_list)

        # 各世代ごとのネットワークの状態を表示
        draw_network(node_list, title=f"Generation {gen + 1}")

    # 最終ネットワークの表示
    draw_network(node_list, title="Final Network with Pheromones")

    # GIFの作成
    create_gif(node_list, all_routes)


if __name__ == "__main__":
    main()
