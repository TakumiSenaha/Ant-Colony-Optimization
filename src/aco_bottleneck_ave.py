# baモデル
import csv
import math
import random
import secrets

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

V = 0.98  # フェロモン揮発量
MIN_F = 100  # フェロモン最小値
MAX_F = 1000000  # フェロモン最大値
TTL = 100  # antのTime to Live
W = 1000  # 帯域幅初期値
ALPHA = 1.0  # フェロモンの重み
BETA = 2.0  # ヒューリスティックの重み

ANT_NUM = 1  # 一回で放つAntの数
GENERATION = 400  # ant，interestを放つ回数(世代)

# /////////////////////////////////////////////////クラス定義/////////////////////////////////////////////////


class Node:
    def __init__(self, connection: list[int], pheromone: list[int], width: list[int]):
        self.connection = connection  # 接続先ノードの配列
        self.pheromone = pheromone  # 接続先ノードとのフェロモンの配列
        self.width = width  # 接続先ノードとの帯域の配列
        self.min_pheromone = MIN_F  # フェロモン最小値
        self.max_pheromone = [MAX_F for _ in pheromone]  # フェロモン最大値


class Ant:
    def __init__(
        self,
        current: int,
        destination: int,
        route: list[int],
        width: list[int],
        strategy: str,
    ):
        self.current = current  # 現在のノード
        self.destination = destination  # コンテンツ保持ノード
        self.route = route  # 辿ってきた経路の配列
        self.width = width  # 辿ってきた経路の帯域の配列
        self.strategy = strategy  # 蟻の探索戦略


class Interest:
    def __init__(self, current: int, destination: int, route: list[int], minwidth: int):
        self.current = current  # 現在のノード
        self.destination = destination  # コンテンツ保持ノード
        self.route = route  # 辿ってきた経路の配列
        self.minwidth = minwidth  # 辿ってきた経路の最小帯域


class Rand(Interest):
    def __init__(self, current: int, destination: int, route: list[int], minwidth: int):
        super().__init__(current, destination, route, minwidth)
        self.best_routes: list[tuple[list[int], int]] = (
            []
        )  # 最高経路と帯域幅のタプルリスト


# /////////////////////////////////////////////////関数定義/////////////////////////////////////////////////


# 戦略に応じた蟻の生成
def generate_ants(start_node, goal_node, generation, total_generation):
    ants = []
    choices = ["random"] * 45 + ["width"] * 45 + ["pheromone"] * 10
    for _ in range(ANT_NUM):
        if generation < total_generation * 0.07:
            strategy = secrets.choice(choices)
        else:
            strategy = "adaptive"
        ants.append(Ant(start_node, goal_node, [start_node], [], strategy))
    return ants


def volatilize(node_list: list[Node]) -> None:
    # node_listの全nodeのフェロモンをV倍する関数(フェロモンの揮発に相当)
    for node in node_list:
        for i in range(len(node.pheromone)):
            new_pheronone = math.floor(node.pheromone[i] * V)
            if new_pheronone <= node.min_pheromone:
                node.pheromone[i] = node.min_pheromone
            else:
                node.pheromone[i] = new_pheronone


def volatilize_by_width(node_list: list[Node]) -> None:
    # 各エッジのフェロモン値を帯域の大きさによって定数倍する関数(フェロモンの揮発に相当)
    for node in node_list:
        for i in range(len(node.pheromone)):
            # 揮発量はwidth100のとき0.99、width10のとき0.91
            # rate = 0.89 + node.width[i] / 1000
            rate = 0.99 * (0.8 ** ((100 - node.width[i]) / 10))
            # rate = 0.99
            new_pheromone = math.floor(node.pheromone[i] * rate)
            if new_pheromone <= node.min_pheromone:
                node.pheromone[i] = node.min_pheromone
            else:
                node.pheromone[i] = new_pheromone


def update_pheromone(ant: Ant, node_list: list[Node]) -> None:
    # 目的ノードに到着したantによるフェロモンの付加(片側)
    for i in range(1, len(ant.route)):
        # ant.routeのi-1番目とi番目のノードを取得
        before_node: Node = node_list[ant.route[i - 1]]
        after_node: Node = node_list[ant.route[i]]
        # before_nodeからafter_nodeへのフェロモン値を変更
        index = before_node.connection.index(ant.route[i])

        # インデックスエラーをチェック
        if index >= len(before_node.pheromone) or index >= len(
            before_node.max_pheromone
        ):
            print(f"Error: Index {index} out of range for node {ant.route[i - 1]}")
            print(f"Connections: {before_node.connection}")
            print(f"Pheromones: {before_node.pheromone}")
            print(f"Max Pheromones: {before_node.max_pheromone}")
            continue

        # フェロモン値を更新
        before_node.pheromone[index] += min(ant.width) * 10
        if before_node.pheromone[index] > before_node.max_pheromone[index]:
            before_node.pheromone[index] = before_node.max_pheromone[index]


def calculate_probabilities(pheromones, alpha):
    """
    フェロモン値に基づいて各候補ノードの選択確率を計算する関数

    :param pheromones: 候補ノードのフェロモン値のリスト
    :param alpha: フェロモンの重み
    :return: フェロモン値に基づいた確率のリスト
    """
    # フェロモン値のα乗の総和を計算
    total = sum(pher**alpha for pher in pheromones)
    # 各フェロモン値のα乗を総和で割って確率を計算
    probabilities = [(pher**alpha) / total for pher in pheromones]
    return probabilities


def update_alpha(current_gen, total_gen, base_alpha=1.0, max_alpha=5.0):
    """
    現在の世代に応じてalpha値を更新する関数。
    alpha値は時間とともに増加し、フェロモンの影響を強める。
    """
    alpha_range = max_alpha - base_alpha
    alpha = base_alpha + (alpha_range * (current_gen / total_gen))
    return alpha


def weighted_choice(choices, weights):
    """
    重み付けされた選択を行うカスタム関数。

    :param choices: 選択肢のリスト
    :param weights: 重みのリスト
    :return: 選択された要素
    """
    scaled_weights = [int(weight * 100) for weight in weights]
    total = sum(scaled_weights)
    cum_weights = [sum(scaled_weights[: i + 1]) for i in range(len(scaled_weights))]
    x = secrets.randbelow(total)

    for choice, cum_weight in zip(choices, cum_weights):
        if x < cum_weight:
            return choice


def ant_next_node(
    ant_list: list[Ant],
    node_list: list[Node],
    current_generation: int,
    ant_log: list[int],
) -> None:
    # antの次のノードを決定
    # 繰り返し中にリストから削除を行うためreversed
    for ant in reversed(ant_list):
        # antが今いるノードの接続ノード・フェロモン値・帯域幅を取得
        connection: list[int] = node_list[ant.current].connection
        pheromone: list[int] = node_list[ant.current].pheromone
        width: list[int] = node_list[ant.current].width

        # 接続ノードの内、antが辿っていないノード番号を取得
        and_set = set(ant.route) & set(connection)
        diff_list = list(set(connection) ^ and_set)

        # 候補先がないなら削除
        if diff_list == []:
            ant_list.remove(ant)
            print("Ant Can't Find Route! → " + str(ant.route))

        # 候補先がある場合
        else:
            # 候補先のフェロモンと帯域幅を取得
            candidacy_pheromones: list[int] = []
            candidacy_width: list[int] = []
            for i in diff_list:
                index = connection.index(i)
                candidacy_pheromones.append(pheromone[index])
                candidacy_width.append(width[index])

            if ant.strategy == "random":
                next_node = secrets.choice(diff_list)
            elif ant.strategy == "width":
                max_width_index = [
                    i
                    for i, w in enumerate(candidacy_width)
                    if w == max(candidacy_width)
                ]
                next_node = diff_list[secrets.choice(max_width_index)]
            elif ant.strategy == "pheromone":
                max_pheromone_index = [
                    i
                    for i, x in enumerate(candidacy_pheromones)
                    if x == max(candidacy_pheromones)
                ]
                next_node = diff_list[secrets.choice(max_pheromone_index)]
            else:  # フェロモン重視の戦略
                dynamic_alpha = update_alpha(current_generation, GENERATION)
                total_pheromone = sum(candidacy_pheromones)
                probabilities = [(p) / total_pheromone for p in candidacy_pheromones]
                next_node = weighted_choice(diff_list, probabilities)

            ant.current = next_node
            ant.route.append(next_node)
            ant.width.append(width[connection.index(next_node)])

            # antが目的ノードならばノードにフェロモンの付加後ant_listから削除
            if ant.current == ant.destination:
                update_pheromone(ant, node_list)
                ant_log.append(min(ant.width))
                ant_list.remove(ant)
                print("Ant Goal! → " + str(ant.route) + " : " + str(min(ant.width)))

            # antがTTLならばant_listから削除
            elif len(ant.route) == TTL:
                ant_list.remove(ant)
                print("Ant TTL! → " + str(ant.route))


def interest_next_node(
    interest_list: list[Interest], node_list: list[Node], interest_log: list[int]
) -> None:
    # interestの次のノードを決定
    # 繰り返し中にリストから削除を行うためreversed
    for interest in reversed(interest_list):
        # interestが今いるノードの接続ノードとフェロモン値を取得
        connection = node_list[interest.current].connection
        pheromone = node_list[interest.current].pheromone
        width = node_list[interest.current].width

        # 接続ノードの内、interestが辿っていないノード番号を取得
        and_set = set(interest.route) & set(connection)
        diff_list = list(set(connection) ^ and_set)

        # 候補先がないなら削除
        if diff_list == []:
            interest_list.remove(interest)
            interest_log.append(0)
            print("Interest Can't Find Route! → " + str(interest.route))

        # 候補先がある場合
        else:
            candidacy_pheromones: list[int] = []
            # interestが辿っていないノード番号(diff_list)のフェロモンを取得
            for i in diff_list:
                index = connection.index(i)
                candidacy_pheromones.append(pheromone[index])

            # フェロモン濃度が最も高いものを選択(最大値が複数ある場合はランダム)
            max_pheromone_index = [
                i
                for i, x in enumerate(candidacy_pheromones)
                if x == max(candidacy_pheromones)
            ]
            next_node = diff_list[random.choice(max_pheromone_index)]

            # interestの属性更新
            # もし現在ノードから次ノードの帯域幅が今までの最小帯域より小さかったら更新
            interest.current = next_node
            interest.route.append(next_node)
            if width[connection.index(next_node)] < interest.minwidth:
                interest.minwidth = width[connection.index(next_node)]

            # interestが目的ノードならばinterest_listから削除
            if interest.current == interest.destination:
                interest_log.append(interest.minwidth)
                interest_list.remove(interest)
                print(
                    "Interest Goal! → "
                    + str(interest.route)
                    + " : "
                    + str(interest.minwidth)
                )

            # interestがTTLならばinterest_listから削除
            elif len(interest.route) == TTL:
                interest_list.remove(interest)
                interest_log.append(0)
                print("Interest TTL! →" + str(interest.route))


def rand_next_node(
    rand_list: list[Rand],
    node_list: list[Node],
    rand_log: list[int],
    best_routes: list[tuple[list[int], int]],
) -> None:
    # randの次のノードを決定
    # 繰り返し中にリストから削除を行うためreversed
    for rand in reversed(rand_list):
        # randが今いるノードの接続ノードとフェロモン値を取得
        connection = node_list[rand.current].connection
        width = node_list[rand.current].width

        # 接続ノードの内、randが辿っていないノード番号を取得
        and_set = set(rand.route) & set(connection)
        diff_list = list(set(connection) ^ and_set)

        # 候補先がないなら削除
        if diff_list == []:
            rand_list.remove(rand)
            rand_log.append(0)
            if max(rand_log) != 0:
                if not best_routes:
                    rand_log[-1] = 0
                else:
                    rand_log[-1] = max(best_routes, key=lambda x: x[1])[1]
            print("Rand Can't Find Route! → " + str(rand.route))

        # 候補先がある場合
        else:
            next_node = secrets.choice(diff_list)

            # randの属性更新
            # もし現在ノードから次ノードの帯域幅が今までの最小帯域より小さかったら更新
            rand.current = next_node
            rand.route.append(next_node)
            if width[connection.index(next_node)] < rand.minwidth:
                rand.minwidth = width[connection.index(next_node)]

            # randが目的ノードならばrand_listから削除
            if rand.current == rand.destination:
                rand_log.append(rand.minwidth)
                # best_routesが未定義または空の場合のチェック
                if not best_routes:
                    print(f"not best_routes {rand.minwidth}")
                    # rand_logの最後の要素をrand.minwidthで更新
                    rand_log[-1] = rand.minwidth
                    # 新しいベストルートを追加
                    best_routes.append((rand.route.copy(), rand.minwidth))
                    print(rand_log)
                else:
                    max_minwidth_in_best_routes = max(best_routes, key=lambda x: x[1])[
                        1
                    ]
                    if max_minwidth_in_best_routes > rand.minwidth:
                        rand_log[-1] = max_minwidth_in_best_routes
                    else:
                        best_routes.append((rand.route.copy(), rand.minwidth))
                        rand_log[-1] = rand.minwidth
                rand_list.remove(rand)
                print("Rand Goal! → " + str(rand.route) + " : " + str(rand.minwidth))

            # randがTTLならばrand_listから削除
            elif len(rand.route) == TTL:
                rand_list.remove(rand)
                rand_log.append(0)
                if max(rand_log) != 0:
                    rand_log[-1] = max(best_routes, key=lambda x: x[1])[1]
                print("Rand TTL! →" + str(rand.route))


def show_node_info(node_list: list[Node]) -> None:
    for i in range(len(node_list)):
        print("Node" + str(i))
        print(str(node_list[i].connection))
        print(str(node_list[i].pheromone))
        print(str(node_list[i].width))


def connect_node_oneway(
    node_list: list[Node], index_a: int, index_b: int, width: int
) -> None:
    node_list[index_a].connection.append(index_b)
    node_list[index_a].pheromone.append(MIN_F)
    node_list[index_a].width.append(width)


def connect_node_twoway(
    node_list: list[Node], index_a: int, index_b: int, width_a2b: int, width_b2a: int
) -> None:
    node_list[index_a].connection.append(index_b)
    node_list[index_a].pheromone.append(MIN_F)
    node_list[index_a].width.append(width_a2b)

    node_list[index_b].connection.append(index_a)
    node_list[index_b].pheromone.append(MIN_F)
    node_list[index_b].width.append(width_b2a)


def ba_model(edge_num: int, node_num: int) -> list[Node]:
    # 初期ノードを準備してそれぞれ連結させる
    node_list = [Node([], [], []) for _ in range(3)]
    connect_node_twoway(
        node_list, 0, 1, random.randint(1, 10) * 10, random.randint(1, 10) * 10
    )
    connect_node_twoway(
        node_list, 1, 2, random.randint(1, 10) * 10, random.randint(1, 10) * 10
    )
    connect_node_twoway(
        node_list, 2, 0, random.randint(1, 10) * 10, random.randint(1, 10) * 10
    )
    node_degree: list[int] = [2, 2, 2]

    # 所定の数になるまでノードを1つずつ追加
    for _ in range(node_num - 3):
        # 1つのノードを追加する
        target: list[int] = []
        node_num: list[int] = list(range(len(node_list)))
        weight: list[int] = node_degree.copy()

        # 新規ノードは3つのノードと接続
        for _ in range(edge_num):
            index = random.choices(range(len(node_num)), weight)[0]
            target.append(node_num.pop(index))
            weight.pop(index)

        node_list.append(Node([], [], []))

        for i in target:
            connect_node_twoway(
                node_list,
                len(node_list) - 1,
                i,
                random.randint(1, 10) * 10,
                random.randint(1, 10) * 10,
            )
            node_degree[i] += 1

        node_degree.append(len(target))

    # print(node_degree) # debug
    return node_list


def make_way(start: int, hop: int):
    current_node: int = start
    route: list[int] = [current_node]

    for _ in range(hop):

        connection: list[int] = node_list[current_node].connection

        diff_list: list[int] = list(set(connection) ^ (set(route) & set(connection)))

        if diff_list == []:
            return 0

        next_node: int = random.choice(diff_list)

        route.append(next_node)

        current_node = next_node

    return route


def node2edge(node_list):
    # node_listからネットワークのグラフ表示のためのedgeのリストedgesを返す
    edges = []
    for i in range(len(node_list)):
        line0 = node_list[i].connection  # i番目ノードの接続先を取得
        line1 = node_list[i].pheromone  # i番目ノードのフェロモン値を取得
        line2 = node_list[i].width  # i番目ノードの帯域幅を取得

        sum_line1 = sum(line1)

        for j in range(len(line0)):
            # 色はフェロモン量の絶対値ではなく、そのノードのフェロモン総和との相対値で決定
            # 太さは帯域÷20
            edge = (
                i,
                line0[j],
                {
                    "minlen": "5.0",
                    "label": str(line2[j]) + ":" + str(line1[j]),
                    "color": "0.000 " + str(round(line1[j] / sum_line1, 2)) + " 1.000",
                    "penwidth": str(line2[j] / 20),
                    "fontsize": "8",
                },
            )
            edges.append(edge)

    return edges


def visualize_graph(node_list):
    nodes = [i for i in range(len(node_list))]
    edges = node2edge(node_list)

    g = nx.MultiDiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    agraph = nx.nx_agraph.to_agraph(g)
    agraph.node_attr["shape"] = "circle"
    agraph.draw("./simulation_result/ba_model_sample.pdf", prog="fdp", format="pdf")


def set_node_min_pheromone_by_degree(node_list: list[Node]) -> None:
    # 各nodeのmin_pheromone属性の値を決定
    for node in node_list:
        degree = len(node.connection)
        node.min_pheromone = MIN_F * 3 // degree

        # pheromoneの値をnode.min_pheromoneに変更
        node.pheromone = [node.min_pheromone for _ in node.pheromone]


def set_node_max_pheromone_by_width(node_list: list[Node]) -> None:
    # 各nodeのmax_pheromone属性の値を帯域幅に基づいて決定
    for node in node_list:
        node.max_pheromone = [width**3 for width in node.width]


def set_node_min_pheromon_uniformly(node_list: list[Node]) -> None:
    for node in node_list:
        node.min_pheromone = MIN_F


def validate_best_routes(
    node_list: list[Node], best_routes: list[tuple[list[int], int]]
) -> list[tuple[list[int], int]]:
    valid_routes = []
    for route, minwidth in best_routes:
        print(f"Checking route: {route}")
        valid = True
        for i in range(len(route) - 1):
            if route[i + 1] not in node_list[route[i]].connection:
                valid = False
                print(f"Invalid segment: {route[i]} -> {route[i + 1]}")
                break
        if valid:
            valid_routes.append((route, minwidth))
            print(f"Valid route found: {route} with min width {minwidth}")
    if not valid_routes:
        print("No valid routes found")
    return valid_routes


def dynamic_topology_change(
    node_list, generation, total_generation, rand_log, best_routes
):
    print(f"Generation: {generation}")
    # ノードやエッジの追加
    if generation % 100 == 0:  # 100世代ごとに実施
        new_node_index = len(node_list)
        node_list.append(Node([], [], []))
        for _ in range(3):  # 新しいノードを3つのランダムな既存ノードに接続
            target_node = random.randint(0, new_node_index - 1)
            width = random.randint(1, 10) * 10
            connect_node_twoway(node_list, new_node_index, target_node, width, width)

        # print("After adding nodes:")
        # show_node_info(node_list)  # ノード情報を出力

        # ルートの存在確認
        valid_routes = validate_best_routes(node_list, best_routes)
        if valid_routes:
            best_routes[:] = valid_routes
        else:
            print(
                "No valid routes found. Updating best routes based on new exploration."
            )
            best_routes.clear()

    # ノードやエッジの削除
    if generation % 150 == 0:  # 150世代ごとに実施
        if len(node_list) > 3:
            remove_node_index = random.randint(0, len(node_list) - 1)
            # 削除対象のノードの接続を全て削除
            for node in node_list:
                if remove_node_index in node.connection:
                    index = node.connection.index(remove_node_index)
                    node.connection.pop(index)
                    node.pheromone.pop(index)
                    node.width.pop(index)

            # ノードリストからノードを削除
            node_list.pop(remove_node_index)

            # インデックスを調整
            for node in node_list:
                node.connection = [
                    i - 1 if i > remove_node_index else i for i in node.connection
                ]

            # フェロモンと最大フェロモンのリストを再調整
            for node in node_list:
                while len(node.pheromone) > len(node.connection):
                    node.pheromone.pop()
                while len(node.pheromone) < len(node.connection):
                    node.pheromone.append(MIN_F)

                while len(node.max_pheromone) > len(node.connection):
                    node.max_pheromone.pop()
                while len(node.max_pheromone) < len(node.connection):
                    node.max_pheromone.append(MAX_F)

            # print("After removing nodes:")
            # show_node_info(node_list)  # ノード情報を出力

            # ルートの存在確認
            valid_routes = validate_best_routes(node_list, best_routes)
            if valid_routes:
                best_routes[:] = valid_routes
            else:
                print(
                    "No valid routes found. Updating best routes based on new exploration."
                )
                best_routes.clear()


# /////////////////////////////////////////////////Main/////////////////////////////////////////////////
if __name__ == "__main__":
    # シミュレーション回数を指定
    for _ in range(100):

        # ! -------------------------------------------------初期化-------------------------------------------------
        node_list: list[Node] = []  # Nodeオブジェクト格納リスト
        ant_list: list[Ant] = []  # Antオブジェクト格納リスト
        interest_list: list[Interest] = []  # Interestオブジェクト格納リスト
        interest_log: list[int] = []  # interestのログ用リスト
        rand_list: list[Rand] = []  # Randオブジェクト格納リスト
        rand_log: list[int] = []  # Randのログ用リスト
        ant_log: list[int] = []  # Antのログ用リスト
        best_routes: list[tuple[list[int], int]] = []  # 最高経路を記録するリスト

        # ! -------------------------------------------------グラフ作成-------------------------------------------------
        print(
            "\n------------------------------------------Start of Graph Creation------------------------------------------\n"
        )
        node_list = ba_model(3, 100)

        START_NODE: int = random.randint(0, 99)

        while True:
            route = make_way(START_NODE, 6)
            if route != 0:
                break

        GOAL_NODE: int = route[-1]

        print(f"Route  ===  {route}")

        # 各ノードのフェロモン最小値を決定
        set_node_min_pheromone_by_degree(node_list)

        # 各ノードのフェロモン最大値を決定
        set_node_max_pheromone_by_width(node_list)

        # ! -------------------------------------------------探索-------------------------------------------------
        for gen in range(GENERATION):
            # ネットワーク構造を変更する。
            # TODO: ランダムの方が保持している最適値はルート的に存在しているかを確認する。
            dynamic_topology_change(node_list, gen, GENERATION, rand_log, best_routes)

            print(
                "\n-------------------------------------Start of Search Gen"
                + str(gen)
                + "-------------------------------------\n"
            )

            # Antによるフェロモン付加フェーズ
            # Antを配置
            ant_list = generate_ants(START_NODE, GOAL_NODE, gen, GENERATION)
            # Antの移動
            for _ in range(TTL):
                ant_next_node(ant_list, node_list, gen, ant_log)

            # 揮発フェーズ
            # option volatilize(node_list)
            volatilize(node_list)

            # Randによる評価
            # Randを配置
            rand_list = [
                Rand(START_NODE, GOAL_NODE, [START_NODE], W) for _ in range(ANT_NUM)
            ]
            # Randの移動
            for _ in range(TTL):
                rand_next_node(rand_list, node_list, rand_log, best_routes)

            # Interestによる評価
            print(
                "\n------------------------------------------Start of Evaluation------------------------------------------\n"
            )
            # Interestによる評価
            # Interestを配置
            interest_list.append(Interest(START_NODE, GOAL_NODE, [START_NODE], W))
            # Interestの移動
            for _ in range(TTL):
                interest_next_node(interest_list, node_list, interest_log)

        # ! -------------------------------------------------結果の処理-------------------------------------------------
        with open("./simulation_result/log_interest.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(interest_log)

        with open("./simulation_result/log_rand.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rand_log)

        with open("./simulation_result/log_ant.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        # visualize_graph(node_list)
