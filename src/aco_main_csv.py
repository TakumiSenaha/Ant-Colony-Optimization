# baモデル
import math
import random
import csv

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

V = 0.99  # フェロモン揮発量
MIN_F = 100  # フェロモン最小値
MAX_F = 1000000000  # フェロモン最大値
TTL = 100  # antのTime to Live
W = 1000  # 帯域幅初期値
BETA = 2  # 経路選択の際のヒューリスティック値に対する重み(累乗)

ANT_NUM = 1  # 一回で放つAntの数
GENERATION = 2000  # ant，interestを放つ回数(世代)

# /////////////////////////////////////////////////クラス定義/////////////////////////////////////////////////


class Node():
    def __init__(self, connection: list[int], pheromone: list[int], width: list[int]):
        self.connection = connection  # 接続先ノードの配列
        self.pheromone = pheromone  # 接続先ノードとのフェロモンの配列
        self.width = width  # 接続先ノードとの帯域の配列
        self.min_pheromone = MIN_F  # フェロモン最小値


class Ant():
    def __init__(self, current: int, destination: int, route: list[int], width: list[int]):
        self.current = current  # 現在のノード
        self.destination = destination  # コンテンツ保持ノード
        self.route = route  # 辿ってきた経路の配列
        self.width = width  # 辿ってきた経路の帯域の配列


class Interest():
    def __init__(self, current: int, destination: int, route: list[int], minwidth: int):
        self.current = current  # 現在のノード
        self.destination = destination  # コンテンツ保持ノード
        self.route = route  # 辿ってきた経路の配列
        self.minwidth = minwidth  # 辿ってきた経路の最小帯域


class Rand(Interest):
    def __init__(self, current: int, destination: int, route: list[int], minwidth: int):
        super().__init__(current, destination, route, minwidth)


# /////////////////////////////////////////////////関数定義/////////////////////////////////////////////////


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
            rate = (0.99 * (0.8**((100-node.width[i])/10)))
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
        before_node: Node = node_list[ant.route[i-1]]
        after_node: Node = node_list[ant.route[i]]
        # before_nodeからafter_nodeへのフェロモン値を変更
        index = before_node.connection.index(ant.route[i])
        # i-1番からi番ノードのフェロモン値を加算
        before_node.pheromone[index] += min(ant.width)
        # option before_node.pheromone[index] += int(( sum(ant.width) / len(ant.width) ))
        # option before_node.pheromone[index] += before_node.width[index] * int(( sum(ant.width) / len(ant.width) ))

        if before_node.pheromone[index] > MAX_F:
            before_node.pheromone[index] = MAX_F


def ant_next_node(ant_list: list[Ant], node_list: list[Node], current_generation: int) -> None:
    # antの次のノードを決定
    # 繰り返し中にリストから削除を行うためreversed
    for ant in reversed(ant_list):
        # antが今いるノードの接続ノード・フェロモン値・帯域幅を取得
        connection: list[int] = node_list[ant.current].connection
        pheromone: list[int] = node_list[ant.current].pheromone
        width:     list[int] = node_list[ant.current].width

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
            candidacy_width:      list[int] = []
            for i in diff_list:
                index = connection.index(i)
                candidacy_pheromones.append(pheromone[index])
                candidacy_width.append(width[index])

            # weight_width = [i ** BETA for i in candidacy_width]
            # weighting = [
            #     x*y for (x, y) in zip(candidacy_pheromones, weight_width)]
            
            # フェロモンの影響力を世代数に応じて調整
            pheromone_influence = (current_generation / GENERATION) * BETA
            weight_width = [i ** BETA for i in candidacy_width]
            weighting = [
                (pheromone ** pheromone_influence) * width_weight for pheromone, width_weight in zip(candidacy_pheromones, weight_width)
            ]

            next_node = random.choices(diff_list, k=1, weights=weighting)[0]

            # antの属性更新(現在地更新・ノード番号追加・帯域の配列に帯域を追加)
            ant.current = next_node
            ant.route.append(next_node)
            ant.width.append(width[connection.index(next_node)])

            # antが目的ノードならばノードにフェロモンの付加後ant_listから削除
            if ant.current == ant.destination:
                update_pheromone(ant, node_list)
                ant_list.remove(ant)
                print("Ant Goal! → " + str(ant.route) +
                      " : " + str(min(ant.width)))

            # antがTTLならばant_listから削除
            elif (len(ant.route) == TTL):
                ant_list.remove(ant)
                print("Ant TTL! → " + str(ant.route))


def interest_next_node(interest_list: list[Interest], node_list: list[Node], interest_log: list[int]) -> None:
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
            max_pheromone_index = [i for i, x in enumerate(
                candidacy_pheromones) if x == max(candidacy_pheromones)]
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
                print("Interest Goal! → " + str(interest.route) +
                      " : " + str(interest.minwidth))

        # interestがTTLならばinterest_listから削除
            elif (len(interest.route) == TTL):
                interest_list.remove(interest)
                interest_log.append(0)
                print("Interest TTL! →" + str(interest.route))


def rand_next_node(rand_list: list[Rand], node_list: list[Node], rand_log: list[int]) -> None:
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
                rand_log[-1] = max(rand_log)
            print("Rand Can't Find Route! → " + str(rand.route))

        # 候補先がある場合
        else:
            next_node = random.choice(diff_list)

            # randの属性更新
            # もし現在ノードから次ノードの帯域幅が今までの最小帯域より小さかったら更新
            rand.current = next_node
            rand.route.append(next_node)
            if width[connection.index(next_node)] < rand.minwidth:
                rand.minwidth = width[connection.index(next_node)]

        # randが目的ノードならばrand_listから削除
            if rand.current == rand.destination:
                rand_log.append(rand.minwidth)
                if max(rand_log) != rand.minwidth:
                    rand_log[-1] = max(rand_log)
                rand_list.remove(rand)
                print("Rand Goal! → " + str(rand.route) +
                      " : " + str(rand.minwidth))

        # randがTTLならばrand_listから削除
            elif (len(rand.route) == TTL):
                rand_list.remove(rand)
                rand_log.append(0)
                if max(rand_log) != 0:
                    rand_log[-1] = max(rand_log)
                print("Rand TTL! →" + str(rand.route))


def show_node_info(node_list: list[Node]) -> None:
    for i in range(len(node_list)):
        print("Node"+str(i))
        print(str(node_list[i].connection))
        print(str(node_list[i].pheromone))
        print(str(node_list[i].width))


def connect_node_oneway(node_list: list[Node], index_a: int, index_b: int, width: int) -> None:
    node_list[index_a].connection.append(index_b)
    node_list[index_a].pheromone.append(MIN_F)
    node_list[index_a].width.append(width)


def connect_node_twoway(node_list: list[Node], index_a: int, index_b: int, width_a2b: int, width_b2a: int) -> None:
    node_list[index_a].connection.append(index_b)
    node_list[index_a].pheromone.append(MIN_F)
    node_list[index_a].width.append(width_a2b)

    node_list[index_b].connection.append(index_a)
    node_list[index_b].pheromone.append(MIN_F)
    node_list[index_b].width.append(width_b2a)


def ba_model(edge_num: int, node_num: int) -> list[Node]:
    # 初期ノードを準備してそれぞれ連結させる
    node_list = [Node([], [], []) for _ in range(3)]
    connect_node_twoway(node_list, 0, 1, random.randint(
        1, 10) * 10, random.randint(1, 10) * 10)
    connect_node_twoway(node_list, 1, 2, random.randint(
        1, 10) * 10, random.randint(1, 10) * 10)
    connect_node_twoway(node_list, 2, 0, random.randint(
        1, 10) * 10, random.randint(1, 10) * 10)
    node_degree: list[int] = [2, 2, 2]

    # 所定の数になるまでノードを1つずつ追加
    for _ in range(node_num-3):
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
            connect_node_twoway(node_list, len(
                node_list)-1, i, random.randint(1, 10) * 10, random.randint(1, 10) * 10)
            node_degree[i] += 1

        node_degree.append(len(target))

    # print(node_degree) # debug
    return node_list


def make_way(start: int, hop: int):
    current_node: int = start
    route: list[int] = [current_node]

    for _ in range(hop):

        connection: list[int] = node_list[current_node].connection

        diff_list: list[int] = list(
            set(connection) ^ (set(route) & set(connection)))

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
            edge = (i, line0[j], {"minlen": "5.0", "label": str(line2[j])+":"+str(line1[j]), "color": "0.000 " + str(
                round(line1[j]/sum_line1, 2)) + " 1.000", "penwidth": str(line2[j]/20), "fontsize": "8"})
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


def set_node_min_pheromon_uniformly(node_list: list[Node]) -> None:
    for node in node_list:
        node.min_pheromone = MIN_F


# /////////////////////////////////////////////////Main/////////////////////////////////////////////////
if __name__ == "__main__":

    # シミュレーション回数を指定
    for _ in range(100):

        # ! -------------------------------------------------初期化-------------------------------------------------
        node_list:     list[Node] = []  # Nodeオブジェクト格納リスト
        ant_list:      list[Ant] = []  # Antオブジェクト格納リスト
        interest_list: list[Interest] = []  # Interestオブジェクト格納リスト
        interest_log:  list[int] = []  # interestのログ用リスト
        rand_list:     list[Rand] = []  # Randオブジェクト格納リスト
        rand_log:      list[int] = []  # Randのログ用リスト

        # ! -------------------------------------------------グラフ作成-------------------------------------------------
        print("\n------------------------------------------Start of Graph Creation------------------------------------------\n")
        node_list = ba_model(3, 100)

        START_NODE: int = random.randint(0, 99)

        while (True):
            route = make_way(START_NODE, 6)
            if route != 0:
                break

        GOAL_NODE: int = route[-1]

        print(f"Route  ===  {route}")

        # routeの帯域を100に書き換え
        for i in range(1, len(route)):
            before_node: Node = node_list[route[i-1]]

            index = before_node.connection.index(route[i])
            before_node.width[index] = 100

        # show_node_info(node_list) # debug

        # 各ノードのフェロモン最小値を決定
        # option set_node_min_pheromon_uniformly(node_list)
        set_node_min_pheromone_by_degree(node_list)

        # show_node_info(node_list) # debug

        # ! -------------------------------------------------探索-------------------------------------------------
        for gen in range(GENERATION):
            print("\n-------------------------------------Start of Search Gen" +
                  str(gen) + "-------------------------------------\n")

            # Antによるフェロモン付加フェーズ
            # Antを配置
            ant_list.extend([Ant(START_NODE, GOAL_NODE, [START_NODE], [])
                            for _ in range(ANT_NUM)])
            # Antの移動
            for _ in range(TTL):
                ant_next_node(ant_list, node_list, gen)

            # 揮発フェーズ
            # option volatilize(node_list)
            volatilize_by_width(node_list)

            # ! -------------------------------------------------評価-------------------------------------------------
            print("\n------------------------------------------Start of Evaluation------------------------------------------\n")
            # Interestによる評価
            # Interestを配置
            interest_list.append(
                Interest(START_NODE, GOAL_NODE, [START_NODE], W))
            # Interestの移動
            for _ in range(TTL):
                interest_next_node(interest_list, node_list, interest_log)

            # Randによる評価
            # Randを配置
            rand_list.extend(
                [Rand(START_NODE, GOAL_NODE, [START_NODE], W) for _ in range(ANT_NUM)])
            # Randの移動
            for _ in range(TTL):
                rand_next_node(rand_list, node_list, rand_log)

        # ! -------------------------------------------------結果の処理-------------------------------------------------

        # show_node_info(node_list)

        # print(interest_log)
        f = open("./simulation_result/log_interest.csv", "a", newline="")
        writer = csv.writer(f)
        writer.writerow(interest_log)
        f.close()

        # print(rand_log)
        f = open("./simulation_result/log_rand.csv", "a", newline="")
        writer = csv.writer(f)
        writer.writerow(rand_log)
        f.close()

        # visualize_graph(node_list)
