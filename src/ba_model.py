import pickle
import random

import networkx as nx

MIN_F = 100  # フェロモン最小値
MAX_F = 1000000000  # フェロモン最大値


class Node:
    def __init__(self, connection: list[int], pheromone: list[int], width: list[int]):
        self.connection = connection  # 接続先ノードの配列
        self.pheromone = pheromone  # 接続先ノードとのフェロモンの配列
        self.width = width  # 接続先ノードとの帯域の配列
        self.min_pheromone = MIN_F  # フェロモン最小値
        self.max_pheromone = [MAX_F for _ in pheromone]  # フェロモン最大値


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

    for _ in range(node_num - 3):
        target: list[int] = []
        node_num: list[int] = list(range(len(node_list)))
        weight: list[int] = node_degree.copy()

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

    return node_list


def save_network(node_list: list[Node], filename: str) -> None:
    with open(filename, "wb") as f:
        pickle.dump(node_list, f)


def load_network(filename: str) -> list[Node]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def find_max_bottleneck_path(node_list: list[Node]) -> tuple:
    max_bottleneck_bandwidth = 0
    best_path = None

    def dfs(node, destination, visited, path, min_bandwidth):
        nonlocal max_bottleneck_bandwidth, best_path
        if node == destination:
            if min_bandwidth > max_bottleneck_bandwidth:
                max_bottleneck_bandwidth = min_bandwidth
                best_path = path[:]
            return

        visited.add(node)
        for i, neighbor in enumerate(node_list[node].connection):
            if neighbor not in visited:
                path.append(neighbor)
                dfs(
                    neighbor,
                    destination,
                    visited,
                    path,
                    min(min_bandwidth, node_list[node].width[i]),
                )
                path.pop()
        visited.remove(node)

    for start in range(len(node_list)):
        for end in range(start + 1, len(node_list)):
            dfs(start, end, set(), [start], float("inf"))

    return best_path, max_bottleneck_bandwidth


def node2edge(node_list: list[Node]) -> list:
    edges = []
    for i in range(len(node_list)):
        line0 = node_list[i].connection
        line1 = node_list[i].pheromone
        line2 = node_list[i].width

        sum_line1 = sum(line1)

        for j in range(len(line0)):
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


def visualize_graph(node_list: list[Node]) -> None:
    nodes = [i for i in range(len(node_list))]
    edges = node2edge(node_list)

    g = nx.MultiDiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    agraph = nx.nx_agraph.to_agraph(g)
    agraph.node_attr["shape"] = "circle"
    agraph.draw("./simulation_result/ba_model_sample.pdf", prog="fdp", format="pdf")


if __name__ == "__main__":
    node_list = ba_model(3, 10)
    save_network(node_list, "ba_model_network.pkl")

    best_path, max_bottleneck_bandwidth = find_max_bottleneck_path(node_list)
    print(
        f"Best Path: {best_path}, Max Bottleneck Bandwidth: {max_bottleneck_bandwidth}"
    )

    visualize_graph(node_list)
