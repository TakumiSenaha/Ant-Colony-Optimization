"""
Max Load Path Algorithms for Weighted Graphs
=============================================

This module provides an algorithm to compute a path from a source to a target that
maximizes the minimum edge weight along the path (i.e. the bottleneck capacity). In
contrast to standard shortest-path algorithms that sum edge weights, this algorithm
propagates the minimum edge weight encountered along a given path and uses a modified
Dijkstra-like approach to maximize that value.
"""

from heapq import heappop, heappush

import networkx as nx

__all__ = ["max_load_path"]


@nx._dispatchable(edge_attrs="weight")
def max_load_path(G, source, target, weight="weight"):
    """
    Returns a path from source to target that maximizes the minimum edge weight along the path,
    i.e. the bottleneck capacity.

    This algorithm is a modification of Dijkstra’s algorithm. Instead of summing the edge
    weights to determine the cost of a path, the cost here is defined as the minimum edge
    weight (bottleneck) encountered along the path. At every step, the algorithm selects the node
    whose current bottleneck value is the highest, and then it updates its neighbors with the
    minimum of the current bottleneck and the connecting edge’s weight.

    Parameters
    ----------
    G : NetworkX graph
        The input graph.
    source : node
        The starting node.
    target : node
        The destination node.
    weight : string or function, optional (default="weight")
        If a string, then the edge weight is obtained from the edge attribute with that key.
        If a function, it should accept exactly three arguments: (u, v, edge_data) and return a numeric weight.
        For multigraphs, the minimum weight among parallel edges is used.

    Returns
    -------
    path : list
        A list of nodes representing the path from source to target that maximizes the bottleneck capacity.

    Raises
    ------
    NodeNotFound
        If either the source or target is not in G.
    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge('a', 'b', weight=5)
    >>> G.add_edge('b', 'c', weight=3)
    >>> G.add_edge('a', 'c', weight=2)
    >>> # The bottleneck capacity of the path ['a', 'b', 'c'] is min(5, 3) = 3,
    >>> # which is higher than the direct edge from 'a' to 'c' (capacity 2).
    >>> max_load_path(G, 'a', 'c')
    ['a', 'b', 'c']
    """
    # Check that the source and target are in the graph.
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in graph")
    if target not in G:
        raise nx.NodeNotFound(f"Target {target} not in graph")

    # Priority queue: we use a min-heap with negative values to simulate a max-heap.
    pq = []

    # bottleneck: record the best (i.e. maximum) bottleneck value found so far for each node.
    # Initialize all nodes with negative infinity; the source is set to positive infinity.
    bottleneck = {node: float("-inf") for node in G}
    bottleneck[source] = float("inf")

    # pred: record the predecessor of each node for path reconstruction.
    pred = {node: None for node in G}

    # visited: to keep track of nodes whose best bottleneck value is finalized.
    visited = set()

    # Push the source node into the priority queue. We push (-bottleneck, node)
    # so that the highest bottleneck (largest positive value) comes out first.
    heappush(pq, (-bottleneck[source], source))

    # Main loop: process nodes in order of decreasing bottleneck value.
    while pq:
        curr_bottle_neg, u = heappop(pq)
        curr_bottle = -curr_bottle_neg  # convert back to positive

        # Skip if the node has already been finalized.
        if u in visited:
            continue
        visited.add(u)

        # If the target is reached, exit early.
        if u == target:
            break

        # Process each neighbor v of u.
        for v in G.neighbors(u):
            # Retrieve the edge weight.
            edge_data = G[u][v]
            if G.is_multigraph():
                # For multigraphs, use the minimum edge weight among all parallel edges.
                w = min(data.get(weight, 1) for data in edge_data.values())
            else:
                w = edge_data.get(weight, 1)

            # The new bottleneck for v is the minimum of the current bottleneck for u and the weight of edge (u,v).
            new_bottle = min(curr_bottle, w)
            # If the new bottleneck is greater than the best known for v, update it.
            if new_bottle > bottleneck[v]:
                bottleneck[v] = new_bottle
                pred[v] = u
                heappush(pq, (-new_bottle, v))

    # If the target's bottleneck value was never updated, no path exists.
    if bottleneck[target] == float("-inf"):
        raise nx.NetworkXNoPath(f"No path found from {source} to {target}")

    # Reconstruct the path from target to source using the predecessor dictionary.
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()

    return path


def bottleneck_capacity(graph: nx.Graph, path: list) -> float:
    """
    経路上のボトルネック（最小のリンク帯域）を計算する。

    Parameters
    ----------
    graph : nx.Graph
        入力グラフ
    path : list
        始点から終点までの経路（ノードのリスト）

    Returns
    -------
    float
        経路のボトルネック帯域（最小エッジの重み）
    """
    return min(graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


if __name__ == "__main__":
    """
    実行例：
    ```bash
    python src/modified_dijkstra.py networks/ba_model_graph 30 32
    ```
    """
    import argparse

    from utils.graph_utils import load_graph

    parser = argparse.ArgumentParser(description="最大ボトルネック帯域値の経路を探索")
    parser.add_argument(
        "graph_file", type=str, help="エッジリスト形式のネットワークファイル"
    )
    parser.add_argument("start_node", type=int, help="開始ノード")
    parser.add_argument("end_node", type=int, help="終了ノード")
    args = parser.parse_args()

    print(f"📂 ファイル '{args.graph_file}' からグラフをロード中...")
    G = load_graph(args.graph_file)

    try:
        path = max_load_path(G, args.start_node, args.end_node)
        bottleneck_value = bottleneck_capacity(G, path)

        print("\n🚀 最大ボトルネック帯域値の経路探索結果 🚀")
        print(f"🔵 開始ノード: {args.start_node}")
        print(f"🔴 終了ノード: {args.end_node}")
        print(f"🔗 経路: {path}")
        print(f"⚡ ボトルネック帯域値: {bottleneck_value} Mbps")

    except nx.NetworkXNoPath:
        print(
            f"❌ エラー: {args.start_node} から {args.end_node} への経路が存在しません。"
        )
