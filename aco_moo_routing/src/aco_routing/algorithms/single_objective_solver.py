"""
単一目的最適化ソルバー

帯域のみを最適化する場合の厳密解を計算します。
Modified Dijkstra法を拡張したアルゴリズムを使用します。
"""

from heapq import heappop, heappush
from typing import List, Tuple

import networkx as nx


def max_load_path(
    graph: nx.Graph, source: int, target: int, weight: str = "bandwidth"
) -> List[int]:
    """
    最大ボトルネック帯域の経路を探索

    Modified Dijkstra法を使用して、経路上の最小エッジ重み（ボトルネック）を
    最大化する経路を見つけます。

    Args:
        graph: ネットワークグラフ
        source: 開始ノード
        target: 目的地ノード
        weight: エッジの重み属性名（デフォルト: "bandwidth"）

    Returns:
        最大ボトルネック帯域の経路（ノードのリスト）

    Raises:
        nx.NodeNotFound: 開始ノードまたは目的地ノードがグラフに存在しない場合
        nx.NetworkXNoPath: 開始ノードから目的地ノードへの経路が存在しない場合
    """
    # ノードの存在確認
    if source not in graph:
        raise nx.NodeNotFound(f"Source {source} not in graph")
    if target not in graph:
        raise nx.NodeNotFound(f"Target {target} not in graph")

    # 優先度キュー（最大ヒープをシミュレートするため負の値を使用）
    pq = []

    # 各ノードの最良ボトルネック値を記録
    bottleneck = {node: float("-inf") for node in graph}
    bottleneck[source] = float("inf")

    # 各ノードの前駆ノードを記録（経路再構築用）
    pred = {node: None for node in graph}

    # 訪問済みノードを記録
    visited = set()

    # 開始ノードを優先度キューに追加
    heappush(pq, (-bottleneck[source], source))

    # メインループ：ボトルネック値が大きい順にノードを処理
    while pq:
        curr_bottle_neg, u = heappop(pq)
        curr_bottle = -curr_bottle_neg  # 負の値を正の値に変換

        # 既に確定済みのノードはスキップ
        if u in visited:
            continue
        visited.add(u)

        # 目的地に到達したら終了
        if u == target:
            break

        # 隣接ノードを処理
        for v in graph.neighbors(u):
            # エッジの重みを取得
            edge_data = graph[u][v]
            if graph.is_multigraph():
                # マルチグラフの場合、並列エッジの最小重みを使用
                w = min(data.get(weight, 1) for data in edge_data.values())
            else:
                w = edge_data.get(weight, 1)

            # 新しいボトルネック値 = min(現在のボトルネック, エッジの重み)
            new_bottle = min(curr_bottle, w)

            # より良いボトルネック値が見つかったら更新
            if new_bottle > bottleneck[v]:
                bottleneck[v] = new_bottle
                pred[v] = u
                heappush(pq, (-new_bottle, v))

    # 経路が存在しない場合
    if bottleneck[target] == float("-inf"):
        raise nx.NetworkXNoPath(f"No path found from {source} to {target}")

    # 経路を再構築
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()

    return path


def bottleneck_capacity(
    graph: nx.Graph, path: List[int], weight: str = "bandwidth"
) -> float:
    """
    経路上のボトルネック（最小のリンク帯域）を計算

    Args:
        graph: ネットワークグラフ
        path: 始点から終点までの経路（ノードのリスト）
        weight: エッジの重み属性名（デフォルト: "bandwidth"）

    Returns:
        経路のボトルネック帯域（最小エッジの重み）
    """
    if len(path) < 2:
        return 0.0

    return min(
        graph.edges[path[i], path[i + 1]].get(weight, 0) for i in range(len(path) - 1)
    )
