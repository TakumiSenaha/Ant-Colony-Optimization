"""
単一目的最適化ソルバー

帯域のみを最適化する場合の厳密解を計算します。
Modified Dijkstra法を拡張したアルゴリズムを使用します。

遅延制約付きの最適化もサポートします。
"""

from heapq import heappop, heappush
from typing import List, Optional, Tuple

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
    current: Optional[int] = target
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


def max_load_path_with_delay_constraint(
    graph: nx.Graph,
    source: int,
    target: int,
    max_delay: float,
    bandwidth_weight: str = "bandwidth",
    delay_weight: str = "delay",
) -> List[int]:
    """
    遅延制約付きで最大ボトルネック帯域の経路を探索

    Modified Dijkstra法を拡張し、遅延制約を満たす経路の中で
    最大ボトルネック帯域を持つ経路を見つけます。

    【アルゴリズム】
    1. 各ノードに対して、到達可能な最大ボトルネック帯域と最小遅延を記録
    2. 遅延制約を満たす経路のみを探索対象とする
    3. ボトルネック帯域が最大の経路を選択

    Args:
        graph: ネットワークグラフ
        source: 開始ノード
        target: 目的地ノード
        max_delay: 最大遅延制約（ms）
        bandwidth_weight: 帯域のエッジ属性名（デフォルト: "bandwidth"）
        delay_weight: 遅延のエッジ属性名（デフォルト: "delay"）

    Returns:
        遅延制約を満たす最大ボトルネック帯域の経路（ノードのリスト）

    Raises:
        nx.NodeNotFound: 開始ノードまたは目的地ノードがグラフに存在しない場合
        nx.NetworkXNoPath: 制約を満たす経路が存在しない場合
    """
    # ノードの存在確認
    if source not in graph:
        raise nx.NodeNotFound(f"Source {source} not in graph")
    if target not in graph:
        raise nx.NodeNotFound(f"Target {target} not in graph")

    # 優先度キュー（最大ヒープをシミュレートするため負の値を使用）
    # 要素: (-bottleneck, delay, node)
    pq = []

    # 各ノードの最良ボトルネック値と最小遅延を記録
    bottleneck = {node: float("-inf") for node in graph}
    min_delay = {node: float("inf") for node in graph}
    bottleneck[source] = float("inf")
    min_delay[source] = 0.0

    # 各ノードの前駆ノードを記録（経路再構築用）
    pred = {node: None for node in graph}

    # 訪問済みノードを記録
    visited = set()

    # 開始ノードを優先度キューに追加
    heappush(pq, (-bottleneck[source], min_delay[source], source))

    # メインループ：ボトルネック値が大きい順にノードを処理
    while pq:
        curr_bottle_neg, curr_delay, u = heappop(pq)
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
            # エッジの属性を取得
            edge_data = graph[u][v]
            if graph.is_multigraph():
                # マルチグラフの場合、並列エッジの最小重みを使用
                bw = min(data.get(bandwidth_weight, 1) for data in edge_data.values())
                delay = min(data.get(delay_weight, 0) for data in edge_data.values())
            else:
                bw = edge_data.get(bandwidth_weight, 1)
                delay = edge_data.get(delay_weight, 0)

            # 新しいボトルネック値 = min(現在のボトルネック, エッジの帯域)
            new_bottle = min(curr_bottle, bw)
            # 新しい遅延 = 現在の遅延 + エッジの遅延
            new_delay = curr_delay + delay

            # 遅延制約を満たす場合のみ処理
            if new_delay > max_delay:
                continue

            # より良いボトルネック値が見つかったら更新
            # または、同じボトルネック値でより小さい遅延が見つかったら更新
            if new_bottle > bottleneck[v] or (
                new_bottle == bottleneck[v] and new_delay < min_delay[v]
            ):
                bottleneck[v] = new_bottle
                min_delay[v] = new_delay
                pred[v] = u
                heappush(pq, (-new_bottle, new_delay, v))

    # 経路が存在しない場合、または制約を満たす経路が存在しない場合
    if bottleneck[target] == float("-inf") or min_delay[target] > max_delay:
        raise nx.NetworkXNoPath(
            f"No path found from {source} to {target} "
            f"satisfying delay constraint (max_delay={max_delay}ms)"
        )

    # 経路を再構築
    path = []
    current: Optional[int] = target
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()

    return path


def calculate_path_delay(
    graph: nx.Graph, path: List[int], delay_weight: str = "delay"
) -> float:
    """
    経路上の累積遅延を計算

    Args:
        graph: ネットワークグラフ
        path: 始点から終点までの経路（ノードのリスト）
        delay_weight: 遅延のエッジ属性名（デフォルト: "delay"）

    Returns:
        経路の累積遅延（ms）
    """
    if len(path) < 2:
        return 0.0

    return sum(
        graph.edges[path[i], path[i + 1]].get(delay_weight, 0)
        for i in range(len(path) - 1)
    )


def find_all_max_bottleneck_paths_with_delay_constraint(
    graph: nx.Graph,
    source: int,
    target: int,
    max_delay: float,
    bandwidth_weight: str = "bandwidth",
    delay_weight: str = "delay",
) -> List[List[int]]:
    """
    遅延制約付きで最大ボトルネック帯域を持つ全ての経路を探索

    まず最大ボトルネック帯域を求め、その帯域を持つ全ての経路を列挙します。

    Args:
        graph: ネットワークグラフ
        source: 開始ノード
        target: 目的地ノード
        max_delay: 最大遅延制約（ms）
        bandwidth_weight: 帯域のエッジ属性名（デフォルト: "bandwidth"）
        delay_weight: 遅延のエッジ属性名（デフォルト: "delay"）

    Returns:
        遅延制約を満たす最大ボトルネック帯域を持つ全ての経路（経路のリスト）
    """
    # まず最大ボトルネック帯域を求める
    try:
        optimal_path = max_load_path_with_delay_constraint(
            graph, source, target, max_delay, bandwidth_weight, delay_weight
        )
        optimal_bottleneck = bottleneck_capacity(graph, optimal_path, bandwidth_weight)
    except nx.NetworkXNoPath:
        return []

    # 最大ボトルネック帯域を持つ全ての経路を列挙（DFS）
    all_optimal_paths: List[List[int]] = []

    def dfs(
        current: int,
        path: List[int],
        current_bottleneck: float,
        current_delay: float,
        visited: set,
    ) -> None:
        """DFSで最大ボトルネック帯域を持つ経路を探索"""
        if current == target:
            # 目的地に到達し、最大ボトルネック帯域を持ち、遅延制約を満たす場合
            if (
                abs(current_bottleneck - optimal_bottleneck) < 1e-6
                and current_delay <= max_delay
            ):
                all_optimal_paths.append(path[:])
            return

        visited.add(current)

        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue

            # エッジの属性を取得
            edge_data = graph[current][neighbor]
            if graph.is_multigraph():
                bw = min(data.get(bandwidth_weight, 1) for data in edge_data.values())
                delay = min(data.get(delay_weight, 0) for data in edge_data.values())
            else:
                bw = edge_data.get(bandwidth_weight, 1)
                delay = edge_data.get(delay_weight, 0)

            # 新しいボトルネック値と遅延を計算
            new_bottleneck = min(current_bottleneck, bw)
            new_delay = current_delay + delay

            # 最大ボトルネック帯域より小さい、または遅延制約を超える場合はスキップ
            if new_bottleneck < optimal_bottleneck - 1e-6 or new_delay > max_delay:
                continue

            # 再帰的に探索
            path.append(neighbor)
            dfs(neighbor, path, new_bottleneck, new_delay, visited)
            path.pop()

        visited.remove(current)

    # DFSを開始
    dfs(source, [source], float("inf"), 0.0, set())

    return all_optimal_paths
