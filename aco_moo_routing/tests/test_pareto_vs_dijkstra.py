"""
パレートソルバーとModified Dijkstraの比較テスト

帯域のみを最適化する場合、同じ結果になるかを確認します。
"""

import sys
from pathlib import Path

import networkx as nx

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from aco_routing.algorithms.pareto_solver import ParetoSolver


def max_load_path_simple(graph: nx.Graph, source: int, target: int) -> tuple:
    """
    Modified Dijkstra法の簡易実装（比較用）

    Returns:
        (path, bottleneck_value)
    """
    from heapq import heappop, heappush

    pq = []
    bottleneck = {node: float("-inf") for node in graph}
    bottleneck[source] = float("inf")
    pred = {node: None for node in graph}
    visited = set()

    heappush(pq, (-bottleneck[source], source))

    while pq:
        curr_bottle_neg, u = heappop(pq)
        curr_bottle = -curr_bottle_neg

        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v in graph.neighbors(u):
            w = graph.edges[u, v]["bandwidth"]
            new_bottle = min(curr_bottle, w)
            if new_bottle > bottleneck[v]:
                bottleneck[v] = new_bottle
                pred[v] = u
                heappush(pq, (-new_bottle, v))

    if bottleneck[target] == float("-inf"):
        raise nx.NetworkXNoPath(f"No path found from {source} to {target}")

    path = []
    current = target
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()

    return path, bottleneck[target]


def test_bandwidth_only_optimization():
    """
    帯域のみを最適化する場合、パレートソルバーとModified Dijkstraが同じ結果を返すかテスト

    注意: パレートソルバーは遅延とホップ数も記録するため、
    同じ帯域でも異なる遅延/ホップ数の経路が複数返される可能性がある。
    そのため、最大帯域の経路が同じかを確認する。
    """
    # テスト用グラフを作成
    graph = nx.Graph()
    graph.add_edge(0, 1, bandwidth=100, delay=5)
    graph.add_edge(1, 2, bandwidth=80, delay=3)
    graph.add_edge(0, 2, bandwidth=60, delay=2)
    graph.add_edge(1, 3, bandwidth=90, delay=4)
    graph.add_edge(3, 2, bandwidth=70, delay=3)

    source = 0
    target = 2

    # Modified Dijkstra法で計算
    dijkstra_path, dijkstra_bottleneck = max_load_path_simple(graph, source, target)

    # パレートソルバーで計算
    pareto_solver = ParetoSolver(graph)
    pareto_frontier = pareto_solver.find_pareto_frontier(source, target)

    # パレートフロンティアから最大帯域の解を取得
    if pareto_frontier:
        max_bandwidth_solution = max(pareto_frontier, key=lambda x: x[0])
        pareto_bottleneck = max_bandwidth_solution[0]
        pareto_path = max_bandwidth_solution[3]

        # 最大帯域が同じか確認
        assert abs(dijkstra_bottleneck - pareto_bottleneck) < 0.01, (
            f"Dijkstra bottleneck: {dijkstra_bottleneck}, "
            f"Pareto bottleneck: {pareto_bottleneck}"
        )

        # 経路が同じか確認（順序は異なる可能性があるので、セットで比較）
        assert set(dijkstra_path) == set(
            pareto_path
        ), f"Dijkstra path: {dijkstra_path}, Pareto path: {pareto_path}"

        print("✅ 帯域のみ最適化の場合、同じ結果を返すことを確認")
    else:
        raise AssertionError("Pareto frontier is empty")


if __name__ == "__main__":
    test_bandwidth_only_optimization()
    print("✅ All tests passed!")
