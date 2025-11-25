"""
グラフモジュール

NetworkXをベースとしたグラフの生成と管理を行います。
各ノードにNodeLearning機能を統合します。
"""

import random
from typing import Dict, List

import networkx as nx

from .node import NodeLearning


class RoutingGraph:
    """ACOルーティング用のグラフクラス（NetworkXラッパー）"""

    def __init__(self, num_nodes: int, config: Dict):
        """
        Args:
            num_nodes: ノード数
            config: 設定辞書（config.yamlから読み込んだもの）
        """
        self.num_nodes = num_nodes
        self.config = config
        self.graph = self._generate_graph()

        # 各ノードに学習機能を追加
        window_size = config["aco"]["learning"]["bkb_window_size"]
        self.node_learning: Dict[int, NodeLearning] = {
            node: NodeLearning(node, window_size) for node in self.graph.nodes()
        }

    def _generate_graph(self) -> nx.Graph:
        """
        グラフを生成し、エッジ属性（帯域、遅延、フェロモン）を初期化

        Returns:
            生成されたグラフ
        """
        graph_type = self.config["graph"]["graph_type"]
        num_nodes = self.config["graph"]["num_nodes"]

        if graph_type == "barabasi_albert":
            num_edges = self.config["graph"]["num_edges"]
            graph = nx.barabasi_albert_graph(num_nodes, num_edges)
        elif graph_type == "erdos_renyi":
            edge_prob = self.config["graph"].get("edge_prob", 0.1)
            graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
        elif graph_type == "grid":
            import math

            side = int(math.sqrt(num_nodes))
            graph = nx.grid_2d_graph(side, side)
            # ノードをint型に変換
            mapping = {(i, j): i * side + j for i in range(side) for j in range(side)}
            graph = nx.relabel_nodes(graph, mapping)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        # エッジ属性を初期化
        self._initialize_edge_attributes(graph)

        return graph

    def _initialize_edge_attributes(self, graph: nx.Graph) -> None:
        """
        エッジ属性（帯域、遅延、フェロモン）を初期化

        Args:
            graph: 初期化するグラフ
        """
        bw_range = self.config["graph"]["bandwidth_range"]
        delay_range = self.config["graph"]["delay_range"]
        min_pheromone = self.config["aco"]["min_pheromone"]
        max_pheromone = self.config["aco"]["max_pheromone"]

        for u, v in graph.edges():
            # 帯域幅（Mbps）: 10刻みで生成
            min_val = ((bw_range[0] + 9) // 10) * 10  # 10の倍数に切り上げ
            max_val = (bw_range[1] // 10) * 10  # 10の倍数に切り下げ
            random_value = random.randint(min_val // 10, max_val // 10)
            bandwidth = random_value * 10

            graph.edges[u, v]["bandwidth"] = float(bandwidth)
            graph.edges[u, v]["original_bandwidth"] = float(bandwidth)  # 変動の基準値

            # 遅延（ms）: 10刻みで生成（10, 20, 30, ..., 100）
            min_delay = delay_range[0] // 10  # 10
            max_delay = delay_range[1] // 10  # 10
            base_delay = random.randint(min_delay, max_delay) * 10
            graph.edges[u, v]["delay"] = float(base_delay)
            graph.edges[u, v]["original_delay"] = float(base_delay)

            # フェロモン（初期値のみ設定、min/maxは後で設定）
            graph.edges[u, v]["pheromone"] = min_pheromone
            graph.edges[u, v]["min_pheromone"] = min_pheromone  # 一時的な値
            graph.edges[u, v]["max_pheromone"] = max_pheromone  # 一時的な値

        # 既存実装と同じ方法でフェロモン最小値・最大値を設定
        self._set_pheromone_min_max_by_degree_and_width(graph, min_pheromone)

    def _set_pheromone_min_max_by_degree_and_width(
        self, graph: nx.Graph, base_min_pheromone: int
    ) -> None:
        """
        ノードの隣接数と帯域幅に基づいてフェロモンの最小値と最大値を双方向に設定
        （既存実装との互換性のため）

        Args:
            graph: ネットワークグラフ
            base_min_pheromone: ベースとなるフェロモン最小値
        """
        for u, v in graph.edges():
            # ノードuとvの隣接ノード数を取得
            degree_u = len(list(graph.neighbors(u)))
            degree_v = len(list(graph.neighbors(v)))

            # フェロモン最小値を隣接ノード数に基づいて設定（既存実装と同じ）
            graph.edges[u, v]["min_pheromone"] = base_min_pheromone * 3 // degree_u
            graph.edges[v, u]["min_pheromone"] = base_min_pheromone * 3 // degree_v

            # 帯域幅に基づいてフェロモン最大値を設定（既存実装と同じ）
            width_u_to_v = graph.edges[u, v]["bandwidth"]
            width_v_to_u = graph.edges[v, u]["bandwidth"]

            graph.edges[u, v]["max_pheromone"] = int(width_u_to_v**5)
            graph.edges[v, u]["max_pheromone"] = int(width_v_to_u**5)

    def get_neighbors(self, node: int) -> List[int]:
        """
        指定されたノードの隣接ノードを取得

        Args:
            node: ノードID

        Returns:
            隣接ノードのリスト
        """
        return list(self.graph.neighbors(node))

    def get_edge_attributes(self, u: int, v: int) -> Dict[str, float]:
        """
        エッジの属性（帯域、遅延、フェロモン）を取得

        Args:
            u: ノードu
            v: ノードv

        Returns:
            エッジ属性の辞書
        """
        return {
            "bandwidth": self.graph.edges[u, v]["bandwidth"],
            "delay": self.graph.edges[u, v]["delay"],
            "pheromone": self.graph.edges[u, v]["pheromone"],
        }

    def update_pheromone(
        self, u: int, v: int, delta_pheromone: float, bidirectional: bool = True
    ) -> None:
        """
        フェロモンを更新

        Args:
            u: ノードu
            v: ノードv
            delta_pheromone: 付加するフェロモン量
            bidirectional: 双方向に更新するか
        """
        # u -> v
        new_pheromone = self.graph.edges[u, v]["pheromone"] + delta_pheromone
        new_pheromone = max(
            self.graph.edges[u, v]["min_pheromone"],
            min(new_pheromone, self.graph.edges[u, v]["max_pheromone"]),
        )
        self.graph.edges[u, v]["pheromone"] = new_pheromone

        # v -> u (双方向)
        if bidirectional:
            new_pheromone_vu = self.graph.edges[v, u]["pheromone"] + delta_pheromone
            new_pheromone_vu = max(
                self.graph.edges[v, u]["min_pheromone"],
                min(new_pheromone_vu, self.graph.edges[v, u]["max_pheromone"]),
            )
            self.graph.edges[v, u]["pheromone"] = new_pheromone_vu

    def evaporate_pheromone(self, evaporation_rate: float) -> None:
        """
        全エッジのフェロモンを揮発

        Args:
            evaporation_rate: 揮発率（0.0 ~ 1.0）
        """
        for u, v in self.graph.edges():
            current = self.graph.edges[u, v]["pheromone"]
            new_pheromone = current * (1.0 - evaporation_rate)
            new_pheromone = max(new_pheromone, self.graph.edges[u, v]["min_pheromone"])
            self.graph.edges[u, v]["pheromone"] = new_pheromone

    def evaporate_node_learning(self, evaporation_rate: float) -> None:
        """
        全ノードの学習値（BKB/BLD/BKH）を揮発

        Args:
            evaporation_rate: 揮発率
        """
        for node_learning in self.node_learning.values():
            node_learning.evaporate(evaporation_rate)

    def __getitem__(self, node: int) -> NodeLearning:
        """
        ノードの学習機能に直接アクセス

        Args:
            node: ノードID

        Returns:
            NodeLearningオブジェクト
        """
        return self.node_learning[node]

    def __repr__(self) -> str:
        return (
            f"RoutingGraph(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )
