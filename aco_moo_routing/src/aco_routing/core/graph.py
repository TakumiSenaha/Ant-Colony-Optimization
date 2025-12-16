"""
グラフモジュール

NetworkXをベースとしたグラフの生成と管理を行います。
各ノードにNodeLearning機能を統合し、ACOルーティングに必要な機能を提供します。

【主要機能】
1. グラフ生成：Barabási-Albert、Erdős-Rényi、Grid、Manualトポロジをサポート
2. エッジ属性管理：帯域、遅延、フェロモンの初期化と更新
3. ノード学習機能：各ノードにNodeLearningインスタンスを付与
4. フェロモン管理：フェロモンの更新、揮発、取得
"""

import random
from typing import Dict, List

import networkx as nx

from .node import NodeLearning


class RoutingGraph:
    """
    ACOルーティング用のグラフクラス（NetworkXラッパー）

    NetworkXのGraphオブジェクトをラップし、ACOルーティングに必要な機能を提供します。
    各ノードにはNodeLearningインスタンスが付与され、分散学習が可能です。

    Attributes:
        num_nodes (int): ノード数
        config (Dict): 設定辞書（config.yamlから読み込んだもの）
        graph (nx.Graph): NetworkXのGraphオブジェクト
        node_learning (Dict[int, NodeLearning]): ノードIDをキーとするNodeLearningインスタンスの辞書

    Example:
        >>> config = {"graph": {"num_nodes": 100, "num_edges": 6, ...}, ...}
        >>> graph = RoutingGraph(num_nodes=100, config=config)
        >>> graph.get_edge_attributes(0, 1)
        {"bandwidth": 50.0, "delay": 5.0, "pheromone": 100}
    """

    def __init__(self, num_nodes: int, config: Dict):
        """
        グラフを初期化します。

        Args:
            num_nodes: ノード数（設定ファイルの値と一致させる必要があります）
            config: 設定辞書（config.yamlから読み込んだもの）

        Note:
            - グラフ生成後、各ノードにNodeLearningインスタンスが自動的に付与されます
            - エッジ属性（帯域、遅延、フェロモン）は自動的に初期化されます
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
        グラフを生成し、エッジ属性（帯域、遅延、フェロモン）を初期化します。

        Returns:
            生成されたNetworkXのGraphオブジェクト

        Note:
            サポートされているグラフタイプ:
            - "barabasi_albert": Barabási-Albertモデル（スケールフリーネットワーク）
            - "erdos_renyi": Erdős-Rényiモデル（ランダムグラフ）
            - "grid": 2次元グリッドグラフ
            - "manual": 手動設定トポロジ（BAモデルで生成後、最適経路を100Mbpsに設定）

        Raises:
            ValueError: 未知のグラフタイプが指定された場合
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
        elif graph_type == "manual":
            # 手動設定トポロジ：BAモデルで生成し、後で最適経路を100Mbpsに設定
            # 最適経路の設定はrun_experiment.pyで行う
            num_edges = self.config["graph"]["num_edges"]
            graph = nx.barabasi_albert_graph(num_nodes, num_edges)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        # エッジ属性を初期化
        self._initialize_edge_attributes(graph)

        return graph

    def _initialize_edge_attributes(self, graph: nx.Graph) -> None:
        """
        エッジ属性（帯域、遅延、フェロモン）を初期化します。

        Args:
            graph: 初期化するNetworkXのGraphオブジェクト

        Note:
            - 帯域幅: 10Mbps刻みでランダムに生成（設定ファイルのbandwidth_rangeに基づく）
            - 遅延: 1ms刻みでランダムに生成（設定ファイルのdelay_rangeに基づく）
            - フェロモン: 最小値で初期化（後でノードの次数と帯域に基づいて最大値が設定される）
            - 先行研究用: local_min_bandwidthとlocal_max_bandwidthも初期化
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

            # 遅延（ms）: 1刻みで生成（1, 2, 3, ..., 10）
            base_delay = random.randint(int(delay_range[0]), int(delay_range[1]))
            graph.edges[u, v]["delay"] = float(base_delay)
            graph.edges[u, v]["original_delay"] = float(base_delay)

            # フェロモン（初期値のみ設定、min/maxは後で設定）
            graph.edges[u, v]["pheromone"] = min_pheromone
            graph.edges[u, v]["min_pheromone"] = min_pheromone  # 一時的な値
            graph.edges[u, v]["max_pheromone"] = max_pheromone  # 一時的な値

            # 先行研究（Previous Method）用：エッジベースの学習値
            # 初期値はエッジの帯域幅に設定
            graph.edges[u, v]["local_min_bandwidth"] = float(bandwidth)
            graph.edges[u, v]["local_max_bandwidth"] = float(bandwidth)

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
        指定されたノードの隣接ノードを取得します。

        Args:
            node: ノードID

        Returns:
            隣接ノードIDのリスト

        Note:
            アリの経路選択時に、次の移動先候補として使用されます。
        """
        return list(self.graph.neighbors(node))

    def get_edge_attributes(self, u: int, v: int) -> Dict[str, float]:
        """
        エッジの属性（帯域、遅延、フェロモン）を取得します。

        Args:
            u: ノードuのID
            v: ノードvのID

        Returns:
            エッジ属性の辞書。キーは "bandwidth", "delay", "pheromone"

        Raises:
            KeyError: エッジ(u, v)が存在しない場合
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
        フェロモンを更新します。

        Args:
            u: ノードuのID
            v: ノードvのID
            delta_pheromone: 付加するフェロモン量（正の値）
            bidirectional: 双方向に更新するか。Trueの場合、エッジ(u, v)と(v, u)の両方に更新

        Note:
            - フェロモン値はmin_pheromoneとmax_pheromoneの範囲内に制限されます
            - 双方向更新により、無向グラフとしての対称性が保たれます
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
        全エッジのフェロモンを揮発させます。

        Args:
            evaporation_rate: 揮発率（0.0 ~ 1.0）。1.0に近いほど揮発が激しくなります

        Note:
            - 各エッジのフェロモン値が (1 - evaporation_rate) 倍されます
            - フェロモン値はmin_pheromoneを下回らないように制限されます
            - 世代終了時に呼び出され、古い情報を忘却します
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

    def reset_node_learning(self) -> None:
        """
        全ノードの学習値（BKB/BLD/BKH）をリセット

        スタートノード切り替え時などに使用
        """
        for node_learning in self.node_learning.values():
            node_learning.reset()

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
