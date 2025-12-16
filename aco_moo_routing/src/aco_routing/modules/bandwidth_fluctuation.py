"""
帯域・遅延変動モデル

AR(1)モデル等を用いて、動的なネットワーク環境を模擬します。
遅延は帯域と連動して変動します（物理的整合性）。
"""

import random
from typing import Dict, Tuple

import networkx as nx


class BandwidthFluctuationModel:
    """
    帯域変動モデルの基底クラス

    動的ネットワーク環境を模擬するためのインターフェースを提供します。

    Attributes:
        graph (nx.Graph): 帯域変動対象となるNetworkXグラフ
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def initialize_states(self, fluctuating_edges: list) -> Dict:
        """変動対象エッジの状態を初期化"""
        raise NotImplementedError

    def update(self, edge_states: Dict, generation: int) -> bool:
        """帯域と遅延を更新"""
        raise NotImplementedError


class AR1Model(BandwidthFluctuationModel):
    """
    AR(1)モデル: B_t = φ * B_{t-1} + (1-φ) * μ + ε

    - φ: AR係数（0.95で既存実装と同じ挙動）
    - μ: 平均利用率（帯域の平均利用割合）
    - ε: ガウス雑音（標準偏差 noise_std）
    """

    def __init__(
        self,
        graph: nx.Graph,
        phi: float = 0.95,  # 既存実装と同じ（AR_COEFFICIENT = 0.95）
        mean_utilization: float = 0.4,
        noise_std: float = 0.03123,  # sqrt(0.000975) ≈ 0.03123
    ):
        """
        Args:
            graph: ネットワークグラフ
            phi: 自己相関係数（0.0 ~ 1.0）既存実装のAR_COEFFICIENTに対応
            mean_utilization: 平均利用率
            noise_std: ノイズの標準偏差（既存実装のNOISE_VARIANCEの平方根）
        """
        super().__init__(graph)
        self.phi = phi
        self.mean_utilization = mean_utilization
        self.noise_std = noise_std

    def initialize_states(self, fluctuating_edges: list) -> Dict[Tuple[int, int], Dict]:
        """
        変動対象エッジの状態を初期化

        Args:
            fluctuating_edges: 変動対象エッジのリスト [(u, v), ...]

        Returns:
            エッジ状態の辞書 {(u, v): {"utilization": float, "original_bandwidth": float}}
        """
        edge_states = {}
        for u, v in fluctuating_edges:
            # グラフ生成時に設定されたoriginal_bandwidthを使用
            original_bw = self.graph.edges[u, v].get(
                "original_bandwidth", self.graph.edges[u, v]["bandwidth"]
            )

            # 初期利用率をランダムに設定（既存実装と同じ）
            initial_util = random.uniform(0.3, 0.5)

            edge_states[(u, v)] = {
                "utilization": initial_util,
                "original_bandwidth": original_bw,
            }

            # 初期可用帯域を設定（既存実装と同じ）
            initial_available = int(round(original_bw * (1.0 - initial_util)))
            initial_available = ((initial_available + 5) // 10) * 10  # 10Mbps刻みに丸め
            self.graph.edges[u, v]["bandwidth"] = float(initial_available)
            self.graph.edges[v, u]["bandwidth"] = float(initial_available)

        return edge_states

    def update(self, edge_states: Dict[Tuple[int, int], Dict], generation: int) -> bool:
        """
        AR(1)モデルで帯域と遅延を更新

        Args:
            edge_states: エッジ状態の辞書
            generation: 現在の世代番号

        Returns:
            帯域が変動したかどうか
        """
        changed = False

        for (u, v), state in edge_states.items():
            # AR(1)モデルで利用率を更新（既存実装と同じ形式）
            current_utilization = state["utilization"]
            noise = random.gauss(0, self.noise_std)
            new_utilization = (
                (1 - self.phi) * self.mean_utilization  # 平均への回帰
                + self.phi * current_utilization  # 過去の値への依存
                + noise  # ランダムノイズ
            )
            # 利用率を0.05 - 0.95の範囲にクリップ（既存実装と同じ）
            new_utilization = max(0.05, min(0.95, new_utilization))

            # 可用帯域を更新（既存実装と同じ計算式）
            # ★重要★: capacity * (1 - utilization) = 可用帯域
            original_bw = state["original_bandwidth"]
            available_bandwidth = original_bw * (1.0 - new_utilization)

            # 10Mbps刻みに丸める（既存実装と同じ）
            available_bandwidth = ((int(available_bandwidth) + 5) // 10) * 10

            old_bandwidth = self.graph.edges[u, v]["bandwidth"]
            if available_bandwidth != old_bandwidth:
                self.graph.edges[u, v]["bandwidth"] = float(available_bandwidth)
                self.graph.edges[v, u]["bandwidth"] = float(
                    available_bandwidth
                )  # 双方向
                changed = True

                # 遅延は更新しない（既存実装と同じ、帯域のみ最適化のため）
                # self._update_delay(u, v, float(available_bandwidth))

            state["utilization"] = new_utilization

        return changed

    def _update_delay(self, u: int, v: int, bandwidth: float) -> None:
        """
        帯域に応じて遅延を更新

        物理的な整合性を保つため、帯域が下がると遅延が上がる。
        計算式: delay = base_delay / bandwidth_ratio + jitter

        Args:
            u: ノードu
            v: ノードv
            bandwidth: 新しい帯域（Mbps）
        """
        # オリジナルの遅延を基準とする
        if "original_delay" not in self.graph.edges[u, v]:
            self.graph.edges[u, v]["original_delay"] = self.graph.edges[u, v]["delay"]
            self.graph.edges[v, u]["original_delay"] = self.graph.edges[v, u]["delay"]

        original_delay = self.graph.edges[u, v]["original_delay"]
        original_bw = self.graph.edges[u, v].get("original_bandwidth", bandwidth)

        # 帯域比に応じて遅延を調整
        bandwidth_ratio = bandwidth / original_bw if original_bw > 0 else 1.0
        new_delay = original_delay / max(bandwidth_ratio, 0.1)

        # ジッター（ランダムな揺らぎ）を追加
        jitter = random.uniform(-0.5, 0.5)
        new_delay = max(0.1, new_delay + jitter)

        self.graph.edges[u, v]["delay"] = new_delay
        self.graph.edges[v, u]["delay"] = new_delay


def select_fluctuating_edges(
    graph: nx.Graph, method: str = "hub", percentage: float = 0.1
) -> list:
    """
    変動対象エッジを選択

    Args:
        graph: ネットワークグラフ
        method: 選択方法 ("hub", "random", "betweenness")
        percentage: 選択する割合

    Returns:
        変動対象エッジのリスト [(u, v), ...]
    """
    if method == "hub":
        # ハブノード（次数が大きいノード）の隣接エッジを選択
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        num_hubs = max(1, int(len(sorted_nodes) * percentage))
        hub_nodes = [node for node, _ in sorted_nodes[:num_hubs]]

        fluctuating_edges = []
        for u, v in graph.edges():
            if u in hub_nodes or v in hub_nodes:
                fluctuating_edges.append((u, v))
        return fluctuating_edges

    elif method == "random":
        # ランダムにエッジを選択
        all_edges = list(graph.edges())
        num_edges = max(1, int(len(all_edges) * percentage))
        return random.sample(all_edges, num_edges)

    elif method == "betweenness":
        # 媒介中心性が高いエッジを選択
        betweenness = nx.edge_betweenness_centrality(graph)
        sorted_edges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        num_edges = max(1, int(len(sorted_edges) * percentage))
        return [edge for edge, _ in sorted_edges[:num_edges]]

    else:
        raise ValueError(f"Unknown method: {method}")
