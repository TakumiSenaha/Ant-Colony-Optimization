"""
アリ（Ant）クラス

経路メモリ、累積メトリクス、タブーリストを管理します。
"""

from typing import List, Tuple


class Ant:
    """ACOにおけるアリを表現するクラス"""

    def __init__(
        self, ant_id: int, start_node: int, destination_node: int, ttl: int = 100
    ):
        """
        Args:
            ant_id: アリの識別子
            start_node: 開始ノード
            destination_node: 目的地ノード
            ttl: Time To Live（最大ステップ数）
        """
        self.ant_id = ant_id
        self.start_node = start_node
        self.destination_node = destination_node
        self.current_node = start_node
        self.ttl = ttl
        self.remaining_ttl = ttl

        # 経路記憶（タブーリスト）
        self.route: List[int] = [start_node]

        # 累積メトリクス
        self.min_bandwidth: float = float("inf")  # 経路上の最小帯域（ボトルネック）
        self.total_delay: float = 0.0  # 累積遅延
        self.hop_count: int = 0  # ホップ数

        # エッジ単位の記録（フェロモン更新用）
        self.bandwidth_log: List[float] = []  # 各エッジの帯域
        self.delay_log: List[float] = []  # 各エッジの遅延

    def move_to(self, next_node: int, bandwidth: float, delay: float) -> None:
        """
        次のノードへ移動し、メトリクスを更新

        Args:
            next_node: 移動先ノード
            bandwidth: 移動に使用したエッジの帯域（Mbps）
            delay: 移動に使用したエッジの遅延（ms）
        """
        self.route.append(next_node)
        self.current_node = next_node
        self.remaining_ttl -= 1

        # メトリクスの更新
        self.min_bandwidth = min(self.min_bandwidth, bandwidth)
        self.total_delay += delay
        self.hop_count += 1

        # ログの記録
        self.bandwidth_log.append(bandwidth)
        self.delay_log.append(delay)

    def has_visited(self, node: int) -> bool:
        """
        指定されたノードを訪問済みかチェック（タブーリスト）

        Args:
            node: チェックするノード

        Returns:
            訪問済みならTrue
        """
        return node in self.route

    def is_alive(self) -> bool:
        """
        アリが生存しているかチェック

        Returns:
            TTLが残っていればTrue
        """
        return self.remaining_ttl > 0

    def has_reached_goal(self) -> bool:
        """
        目的地に到達したかチェック

        Returns:
            目的地に到達していればTrue
        """
        return self.current_node == self.destination_node

    def get_solution(self) -> Tuple[float, float, int]:
        """
        アリが見つけた解（帯域, 遅延, ホップ数）を取得

        Returns:
            (min_bandwidth, total_delay, hop_count)
        """
        return (
            self.min_bandwidth if self.min_bandwidth != float("inf") else 0.0,
            self.total_delay,
            self.hop_count,
        )

    def get_route_edges(self) -> List[Tuple[int, int]]:
        """
        経路のエッジリストを取得

        Returns:
            [(node_i, node_j), ...] のリスト
        """
        return [(self.route[i], self.route[i + 1]) for i in range(len(self.route) - 1)]

    def __repr__(self) -> str:
        return (
            f"Ant(id={self.ant_id}, current={self.current_node}, "
            f"route_len={len(self.route)}, ttl={self.remaining_ttl}, "
            f"B={self.min_bandwidth:.1f}, D={self.total_delay:.1f}, H={self.hop_count})"
        )
