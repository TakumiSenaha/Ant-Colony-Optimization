"""
アリ（Ant）クラス

ACOにおける探索エージェントを表現するモジュール。

【アリの役割】
ACOにおける探索エージェント。スタートノードからゴールノードまで探索し、
経路上の各エッジの品質（帯域、遅延）を記録する。

【主要機能】
1. 経路記憶（タブーリスト）：訪問済みノードを記録し、ループを防止
2. 累積メトリクス：ボトルネック帯域、累積遅延、ホップ数を追跡
3. エッジ単位の記録：フェロモン更新時に使用

【メトリクスの更新ロジック】
- ボトルネック帯域：min演算（経路上の最小帯域が全体のボトルネック）
- 累積遅延：加算（経路上の全エッジの遅延の合計）
- ホップ数：カウント（経路上のエッジ数）
"""

from typing import List, Tuple


class Ant:
    """
    ACOにおけるアリを表現するクラス

    各アリは独立して探索を行い、経路上の情報を記録します。
    ゴールに到達したアリは、経路上の各エッジにフェロモンを付加します。

    Attributes:
        ant_id (int): アリの識別子
        start_node (int): 開始ノードID
        destination_node (int): 目的地ノードID
        current_node (int): 現在のノードID
        ttl (int): Time To Live（最大ステップ数）
        remaining_ttl (int): 残りのTTL
        route (List[int]): 訪問済みノードのリスト（タブーリスト）
        min_bandwidth (float): 経路上の最小帯域（ボトルネック帯域、Mbps）
        total_delay (float): 累積遅延（ms）
        hop_count (int): ホップ数（経路上のエッジ数）
        bandwidth_log (List[float]): 各エッジの帯域の記録（フェロモン更新用）
        delay_log (List[float]): 各エッジの遅延の記録（フェロモン更新用）

    Example:
        >>> ant = Ant(ant_id=0, start_node=0, destination_node=10, ttl=100)
        >>> ant.move_to(next_node=1, bandwidth=100.0, delay=5.0)
        >>> ant.has_reached_goal()
        False
        >>> ant.get_solution()
        (100.0, 5.0, 1)
    """

    def __init__(
        self, ant_id: int, start_node: int, destination_node: int, ttl: int = 100
    ):
        """
        アリを初期化します。

        Args:
            ant_id: アリの識別子（世代内で一意）
            start_node: 開始ノードID
            destination_node: 目的地ノードID
            ttl: Time To Live（最大ステップ数）。このステップ数を超えるとアリは死亡します。

        Note:
            - 初期状態では、start_nodeがrouteに追加されます
            - min_bandwidthは無限大で初期化され、最初の移動で更新されます
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
        次のノードへ移動し、メトリクスを更新します。

        【メトリクス更新のロジック】
        - ボトルネック帯域：min演算（経路上の最小帯域が全体のボトルネック）
        - 累積遅延：加算（経路上の全エッジの遅延の合計）
        - ホップ数：カウント（経路上のエッジ数）

        Args:
            next_node: 移動先ノードID
            bandwidth: 移動に使用したエッジの帯域（Mbps）
            delay: 移動に使用したエッジの遅延（ms）

        Note:
            - next_nodeはrouteに追加され、タブーリストに含まれます
            - remaining_ttlが1減少します
            - bandwidth_logとdelay_logに記録が追加されます
        """
        self.route.append(next_node)
        self.current_node = next_node
        self.remaining_ttl -= 1

        # 【メトリクスの更新】
        # ボトルネック帯域：min演算（経路上の最小帯域が全体のボトルネック）
        self.min_bandwidth = min(self.min_bandwidth, bandwidth)
        # 累積遅延：加算（経路上の全エッジの遅延の合計）
        self.total_delay += delay
        # ホップ数：カウント（経路上のエッジ数）
        self.hop_count += 1

        # 【エッジ単位の記録】フェロモン更新時に使用
        self.bandwidth_log.append(bandwidth)
        self.delay_log.append(delay)

    def has_visited(self, node: int) -> bool:
        """
        指定されたノードを訪問済みかチェックします（タブーリスト）。

        Args:
            node: チェックするノードID

        Returns:
            訪問済みならTrue、未訪問ならFalse

        Note:
            タブーリスト（route）に含まれるノードは訪問済みとみなされます。
            これにより、ループを防止します。
        """
        return node in self.route

    def is_alive(self) -> bool:
        """
        アリが生存しているかチェックします。

        Returns:
            TTLが残っていればTrue、TTLが0以下ならFalse

        Note:
            remaining_ttlが0以下になると、アリは死亡し、それ以上移動できません。
        """
        return self.remaining_ttl > 0

    def has_reached_goal(self) -> bool:
        """
        目的地に到達したかチェックします。

        Returns:
            目的地に到達していればTrue、未到達ならFalse

        Note:
            current_nodeがdestination_nodeと一致する場合、目的地に到達したとみなします。
        """
        return self.current_node == self.destination_node

    def get_solution(self) -> Tuple[float, float, int]:
        """
        アリが見つけた解（帯域, 遅延, ホップ数）を取得します。

        Returns:
            (min_bandwidth, total_delay, hop_count)のタプル
            - min_bandwidth: ボトルネック帯域（Mbps）。infの場合は0.0を返します
            - total_delay: 累積遅延（ms）
            - hop_count: ホップ数

        Note:
            この解は、フェロモン更新やノードの学習値更新に使用されます。
        """
        return (
            self.min_bandwidth if self.min_bandwidth != float("inf") else 0.0,
            self.total_delay,
            self.hop_count,
        )

    def get_route_edges(self) -> List[Tuple[int, int]]:
        """
        経路のエッジリストを取得します。

        Returns:
            [(node_i, node_j), ...] のリスト。routeの連続するノードペアをエッジとして返します。

        Example:
            >>> ant = Ant(0, 0, 10)
            >>> ant.route = [0, 1, 2, 3]
            >>> ant.get_route_edges()
            [(0, 1), (1, 2), (2, 3)]

        Note:
            このメソッドは、フェロモン更新時に経路上の各エッジを特定するために使用されます。
        """
        return [(self.route[i], self.route[i + 1]) for i in range(len(self.route) - 1)]

    def __repr__(self) -> str:
        return (
            f"Ant(id={self.ant_id}, current={self.current_node}, "
            f"route_len={len(self.route)}, ttl={self.remaining_ttl}, "
            f"B={self.min_bandwidth:.1f}, D={self.total_delay:.1f}, H={self.hop_count})"
        )
