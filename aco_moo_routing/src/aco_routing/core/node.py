"""
ノードの自律学習モジュール

各ノードが「自分を通ってゴールしたアリの功績」をリングバッファで学習します。
これにより、ノードは分散的に「このノードを通れば、この品質でゴールできる」という情報を記憶します。

【自律学習の概念】
各ノードは独立して学習を行い、他のノードと情報を共有しません。
これにより、完全分散型の学習が実現されます。

【学習指標】
- BKB (Best Known Bottleneck): バッファ内の最大値（帯域は大きいほど良い）
- BLD (Best Known Lowest Delay): バッファ内の最小値（遅延は小さいほど良い）
- BKH (Best Known Hops): バッファ内の最小値（ホップ数は小さいほど良い）

【リングバッファの役割】
- 直近N個の観測値を記憶（window_sizeで指定）
- 古い観測値は自動的に忘却される（FIFO方式）
- これにより、動的環境への適応が可能
"""

from collections import deque
from typing import Dict


class NodeLearning:
    """
    ノードの分散学習機能を提供するクラス

    各ノードが独立して学習を行い、通過したアリの解品質を記憶します。
    リングバッファを使用して、直近N個の観測値を保持します。

    Attributes:
        node_id (int): ノードID
        window_size (int): リングバッファサイズ（直近N個の観測値を記憶）
        bandwidth_buffer (deque[float]): 帯域の観測値のリングバッファ
        delay_buffer (deque[float]): 遅延の観測値のリングバッファ
        hops_buffer (deque[int]): ホップ数の観測値のリングバッファ
        score_buffer (deque[float]): スコアの観測値のリングバッファ（遅延制約が有効な場合）
        bkb (float): Best Known Bottleneck（現在の最大値）
        bld (float): Best Known Lowest Delay（現在の最小値）
        bkh (float): Best Known Hops（現在の最小値）

    Example:
        >>> node_learning = NodeLearning(node_id=0, window_size=10)
        >>> node_learning.update_bandwidth(100.0)
        True
        >>> node_learning.bkb
        100.0
    """

    def __init__(self, node_id: int, window_size: int = 10):
        """
        ノードの学習機能を初期化します。

        Args:
            node_id: ノードID（識別用）
            window_size: リングバッファサイズ（直近N個の観測値を記憶）

        Note:
            - 初期状態では、bkb=0.0、bld=inf、bkh=infで初期化されます
            - リングバッファはdequeを使用し、maxlenで自動的に古い値を削除します
        """
        self.node_id = node_id
        self.window_size = window_size

        # リングバッファ（dequeを使用してFIFO）
        self.bandwidth_buffer: deque[float] = deque(maxlen=window_size)
        self.delay_buffer: deque[float] = deque(maxlen=window_size)
        self.hops_buffer: deque[int] = deque(maxlen=window_size)
        self.score_buffer: deque[float] = deque(
            maxlen=window_size
        )  # bandwidth/delayスコア（遅延制約が有効な場合）

        # 現在のベスト値（キャッシュ）
        self.bkb: float = (
            0.0  # Best Known Bottleneck（帯域のみ最適化：帯域の最大値、遅延制約あり：bandwidth/delayスコアの最大値）
        )
        self.bld: float = float("inf")  # Best Known Lowest Delay
        self.bkh: float = float("inf")  # Best Known Hops（比較用なのでfloat）

    def update_bandwidth(self, bottleneck: float) -> bool:
        """
        ボトルネック帯域を観測し、BKBを更新します。

        【更新プロセス】
        1. 新しい観測値をリングバッファに追加
        2. バッファ内の最大値をBKBとして更新
        3. 帯域は10Mbps刻みなので整数として扱う

        Args:
            bottleneck: 新しく観測されたボトルネック帯域（Mbps）

        Returns:
            BKBが更新されたかどうか（True: 更新された、False: 更新されなかった）

        Note:
            - 帯域は大きいほど良いため、最大値をBKBとして保持します
            - 帯域は10Mbps刻みなので、整数値に変換してから保存します
        """
        old_bkb = self.bkb
        self.bandwidth_buffer.append(bottleneck)
        # バッファ内の最大値を取得（帯域は大きいほど良い）
        time_window_max = max(self.bandwidth_buffer) if self.bandwidth_buffer else 0.0
        # 【既存実装との互換性】既存実装ではintとして保存されているため、intとして保存
        # ただし、float型の属性として保持するため、float(int(...))として保存
        # 既存実装: graph.nodes[node]["best_known_bottleneck"] = int(time_window_max)
        self.bkb = float(int(time_window_max))
        return self.bkb > old_bkb

    def update_delay(self, total_delay: float) -> bool:
        """
        累積遅延を観測し、BLDを更新します。

        Args:
            total_delay: 新しく観測された累積遅延（ms）

        Returns:
            BLDが更新されたかどうか（True: 更新された、False: 更新されなかった）

        Note:
            - 遅延は小さいほど良いため、最小値をBLDとして保持します
            - バッファが空の場合はinfを返します
        """
        old_bld = self.bld
        self.delay_buffer.append(total_delay)
        self.bld = min(self.delay_buffer) if self.delay_buffer else float("inf")
        return self.bld < old_bld

    def update_hops(self, hop_count: int) -> bool:
        """
        ホップ数を観測し、BKHを更新します。

        Args:
            hop_count: 新しく観測されたホップ数

        Returns:
            BKHが更新されたかどうか（True: 更新された、False: 更新されなかった）

        Note:
            - ホップ数は小さいほど良いため、最小値をBKHとして保持します
            - バッファが空の場合はinfを返します
        """
        old_bkh = self.bkh
        self.hops_buffer.append(hop_count)
        self.bkh = min(self.hops_buffer) if self.hops_buffer else float("inf")
        return self.bkh < old_bkh

    def update_score(self, score: float) -> bool:
        """
        bandwidth/delayスコアを観測し、BKBを更新（遅延制約が有効な場合）

        【更新プロセス】
        1. 新しいスコアをリングバッファに追加
        2. バッファ内の最大値をBKBとして更新

        Args:
            score: 新しく観測されたbandwidth/delayスコア

        Returns:
            BKBが更新されたかどうか
        """
        old_bkb = self.bkb
        self.score_buffer.append(score)
        # バッファ内の最大値を取得（スコアは大きいほど良い）
        time_window_max = max(self.score_buffer) if self.score_buffer else 0.0
        self.bkb = time_window_max
        return self.bkb > old_bkb

    def update_all(
        self, bottleneck: float, total_delay: float, hop_count: int
    ) -> Dict[str, bool]:
        """
        全ての指標を一括更新（帯域のみ最適化の場合）

        Args:
            bottleneck: ボトルネック帯域（Mbps）
            total_delay: 累積遅延（ms）
            hop_count: ホップ数

        Returns:
            各指標が更新されたかどうかの辞書
        """
        return {
            "bandwidth": self.update_bandwidth(bottleneck),
            "delay": self.update_delay(total_delay),
            "hops": self.update_hops(hop_count),
        }

    def evaporate(self, evaporation_rate: float) -> None:
        """
        BKB/BLD/BKH値を揮発させる（動的環境への適応）

        【揮発の方向性】
        - BKB（帯域またはスコア）：減少（悪化方向）→ 環境変化により帯域が低下した可能性を反映
        - BLD（遅延）：増加（悪化方向）→ 環境変化により遅延が増加した可能性を反映
        - BKH（ホップ数）：増加（悪化方向）→ 環境変化により経路が長くなった可能性を反映

        これにより、古い学習値が自動的に忘却され、新しい環境に適応できる。

        【既存実装との互換性】
        既存実装（src/bkb_learning.py）では、evaporation_rateが残存率（0.999）として
        直接使用されています。新実装でも同じように、残存率として扱います。

        Args:
            evaporation_rate: 揮発率（0.0 ~ 1.0）
                             既存実装では残存率（0.999）として使用されているため、
                             新実装では揮発率（0.001）から残存率（0.999）に変換して使用
        """
        # 【既存実装との互換性】残存率として扱う
        # 既存実装: evaporation_rate = 0.999（残存率）
        # 新実装: evaporation_rate = 0.001（揮発率）→ 残存率 = 1 - 0.001 = 0.999
        retention_rate = 1.0 - evaporation_rate  # 残存率に変換

        # BKBは揮発により減少（悪化方向）
        # 帯域のみ最適化：帯域の最大値
        # 遅延制約が有効：bandwidth/delayスコアの最大値
        # 既存実装と同じ: new_value = old_value * retention_rate
        self.bkb = self.bkb * retention_rate

        # BLDとBKHは揮発により増加（悪化方向、忘却）
        if self.bld != float("inf"):
            self.bld = self.bld * (1.0 + evaporation_rate)
        if self.bkh != float("inf"):
            self.bkh = int(self.bkh * (1.0 + evaporation_rate))

    def reset(self) -> None:
        """
        ノードの学習値をリセット（スタートノード切り替え時などに使用）

        リングバッファもクリアし、全ての値を初期状態に戻す
        """
        self.bandwidth_buffer.clear()
        self.delay_buffer.clear()
        self.hops_buffer.clear()
        self.score_buffer.clear()
        self.bkb = 0.0
        self.bld = float("inf")
        self.bkh = float("inf")

    def get_memory_state(self) -> Dict:
        """
        現在の記憶状態を取得（デバッグ用）

        Returns:
            記憶状態の辞書
        """
        return {
            "bkb": self.bkb,
            "bld": self.bld if self.bld != float("inf") else -1,
            "bkh": self.bkh if self.bkh != float("inf") else -1,
            "buffer_size": {
                "bandwidth": len(self.bandwidth_buffer),
                "delay": len(self.delay_buffer),
                "hops": len(self.hops_buffer),
            },
        }

    def __repr__(self) -> str:
        return (
            f"NodeLearning(id={self.node_id}, "
            f"BKB={self.bkb:.2f}, BLD={self.bld:.2f}, BKH={self.bkh})"
        )
