"""
ノードの自律学習モジュール

各ノードは「自分を通ってゴールしたアリの功績」をリングバッファで学習します。
- BKB (Best Known Bottleneck): バッファ内の最大値
- BLD (Best Known Lowest Delay): バッファ内の最小値
- BKH (Best Known Hops): バッファ内の最小値
"""

from collections import deque
from typing import Dict


class NodeLearning:
    """ノードの分散学習機能を提供するクラス"""

    def __init__(self, node_id: int, window_size: int = 10):
        """
        Args:
            node_id: ノードID
            window_size: リングバッファサイズ（直近N個の観測値を記憶）
        """
        self.node_id = node_id
        self.window_size = window_size

        # リングバッファ（dequeを使用してFIFO）
        self.bandwidth_buffer: deque[float] = deque(maxlen=window_size)
        self.delay_buffer: deque[float] = deque(maxlen=window_size)
        self.hops_buffer: deque[int] = deque(maxlen=window_size)

        # 現在のベスト値（キャッシュ）
        self.bkb: float = 0.0  # Best Known Bottleneck
        self.bld: float = float("inf")  # Best Known Lowest Delay
        self.bkh: float = float("inf")  # Best Known Hops（比較用なのでfloat）

    def update_bandwidth(self, bottleneck: float) -> bool:
        """
        ボトルネック帯域を観測し、BKBを更新

        Args:
            bottleneck: 新しく観測されたボトルネック帯域（Mbps）

        Returns:
            BKBが更新されたかどうか
        """
        old_bkb = self.bkb
        self.bandwidth_buffer.append(bottleneck)
        # 既存実装と同じく int() で変換（帯域は10Mbps刻みなので整数として扱う）
        time_window_max = max(self.bandwidth_buffer) if self.bandwidth_buffer else 0.0
        self.bkb = float(int(time_window_max))
        return self.bkb > old_bkb

    def update_delay(self, total_delay: float) -> bool:
        """
        累積遅延を観測し、BLDを更新

        Args:
            total_delay: 新しく観測された累積遅延（ms）

        Returns:
            BLDが更新されたかどうか
        """
        old_bld = self.bld
        self.delay_buffer.append(total_delay)
        self.bld = min(self.delay_buffer) if self.delay_buffer else float("inf")
        return self.bld < old_bld

    def update_hops(self, hop_count: int) -> bool:
        """
        ホップ数を観測し、BKHを更新

        Args:
            hop_count: 新しく観測されたホップ数

        Returns:
            BKHが更新されたかどうか
        """
        old_bkh = self.bkh
        self.hops_buffer.append(hop_count)
        self.bkh = min(self.hops_buffer) if self.hops_buffer else float("inf")
        return self.bkh < old_bkh

    def update_all(
        self, bottleneck: float, total_delay: float, hop_count: int
    ) -> Dict[str, bool]:
        """
        全ての指標を一括更新

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

        Args:
            evaporation_rate: 揮発率（0.0 ~ 1.0）
        """
        # BKBは揮発により減少
        self.bkb = self.bkb * (1.0 - evaporation_rate)

        # BLDとBKHは揮発により増加（忘却）
        if self.bld != float("inf"):
            self.bld = self.bld * (1.0 + evaporation_rate)
        if self.bkh != float("inf"):
            self.bkh = int(self.bkh * (1.0 + evaporation_rate))

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
