"""
評価関数モジュール

多目的最適化における評価関数 f(B, D, H) を計算します。
"""

from typing import List, Tuple


class SolutionEvaluator:
    """
    解の評価を行うクラス

    Attributes:
        target_objectives (List[str]): 目的関数のリスト（例: ["bandwidth", "delay"]）
    """

    def __init__(self, target_objectives: List[str]):
        """
        Args:
            target_objectives: 目的関数のリスト ["bandwidth", "delay", "hops"]
        """
        self.target_objectives = target_objectives

    def evaluate(self, bandwidth: float, delay: float, hops: int) -> float:
        """
        解を評価し、スコアを計算

        Args:
            bandwidth: ボトルネック帯域（Mbps）
            delay: 累積遅延（ms）
            hops: ホップ数

        Returns:
            評価スコア（大きいほど良い）
        """
        objectives = self.target_objectives

        # 帯域のみ最適化（既存実装との互換性）
        if objectives == ["bandwidth"]:
            return self._evaluate_bandwidth_only(bandwidth)

        # Step 1: Bandwidth vs Hops
        elif set(objectives) == {"bandwidth", "hops"}:
            return self._evaluate_bandwidth_hops(bandwidth, hops)

        # Step 2: Bandwidth vs Delay
        elif set(objectives) == {"bandwidth", "delay"}:
            return self._evaluate_bandwidth_delay(bandwidth, delay)

        # Step 3: Bandwidth vs Delay vs Hops
        elif set(objectives) == {"bandwidth", "delay", "hops"}:
            return self._evaluate_bandwidth_delay_hops(bandwidth, delay, hops)

        else:
            raise ValueError(f"Unknown objectives: {objectives}")

    def _evaluate_bandwidth_only(self, bandwidth: float) -> float:
        """
        帯域のみの評価

        Args:
            bandwidth: ボトルネック帯域（Mbps）

        Returns:
            評価スコア（bandwidth）
        """
        return bandwidth

    def _evaluate_bandwidth_hops(self, bandwidth: float, hops: int) -> float:
        """
        帯域とホップ数の評価

        Args:
            bandwidth: ボトルネック帯域（Mbps）
            hops: ホップ数

        Returns:
            評価スコア（bandwidth / hops）
        """
        if hops == 0:
            return 0.0
        return bandwidth / hops

    def _evaluate_bandwidth_delay(self, bandwidth: float, delay: float) -> float:
        """
        帯域と遅延の評価

        Args:
            bandwidth: ボトルネック帯域（Mbps）
            delay: 累積遅延（ms）

        Returns:
            評価スコア
        """
        if delay <= 0:
            return 0.0
        return bandwidth / delay

    def _evaluate_bandwidth_delay_hops(
        self, bandwidth: float, delay: float, hops: int
    ) -> float:
        """
        帯域、遅延、ホップ数の評価

        Args:
            bandwidth: ボトルネック帯域（Mbps）
            delay: 累積遅延（ms）
            hops: ホップ数

        Returns:
            評価スコア
        """
        if delay <= 0 or hops == 0:
            return 0.0
        return bandwidth / (delay * hops)

    def check_bonus_condition(
        self,
        ant_solution: Tuple[float, float, int],
        node_memory: Tuple[float, float, float],
        delay_tolerance: float = 5.0,
    ) -> bool:
        """
        功績ボーナスの条件をチェック（スコアベース判定）

        【核心】評価関数 f(B, D, H) によるスコア比較で判定。
        これにより、フェロモン更新とボーナス判定の基準が一致し、
        トレードオフ（帯域UP、遅延DOWNなど）を適切に評価できる。

        Args:
            ant_solution: アリの解 (bandwidth, delay, hops)
            node_memory: ノードの記憶 (BKB, BLD, BKH)
            delay_tolerance: 遅延の許容誤差（ms）（後方互換性のため保持、使用しない）

        Returns:
            ボーナス条件を満たすか（スコアが改善していればTrue）
        """
        b_ant, d_ant, h_ant = ant_solution
        k_j, l_j, m_j = node_memory

        # 帯域のみ最適化（既存実装との互換性）
        # 帯域のみの場合は、スコアベースでも bandwidth >= BKB と同じ結果になる
        if self.target_objectives == ["bandwidth"]:
            return b_ant >= k_j

        # 多目的最適化の場合：評価関数によるスコア比較
        # アリの解のスコア（×10はかけない、純粋なスコア）
        score_ant = self.evaluate(b_ant, d_ant, h_ant)

        # ノードの記憶値のスコア（仮想的な解として）
        # 注意: BLD/BKHがinfの場合は、帯域のみで評価
        # （infの値を使うとスコア計算ができないため）
        if l_j == float("inf") and m_j == float("inf"):
            # 帯域のみが有効な場合 → 帯域のみ最適化として扱う
            score_memory = self._evaluate_bandwidth_only(k_j)
        elif l_j == float("inf"):
            # 帯域とホップ数のみ（遅延はinfなので考慮しない）
            if self.target_objectives == ["bandwidth", "hops"]:
                score_memory = self._evaluate_bandwidth_hops(k_j, int(m_j))
            else:
                # 帯域のみで評価
                score_memory = self._evaluate_bandwidth_only(k_j)
        elif m_j == float("inf"):
            # 帯域と遅延のみ（ホップ数はinfなので考慮しない）
            if self.target_objectives == ["bandwidth", "delay"]:
                score_memory = self._evaluate_bandwidth_delay(k_j, l_j)
            else:
                # 帯域のみで評価
                score_memory = self._evaluate_bandwidth_only(k_j)
        else:
            # すべて有効
            score_memory = self.evaluate(k_j, l_j, int(m_j))

        # スコアが改善していればボーナス（×10はかけない、純粋なスコアで比較）
        return score_ant > score_memory
