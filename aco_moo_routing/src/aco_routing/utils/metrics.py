"""
評価指標モジュール

パレート最適解への到達率、支配率、Hypervolume等を計算します。
"""

from typing import List, Optional, Tuple


class MetricsCalculator:
    """
    評価指標を計算するクラス

    - パレート到達率、支配率、Hypervolume、収束率を計算
    - 目的関数の組み合わせに依存せず利用可能

    Attributes:
        reference_point (List[float]): Hypervolume計算用の基準点 [bandwidth, delay, hops]
        target_objectives (List[str]): 最適化対象の目的関数のリスト
    """

    def __init__(
        self,
        reference_point: List[float],
        target_objectives: Optional[List[str]] = None,
    ):
        """
        Args:
            reference_point: Hypervolume計算用の基準点 [bandwidth, delay, hops]
            target_objectives: 最適化対象の目的関数のリスト ["bandwidth"], ["bandwidth", "delay"], etc.
        """
        self.reference_point = reference_point
        self.target_objectives = target_objectives or []

    def calculate_pareto_discovery_rate(
        self,
        aco_solutions: List[Tuple[float, float, int]],
        optimal_solutions: List[Tuple[float, float, int]],
    ) -> float:
        """
        最適解への到達率を計算（単一最適解でもパレートフロンティアでも同じ）

        定義: (ACOが見つけた解のうち、最適解と一致する数) / (最適解の総数)

        Args:
            aco_solutions: ACOが見つけた解のリスト [(bandwidth, delay, hops), ...]
            optimal_solutions: 最適解のリスト [(bandwidth, delay, hops), ...]

        Returns:
            最適解への到達率（0.0 ~ 1.0）
        """
        if not optimal_solutions:
            return 0.0

        # 最適解のセット
        optimal_set = set(optimal_solutions)

        # ACOが発見した最適解の数
        discovered = 0
        for solution in aco_solutions:
            # 完全一致（わずかな誤差を許容）
            for opt_solution in optimal_set:
                if (
                    abs(solution[0] - opt_solution[0]) < 0.01
                    and abs(solution[1] - opt_solution[1]) < 0.01
                    and abs(solution[2] - opt_solution[2]) < 0.01
                ):
                    discovered += 1
                    break

        return discovered / len(optimal_solutions)

    def calculate_dominance_rate(
        self,
        aco_solutions: List[Tuple[float, float, int]],
        optimal_solutions: List[Tuple[float, float, int]],
    ) -> float:
        """
        支配率を計算（単一最適解でもパレートフロンティアでも同じ）

        定義: (ACO解のうち、最適解に支配されない割合)

        Args:
            aco_solutions: ACOが見つけた解のリスト
            optimal_solutions: 最適解のリスト

        Returns:
            支配率（0.0 ~ 1.0）
        """
        if not aco_solutions:
            return 0.0

        non_dominated_count = 0
        for aco_sol in aco_solutions:
            if not self._is_dominated(aco_sol, optimal_solutions):
                non_dominated_count += 1

        return non_dominated_count / len(aco_solutions)

    def _is_dominated(
        self,
        solution: Tuple[float, float, int],
        optimal_solutions: List[Tuple[float, float, int]],
    ) -> bool:
        """
        解が最適解に支配されるか判定

        Args:
            solution: 解 (bandwidth, delay, hops)
            optimal_solutions: 最適解のリスト

        Returns:
            支配される場合True
        """
        b, d, h = solution
        for opt in optimal_solutions:
            opt_b, opt_d, opt_h = opt
            # 最適解が支配する条件
            if (opt_b >= b) and (opt_d <= d) and (opt_h <= h):
                # 少なくとも1つの項目で優れている
                if (opt_b > b) or (opt_d < d) or (opt_h < h):
                    return True
        return False

    def _is_pareto_optimal(
        self,
        solution: Tuple[float, float, int],
        optimal_solutions: List[Tuple[float, float, int]],
    ) -> bool:
        """
        解が最適解かどうかを判定

        Args:
            solution: 解 (bandwidth, delay, hops)
            optimal_solutions: 最適解のリスト

        Returns:
            最適解の場合True
        """
        b, d, h = solution

        # 帯域のみ最適化の場合：帯域幅だけを比較
        if self.target_objectives == ["bandwidth"]:
            for opt in optimal_solutions:
                opt_b, _, _ = opt
                if abs(b - opt_b) < 0.01:
                    return True
            return False

        # パレート最適化の場合：完全一致（わずかな誤差を許容）
        for opt in optimal_solutions:
            opt_b, opt_d, opt_h = opt
            if (
                abs(b - opt_b) < 0.01
                and abs(d - opt_d) < 0.01
                and abs(h - opt_h) < 0.01
            ):
                return True
        return False

    def find_optimal_solution_index(
        self,
        solution: Tuple[float, float, int],
        optimal_solutions: List[Tuple[float, float, int]],
    ) -> int:
        """
        解がどの最適解に一致するかを判定（インデックスを返す）

        Args:
            solution: 解 (bandwidth, delay, hops)
            optimal_solutions: 最適解のリスト

        Returns:
            最適解のインデックス（0-indexed）、一致しない場合は-1
        """
        b, d, h = solution

        # 帯域のみ最適化の場合：帯域幅だけを比較
        if self.target_objectives == ["bandwidth"]:
            for idx, opt in enumerate(optimal_solutions):
                opt_b, _, _ = opt
                if abs(b - opt_b) < 0.01:
                    return idx
            return -1

        # パレート最適化の場合：完全一致（わずかな誤差を許容）
        for idx, opt in enumerate(optimal_solutions):
            opt_b, opt_d, opt_h = opt
            if (
                abs(b - opt_b) < 0.01
                and abs(d - opt_d) < 0.01
                and abs(h - opt_h) < 0.01
            ):
                return idx
        return -1

    def extract_pareto_frontier(
        self, solutions: List[Tuple[float, float, int]]
    ) -> List[Tuple[float, float, int]]:
        """
        ACOが見つけた解から、パレートフロンティア（支配されていない解の集合）を抽出

        Args:
            solutions: ACOが見つけた解のリスト [(bandwidth, delay, hops), ...]

        Returns:
            パレートフロンティア（支配されていない解のリスト）
        """
        if not solutions:
            return []

        pareto_frontier = []
        for solution in solutions:
            # 他のすべての解と比較
            is_dominated = False
            for other in solutions:
                if solution == other:
                    continue
                # otherがsolutionを支配するかチェック
                if self._dominates(other, solution):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_frontier.append(solution)

        return pareto_frontier

    def _dominates(
        self, solution1: Tuple[float, float, int], solution2: Tuple[float, float, int]
    ) -> bool:
        """
        solution1がsolution2を支配するか判定

        Args:
            solution1: 解1 (bandwidth, delay, hops)
            solution2: 解2 (bandwidth, delay, hops)

        Returns:
            solution1がsolution2を支配する場合True
        """
        b1, d1, h1 = solution1
        b2, d2, h2 = solution2

        # 帯域のみ最適化の場合：帯域幅だけを比較
        if self.target_objectives == ["bandwidth"]:
            return b1 > b2

        # 多目的最適化の場合：すべての目的関数で同等以上、かつ少なくとも1つで優れている
        bandwidth_ge = b1 >= b2
        delay_le = d1 <= d2
        hops_le = h1 <= h2

        if not (bandwidth_ge and delay_le and hops_le):
            return False

        # 少なくとも1つの項目で優れている
        bandwidth_gt = b1 > b2
        delay_lt = d1 < d2
        hops_lt = h1 < h2

        return bandwidth_gt or delay_lt or hops_lt

    def calculate_hypervolume(self, solutions: List[Tuple[float, float, int]]) -> float:
        """
        Hypervolumeを計算（3目的の場合）

        【重要】パレートフロンティア（支配されていない解の集合）に対して計算します。

        Args:
            solutions: 解のリスト [(bandwidth, delay, hops), ...]
                      （ACOが見つけた解。内部でパレートフロンティアを抽出します）

        Returns:
            Hypervolume
        """
        if not solutions:
            return 0.0

        # ACOが見つけた解から、パレートフロンティアを抽出
        pareto_frontier = self.extract_pareto_frontier(solutions)

        if not pareto_frontier:
            return 0.0

        # 基準点
        ref_b, ref_d, ref_h = self.reference_point

        # 正規化（目的関数の方向を揃える）
        # bandwidth: 最大化 → そのまま
        # delay: 最小化 → ref_d - delay
        # hops: 最小化 → ref_h - hops

        normalized_solutions = []
        for b, d, h in pareto_frontier:
            norm_b = b
            norm_d = ref_d - d
            norm_h = ref_h - h

            # 基準点より悪い解は除外
            if norm_b > 0 and norm_d > 0 and norm_h > 0:
                normalized_solutions.append((norm_b, norm_d, norm_h))

        if not normalized_solutions:
            return 0.0

        # 簡易的なHypervolume計算（厳密ではないが近似）
        # 各解の体積を足し合わせる
        total_volume = 0.0
        for norm_b, norm_d, norm_h in normalized_solutions:
            volume = norm_b * norm_d * norm_h
            total_volume += volume

        return total_volume

    def calculate_convergence_rate(
        self, results: List[dict], optimal_solutions: List[Tuple[float, float, int]]
    ) -> List[float]:
        """
        世代ごとの収束率を計算

        定義: 各世代での最適解への到達率

        Args:
            results: ACOの結果リスト（世代ごと）
            optimal_solutions: 最適解のリスト

        Returns:
            世代ごとの収束率のリスト
        """
        convergence_rates = []
        for result in results:
            solutions = result["solutions"]
            rate = self.calculate_pareto_discovery_rate(solutions, optimal_solutions)
            convergence_rates.append(rate)
        return convergence_rates

    def calculate_pch_at_k(
        self,
        aco_ranking: List[Tuple[Tuple[float, float, int], float, int]],
        optimal_solutions: List[Tuple[float, float, int]],
        k: int,
    ) -> float:
        """
        PCH@K (Pareto Coverage at K) を計算

        PCH@K = (一致した真のパレート解の数 / K) × 100

        Args:
            aco_ranking: ACOが見つけた解のランキング [(path, selection_rate, count), ...]
                        選択率で降順にソート済み
            optimal_solutions: 真のパレート最適解のリスト
            k: 上位K個の解を評価

        Returns:
            PCH@K (0.0 ~ 100.0)
        """
        if not optimal_solutions or k == 0:
            return 0.0

        # 上位K個の解を取得
        top_k_solutions = [path for path, _, _ in aco_ranking[:k]]

        # 上位K個の解のうち、最適解と一致する数をカウント
        matched_count = 0
        for path in top_k_solutions:
            b, d, h = path
            for opt_b, opt_d, opt_h in optimal_solutions:
                # 完全一致（誤差を許容）
                bandwidth_match = abs(b - opt_b) < max(0.01, opt_b * 0.01)
                delay_match = abs(d - opt_d) < max(0.1, opt_d * 0.01)
                hops_match = h == opt_h
                if bandwidth_match and delay_match and hops_match:
                    matched_count += 1
                    break

        # PCH@K = (一致した真のパレート解の数 / K) × 100
        pch_at_k = (matched_count / k) * 100
        return pch_at_k
