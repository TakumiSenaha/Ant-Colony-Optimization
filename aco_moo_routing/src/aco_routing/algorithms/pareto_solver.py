"""
パレートフロンティア計算モジュール

多目的ラベリング法を用いて、厳密なパレート最適解を計算します。
これはACOの評価における「正解データ」として使用されます。
"""

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, List, Tuple

import networkx as nx


@dataclass
class Label:
    """ノードに到達する経路のラベル（帯域, 遅延, ホップ数, 経路）"""

    bandwidth: float  # ボトルネック帯域（最大化したい）
    delay: float  # 累積遅延（最小化したい）
    hops: int  # ホップ数（最小化したい）
    path: List[int]  # 経路

    def dominates(self, other: "Label") -> bool:
        """
        このラベルが他のラベルを支配するか判定

        支配の定義:
        - bandwidth >= other.bandwidth
        - delay <= other.delay
        - hops <= other.hops
        - かつ、少なくとも1つの項目で不等号が成立

        Args:
            other: 比較対象のラベル

        Returns:
            支配する場合True
        """
        bandwidth_ge = self.bandwidth >= other.bandwidth
        delay_le = self.delay <= other.delay
        hops_le = self.hops <= other.hops

        # 全ての項目で同等以上
        if not (bandwidth_ge and delay_le and hops_le):
            return False

        # 少なくとも1つの項目で優れている
        bandwidth_gt = self.bandwidth > other.bandwidth
        delay_lt = self.delay < other.delay
        hops_lt = self.hops < other.hops

        return bandwidth_gt or delay_lt or hops_lt

    def is_dominated_by_any(self, labels: List["Label"]) -> bool:
        """
        ラベルリストのいずれかに支配されるか判定

        Args:
            labels: ラベルのリスト

        Returns:
            支配される場合True
        """
        return any(label.dominates(self) for label in labels)

    def __lt__(self, other: "Label") -> bool:
        """優先度キューのための比較演算子（帯域が大きい方が優先）"""
        return self.bandwidth > other.bandwidth


class ParetoSolver:
    """パレートフロンティアを計算するクラス"""

    def __init__(self, graph: nx.Graph, max_labels_per_node: int = 1000):
        """
        Args:
            graph: ネットワークグラフ
            max_labels_per_node: 各ノードで保持する最大ラベル数（メモリ制約）
        """
        self.graph = graph
        self.max_labels_per_node = max_labels_per_node

    def find_pareto_frontier(
        self, source: int, destination: int
    ) -> List[Tuple[float, float, int, List[int]]]:
        """
        パレートフロンティアを計算

        Args:
            source: 開始ノード
            destination: 目的地ノード

        Returns:
            パレート最適解のリスト [(bandwidth, delay, hops, path), ...]
        """
        # 各ノードのラベルリスト
        labels: Dict[int, List[Label]] = {node: [] for node in self.graph.nodes()}

        # 優先度キュー
        pq: List[Label] = []

        # 初期ラベル（開始ノードに無限大の帯域で到達）
        initial_label = Label(bandwidth=float("inf"), delay=0.0, hops=0, path=[source])
        heappush(pq, initial_label)
        labels[source].append(initial_label)

        # 訪問済みラベルを追跡（重複探索を防ぐ）
        visited_labels: Dict[int, List[Label]] = {
            node: [] for node in self.graph.nodes()
        }

        while pq:
            current_label = heappop(pq)
            current_node = current_label.path[-1]

            # 既に訪問済みのラベルに支配されている場合はスキップ
            if current_label.is_dominated_by_any(visited_labels[current_node]):
                continue

            visited_labels[current_node].append(current_label)

            # 目的地に到達した場合はこのラベルを保持（探索は続行）
            if current_node == destination:
                continue

            # 隣接ノードへ探索
            for neighbor in self.graph.neighbors(current_node):
                # 経路に既に含まれている場合はスキップ（ループ回避）
                if neighbor in current_label.path:
                    continue

                # エッジの属性を取得
                edge_bandwidth = self.graph.edges[current_node, neighbor]["bandwidth"]
                edge_delay = self.graph.edges[current_node, neighbor]["delay"]

                # 新しいラベルを生成
                new_bandwidth = min(current_label.bandwidth, edge_bandwidth)
                new_delay = current_label.delay + edge_delay
                new_hops = current_label.hops + 1
                new_path = current_label.path + [neighbor]

                new_label = Label(
                    bandwidth=new_bandwidth,
                    delay=new_delay,
                    hops=new_hops,
                    path=new_path,
                )

                # 既存のラベルと比較
                dominated = new_label.is_dominated_by_any(labels[neighbor])
                if dominated:
                    continue  # 既存ラベルに支配されている場合は破棄

                # 新しいラベルが既存ラベルを支配する場合、既存ラベルを削除
                labels[neighbor] = [
                    label
                    for label in labels[neighbor]
                    if not new_label.dominates(label)
                ]

                # 新しいラベルを追加
                labels[neighbor].append(new_label)

                # メモリ制約: ラベル数が多すぎる場合は剪定
                if len(labels[neighbor]) > self.max_labels_per_node:
                    labels[neighbor] = self._prune_labels(
                        labels[neighbor], self.max_labels_per_node
                    )

                # 優先度キューに追加
                heappush(pq, new_label)

        # 目的地のラベルをパレート最適解として返す
        pareto_solutions = [
            (label.bandwidth, label.delay, label.hops, label.path)
            for label in labels[destination]
        ]

        return pareto_solutions

    def _prune_labels(self, labels: List[Label], max_count: int) -> List[Label]:
        """
        ラベルを剪定（帯域が大きいものを優先的に保持）

        Args:
            labels: ラベルのリスト
            max_count: 保持する最大数

        Returns:
            剪定後のラベルリスト
        """
        # 帯域が大きい順にソート
        sorted_labels = sorted(labels, key=lambda x: x.bandwidth, reverse=True)
        return sorted_labels[:max_count]

    def is_pareto_optimal(
        self, solution: Tuple[float, float, int], pareto_frontier: List[Tuple]
    ) -> bool:
        """
        解がパレート最適解かどうかを判定（完全一致）

        Args:
            solution: 解 (bandwidth, delay, hops)
            pareto_frontier: パレートフロンティア [(bandwidth, delay, hops, path), ...]

        Returns:
            パレート最適解の場合True
        """
        b, d, h = solution
        for pf_b, pf_d, pf_h, _ in pareto_frontier:
            # 完全一致（わずかな誤差を許容）
            if abs(b - pf_b) < 0.01 and abs(d - pf_d) < 0.01 and abs(h - pf_h) < 0.01:
                return True
        return False

    def dominance_check(
        self, solution: Tuple[float, float, int], pareto_frontier: List[Tuple]
    ) -> bool:
        """
        解がパレートフロンティアに支配されないか判定

        Args:
            solution: 解 (bandwidth, delay, hops)
            pareto_frontier: パレートフロンティア

        Returns:
            支配されない場合True
        """
        b, d, h = solution
        solution_label = Label(bandwidth=b, delay=d, hops=h, path=[])

        for pf_b, pf_d, pf_h, _ in pareto_frontier:
            pf_label = Label(bandwidth=pf_b, delay=pf_d, hops=pf_h, path=[])
            if pf_label.dominates(solution_label):
                return False  # 支配されている

        return True  # 支配されていない
