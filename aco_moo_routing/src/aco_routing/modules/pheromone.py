"""
フェロモン更新ロジック

アリがゴールに到達した際のフェロモン更新と功績ボーナスの計算を行います。
"""

import math
from typing import Dict, Optional, Tuple

from ..core.ant import Ant
from ..core.graph import RoutingGraph
from .evaluator import SolutionEvaluator


class PheromoneUpdater:
    """フェロモン更新を管理するクラス"""

    def __init__(self, config: Dict, evaluator: SolutionEvaluator):
        """
        Args:
            config: 設定辞書
            evaluator: 評価関数オブジェクト
        """
        self.config = config
        self.evaluator = evaluator
        self.bonus_factor = config["aco"]["learning"]["bonus_factor"]
        self.delay_tolerance = config["aco"]["learning"]["delay_tolerance"]

    def update_from_ant(self, ant: Ant, graph: RoutingGraph) -> None:
        """
        アリがゴールに到達した際にフェロモンとノードの学習値を更新

        Args:
            ant: ゴールに到達したアリ
            graph: ルーティンググラフ
        """
        # アリの解を取得
        bandwidth, delay, hops = ant.get_solution()

        # Step 1: 各ノードの学習値（BKB/BLD/BKH）を更新し、更新前の値を記録
        node_old_memory: Dict[int, Tuple[float, float, float]] = {}
        for node in ant.route:
            # 更新前の値を記録
            node_old_memory[node] = (
                graph[node].bkb,
                graph[node].bld,
                graph[node].bkh,
            )

            # 学習値を更新
            graph[node].update_all(bandwidth, delay, hops)

        # Step 2: 経路上の各エッジにフェロモンを付加（帰還時の処理）
        route_edges = ant.get_route_edges()

        for i, (u, v) in enumerate(route_edges):
            # 評価関数でスコアを計算
            score = self.evaluator.evaluate(bandwidth, delay, hops)

            # 功績ボーナスの判定（分散判断）
            # 注意: エッジ(u, v)の処理 = 帰還時にノードvからノードuへ戻る処理に対応
            # この時点で、アリはノードvの記憶値のみを知っている
            k_v, l_v, m_v = node_old_memory[v]
            ant_solution = (bandwidth, delay, hops)
            node_memory = (k_v, l_v, m_v)

            # フェロモン付加量 = スコア × 10（既存実装との互換性）
            base_pheromone = score * 10.0

            if self.evaluator.check_bonus_condition(
                ant_solution, node_memory, self.delay_tolerance
            ):
                # ボーナスあり
                delta_pheromone = base_pheromone * self.bonus_factor
            else:
                # ボーナスなし
                delta_pheromone = base_pheromone

            # フェロモンを更新（双方向）
            graph.update_pheromone(u, v, delta_pheromone, bidirectional=True)


class PheromoneEvaporator:
    """フェロモン揮発を管理するクラス"""

    def __init__(self, config: Dict, evaluator: Optional[SolutionEvaluator] = None):
        """
        Args:
            config: 設定辞書
            evaluator: 評価関数オブジェクト（多目的最適化の場合に使用）
        """
        self.config = config
        self.evaporation_rate = config["aco"]["evaporation_rate"]
        self.penalty_factor = config["aco"]["learning"]["penalty_factor"]
        self.volatilization_mode = config["aco"]["learning"]["volatilization_mode"]
        self.evaluator = evaluator
        self.delay_tolerance = config["aco"]["learning"]["delay_tolerance"]

    def evaporate(self, graph: RoutingGraph) -> None:
        """
        フェロモンを揮発

        Args:
            graph: ルーティンググラフ
        """
        if self.volatilization_mode == 3:
            # BKBベースのペナルティ付き揮発
            self._evaporate_with_bkb_penalty(graph)
        else:
            # 通常の揮発
            graph.evaporate_pheromone(self.evaporation_rate)

    def _evaporate_with_bkb_penalty(self, graph: RoutingGraph) -> None:
        """
        BKBベースのペナルティ付き揮発（帯域を主基準に）

        【核心】エッジ単体の帯域がBKBより低ければ、そのエッジを使った経路が
        BKBを超えることは数学的に不可能。遅延は経路全体で挽回可能なため、
        エッジ単体でのペナルティ判定は帯域を主基準とする。

        Args:
            graph: ルーティンググラフ
        """
        for u, v in graph.graph.edges():
            # エッジの属性を取得
            edge_bandwidth = graph.graph.edges[u, v]["bandwidth"]

            # ノードuの学習値（BKB）
            bkb_u = graph[u].bkb

            base_evaporation = self.evaporation_rate  # 0.02（基本揮発率）

            # ペナルティ判定：帯域を主基準に
            should_penalize = False

            # 【主判定】エッジの帯域がBKBより低い場合
            # ボトルネック帯域は物理的な上限（min）なので、
            # エッジ単体の帯域がBKBより低ければ、そのエッジを使った経路が
            # BKBを超えることは数学的に不可能
            if edge_bandwidth < bkb_u:
                should_penalize = True

            # 【補助判定】遅延によるペナルティ（オプション）
            # エッジ単体の遅延が、BLD（Best Known Delay）全体を超えている場合のみ
            # ただし、遅延は経路全体で挽回可能なため、緩和する
            # 現在は無効化（帯域基準のみ）
            # 将来の拡張用:
            # edge_delay = graph.graph.edges[u, v]["delay"]
            # bld_u = graph[u].bld
            # if bld_u != float("inf") and edge_delay > bld_u * 2.0:
            #     should_penalize = True

            if should_penalize:
                # ペナルティあり（揮発を促進）
                # 既存実装: 残存率 0.98 * 0.5 = 0.49（51%消える）
                # → 揮発率に換算: 1 - 0.49 = 0.51（51%消える）
                evaporation = 1.0 - (1.0 - base_evaporation) * self.penalty_factor
            else:
                # ペナルティなし
                evaporation = base_evaporation

            # 【最終的に既存実装と同じ結果】
            # 残存率 = 1 - 揮発率
            retention_rate = 1.0 - evaporation

            # フェロモンを揮発（既存実装と同じ計算式）
            current = graph.graph.edges[u, v]["pheromone"]
            new_pheromone = math.floor(current * retention_rate)
            new_pheromone = max(new_pheromone, graph.graph.edges[u, v]["min_pheromone"])
            graph.graph.edges[u, v]["pheromone"] = new_pheromone
