"""
ACO Solverモジュール

ACOのメインループを実装します。
"""

import random
from typing import Dict, List, Optional, Tuple

from ..core.ant import Ant
from ..core.graph import RoutingGraph
from ..modules.bandwidth_fluctuation import (
    AR1Model,
    BandwidthFluctuationModel,
    select_fluctuating_edges,
)
from ..modules.evaluator import SolutionEvaluator
from ..modules.pheromone import PheromoneEvaporator, PheromoneUpdater
from .single_objective_solver import bottleneck_capacity, max_load_path


class ACOSolver:
    """ACOソルバークラス"""

    def __init__(self, config: Dict, graph: RoutingGraph):
        """
        Args:
            config: 設定辞書
            graph: ルーティンググラフ
        """
        self.config = config
        self.graph = graph

        # 評価関数
        self.evaluator = SolutionEvaluator(config["experiment"]["target_objectives"])

        # フェロモン更新・揮発
        self.pheromone_updater = PheromoneUpdater(config, self.evaluator)
        self.pheromone_evaporator = PheromoneEvaporator(config, self.evaluator)

        # 帯域変動モデル
        self.fluctuation_model: Optional[BandwidthFluctuationModel] = None
        self.edge_states: Dict = {}
        if config["graph"]["fluctuation"]["enabled"]:
            self._initialize_fluctuation_model()

        # ACOパラメータ
        self.alpha = config["aco"]["alpha"]
        self.beta_bandwidth = config["aco"]["beta_bandwidth"]
        self.beta_delay = config["aco"]["beta_delay"]
        self.epsilon = config["aco"]["epsilon"]
        self.ttl = config["aco"]["ttl"]

    def _initialize_fluctuation_model(self) -> None:
        """帯域変動モデルを初期化"""
        model_name = self.config["graph"]["fluctuation"]["model"]

        if model_name == "ar1":
            self.fluctuation_model = AR1Model(self.graph.graph)
        else:
            raise ValueError(f"Unknown fluctuation model: {model_name}")

        # 変動対象エッジを選択
        method = self.config["graph"]["fluctuation"]["target_method"]
        percentage = self.config["graph"]["fluctuation"]["target_percentage"]
        fluctuating_edges = select_fluctuating_edges(
            self.graph.graph, method, percentage
        )

        # エッジ状態を初期化
        self.edge_states = self.fluctuation_model.initialize_states(fluctuating_edges)

    def run(
        self,
        start_node: int,
        goal_node: int,
        generations: int,
        optimal_solutions: Optional[List[Tuple[float, float, int]]] = None,
        metrics_calculator=None,
    ) -> Tuple[List[Dict], List[int]]:
        """
        ACOを実行

        Args:
            start_node: 開始ノード
            goal_node: 目的地ノード
            generations: 世代数
            optimal_solutions: 最適解のリスト [(bandwidth, delay, hops), ...]
                             単一最適解でもパレートフロンティアでも同じ形式
            metrics_calculator: 評価指標計算オブジェクト

        Returns:
            (各世代の結果のリスト, アリごとの成功ログ)
        """
        if optimal_solutions is None:
            optimal_solutions = []
        results = []
        ant_log = []  # 各アリが到達した時の成功/失敗を記録

        # 帯域のみ最適化の場合、最適解を再計算するかどうか
        recalculate_optimal = self.config["experiment"]["target_objectives"] == [
            "bandwidth"
        ]

        # update_intervalを取得
        # update_interval == 0 の場合は変動を無効化（enabled: false と同じ）
        update_interval = self.config["graph"]["fluctuation"].get("update_interval", 1)
        if update_interval == 0:
            update_interval = float("inf")  # 変動しない（無限大に設定）

        # 現在の最適解を保持
        current_optimal_bottleneck = None

        for generation in range(generations):
            # 帯域変動（update_intervalに応じて更新頻度を制御）
            bandwidth_updated = False
            if self.fluctuation_model is not None and update_interval != float("inf"):
                if generation % update_interval == 0:
                    self.fluctuation_model.update(self.edge_states, generation)
                    bandwidth_updated = True

            # ★最適解の再計算★
            # enabled: false または update_interval == 0 の場合：
            #   帯域が変動しないので、最初の世代のみ計算
            # enabled: true の場合：
            #   帯域が変動した時（update_intervalごと）に再計算
            # 最適解かどうかの判定は毎世代行う（155行目以降）
            should_recalculate = recalculate_optimal and (
                generation == 0  # 最初の世代は必ず計算
                or (
                    self.fluctuation_model is not None
                    and update_interval != float("inf")
                    and bandwidth_updated
                )  # 帯域が変動した時のみ再計算
            )
            if should_recalculate:
                try:
                    optimal_path = max_load_path(
                        self.graph.graph, start_node, goal_node, weight="bandwidth"
                    )
                    current_optimal_bottleneck = bottleneck_capacity(
                        self.graph.graph, optimal_path, weight="bandwidth"
                    )
                except Exception:
                    # 経路が存在しない場合はスキップ
                    current_optimal_bottleneck = None

            # アリの生成
            num_ants = self.config["experiment"]["num_ants"]
            ants = [Ant(i, start_node, goal_node, self.ttl) for i in range(num_ants)]

            # アリの探索
            generation_solutions = []
            for ant in ants:
                success = self._ant_search(ant, goal_node)
                if success:
                    solution = ant.get_solution()
                    generation_solutions.append(solution)
                    # フェロモンとノードの学習値を更新
                    self.pheromone_updater.update_from_ant(ant, self.graph)

                    # 各アリが到達した時点で、その世代の最適解と一致するかチェック
                    # 値の意味: -1=最適解ではない, 0以上=最適解のインデックス
                    optimal_index = -1

                    # 帯域のみ最適化の場合：各世代で再計算した最適解と比較
                    if recalculate_optimal and current_optimal_bottleneck is not None:
                        solution_bandwidth = solution[0]
                        if abs(solution_bandwidth - current_optimal_bottleneck) < 0.01:
                            optimal_index = 0  # 帯域のみ最適化では最適解は1つ
                    # 最適解リストと比較（パレート最適化、または帯域のみ最適化のフォールバック）
                    elif optimal_solutions and metrics_calculator:
                        optimal_index = metrics_calculator.find_optimal_solution_index(
                            solution, optimal_solutions
                        )

                    ant_log.append(optimal_index)
                else:
                    # ゴールに到達できなかった場合
                    ant_log.append(-1)

            # フェロモンの揮発
            self.pheromone_evaporator.evaporate(self.graph)

            # ノードの学習値の揮発
            bkb_evaporation_rate = self.config["aco"]["learning"][
                "bkb_evaporation_rate"
            ]
            self.graph.evaporate_node_learning(bkb_evaporation_rate)

            # 結果を記録
            results.append(
                {"generation": generation, "solutions": generation_solutions}
            )

        return results, ant_log

    def _ant_search(self, ant: Ant, goal_node: int) -> bool:
        """
        アリの経路探索

        Args:
            ant: アリ
            goal_node: 目的地ノード

        Returns:
            ゴールに到達した場合True
        """
        while ant.is_alive() and not ant.has_reached_goal():
            # 次のノードを選択
            next_node = self._select_next_node(ant)

            if next_node is None:
                # 行き先がない場合は探索失敗
                return False

            # エッジの属性を取得
            edge_attr = self.graph.get_edge_attributes(ant.current_node, next_node)

            # 移動
            ant.move_to(next_node, edge_attr["bandwidth"], edge_attr["delay"])

        return ant.has_reached_goal()

    def _select_next_node(self, ant: Ant) -> Optional[int]:
        """
        ε-Greedy法で次のノードを選択

        Args:
            ant: アリ

        Returns:
            次のノード（候補がない場合はNone）
        """
        neighbors = self.graph.get_neighbors(ant.current_node)
        candidates = [n for n in neighbors if not ant.has_visited(n)]

        if not candidates:
            return None

        # ε-Greedy選択
        if random.random() < self.epsilon:
            # ランダム選択（探索）
            return random.choice(candidates)
        else:
            # フェロモンとヒューリスティックに基づく選択（活用）
            return self._probabilistic_selection(ant, candidates)

    def _probabilistic_selection(self, ant: Ant, candidates: List[int]) -> int:
        """
        フェロモンとヒューリスティックに基づく確率的選択

        Args:
            ant: アリ
            candidates: 候補ノードのリスト

        Returns:
            選択されたノード
        """
        weights = []
        for candidate in candidates:
            edge_attr = self.graph.get_edge_attributes(ant.current_node, candidate)

            # フェロモン
            pheromone = edge_attr["pheromone"]
            tau = pheromone**self.alpha

            # ヒューリスティック情報（ローカル情報のみ）
            bandwidth = edge_attr["bandwidth"]
            delay = edge_attr["delay"]

            # ヒューリスティック値の計算
            # 目的関数に応じて計算方法を変える
            objectives = self.config["experiment"]["target_objectives"]

            if "delay" in objectives and len(objectives) > 1:
                # 遅延も考慮する場合（Step 2, 3）
                if delay > 0:
                    eta = (bandwidth**self.beta_bandwidth) / (delay**self.beta_delay)
                else:
                    eta = bandwidth**self.beta_bandwidth
            else:
                # 帯域のみ考慮する場合（Step 1: bandwidth vs hops）
                # 既存実装と同じヒューリスティック
                eta = bandwidth**self.beta_bandwidth

            # 重み
            weight = tau * eta
            weights.append(weight)

        # 重みが全て0の場合はランダム選択
        if sum(weights) == 0:
            return random.choice(candidates)

        # 確率的に選択
        return random.choices(candidates, weights=weights, k=1)[0]
