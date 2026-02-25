"""
ACO Solverモジュール

提案手法（Proposed Method）のACOアルゴリズムを実装します。

【アルゴリズム概要】
1. 各世代で複数のアリ（エージェント）を生成し、スタートノードから探索を開始
2. アリはε-Greedy戦略で次のノードを選択（探索と活用のバランス）
3. ゴール群のいずれかに到達したアリは、即座にフェロモンを更新（完全分散方式）
4. 各ノードは通過したアリの解品質を学習（BKB/BLD/BKH）
5. フェロモン揮発時に、ノードの学習値に基づいてペナルティを適用
6. 世代終了時に全エッジのフェロモンを揮発させ、古い情報を忘却

【既存実装との互換性】
このモジュールは、既存実装（aco_main_bkb_available_bandwidth.py）と
同じ結果を得られるように設計されています。
"""

import math
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
from .single_objective_solver import (
    bottleneck_capacity,
    calculate_path_delay,
    find_all_max_bottleneck_paths_with_delay_constraint,
    max_load_path,
    max_load_path_with_delay_constraint,
)

# ===== 既存実装互換モード用のパラメータ =====
# 削除: config.yamlから取得するため、ハードコードは不要


class ACOSolver:
    """
    提案手法（ノードベース学習/BKB/BLD/BKH）のACOソルバー

    - ノード学習: 各ノードがBKB/BLD/BKHを保持し、分散学習を行う
    - フェロモン: ノード学習値に基づく更新・ペナルティ付き揮発を実装
    - 探索戦略: ε-Greedyによる確率的遷移（フェロモン×ヒューリスティック）
    - 帯域変動: AR(1)モデルなどの動的帯域モデルに対応（オプション）

    Attributes:
        config (Dict): 設定辞書
        graph (RoutingGraph): ルーティンググラフ（NetworkXラッパー）
        evaluator (SolutionEvaluator): 目的関数に基づく評価器
        pheromone_updater (PheromoneUpdater): フェロモン付加ロジック
        pheromone_evaporator (PheromoneEvaporator): フェロモン揮発ロジック
        fluctuation_model (Optional[BandwidthFluctuationModel]): 帯域変動モデル（未使用ならNone）
        alpha (float): フェロモンの重み
        beta_bandwidth (float): 帯域ヒューリスティックの重み
        beta_delay (float): 遅延ヒューリスティックの重み
        epsilon (float): ε-Greedyの確率
        ttl (int): アリの最大ステップ数
        delay_constraint_enabled (bool): 遅延制約の有無
        max_delay (float): 遅延制約の上限値
    """

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

        # 遅延制約
        delay_constraint = config["experiment"].get("delay_constraint", {})
        self.delay_constraint_enabled = delay_constraint.get("enabled", False)
        self.max_delay = delay_constraint.get("max_delay", float("inf"))

        # 既存実装互換モード用のパラメータ
        self.penalty_factor = config["aco"]["learning"]["penalty_factor"]
        # 【重要】既存実装との互換性: BKB揮発率は「残存率」として使用される
        # 既存実装: BKB_EVAPORATION_RATE = 0.999 で bkb *= 0.999 (99.9%残存)
        # config.yamlのbkb_evaporation_rateは「揮発率」として定義されているため、
        # 残存率に変換する必要がある: 残存率 = 1 - 揮発率
        # ただし、既存実装との完全互換のため、0.999を使用
        self.bkb_evaporation_rate = 0.999  # 残存率（既存実装と同じ）
        self.time_window_size = config["aco"]["learning"]["bkb_window_size"]

        # フェロモン揮発の残存率（既存実装: V = 0.98）
        # config.yamlのevaporation_rateは「揮発率」として定義されているため、
        # 残存率に変換: 残存率 = 1 - 揮発率
        evaporation_rate = config["aco"]["evaporation_rate"]
        self.pheromone_retention_rate = 1.0 - evaporation_rate  # 0.98

        # 功績ボーナス係数（既存実装: ACHIEVEMENT_BONUS = 2.0）
        self.bonus_factor = config["aco"]["learning"]["bonus_factor"]

        # グラフのノードにBKB属性を初期化（既存実装互換）
        self._initialize_node_bkb()

    def _initialize_node_bkb(self) -> None:
        """
        グラフのノードにBKB関連属性を初期化（既存実装互換）

        既存実装（aco_main_bkb_available_bandwidth.py）では、
        グラフのノード属性としてBKBを管理しています。
        """
        G = self.graph.graph
        for node in G.nodes():
            G.nodes[node]["best_known_bottleneck"] = 0
            G.nodes[node]["time_window_values"] = []

    def _update_pheromone_compatible(
        self, route: List[int], widths: List[float], total_delay: float = 0.0
    ) -> None:
        """
        既存実装互換のフェロモン更新（遅延制約拡張版）

        既存実装（aco_main_bkb_available_bandwidth.py）と同じロジックで
        フェロモンとBKBを更新します。

        【遅延制約拡張】
        遅延制約が有効な場合、フェロモン付加量を遅延で調整：
        - 基本スコア: Δτ = C × B/D_path （C=10）
        - 遅延が小さいほど、より多くのフェロモンを付加
        - 同じ帯域でも遅延が小さい経路を優遇

        Args:
            route: アリの経路 [start, ..., goal]
            widths: 各エッジの帯域幅リスト
            total_delay: 経路の総遅延（ms）。遅延制約が有効な場合に使用
        """
        G = self.graph.graph
        if not widths:
            return

        bottleneck = min(widths)
        bottleneck_int = int(bottleneck)

        # BKB更新前の値を記録（功績ボーナス判定用）
        node_old_bkb: Dict[int, float] = {}
        for node in route:
            old_bkb = G.nodes[node].get("best_known_bottleneck", 0)
            node_old_bkb[node] = old_bkb

            # リングバッファに追加（既存実装と同じ）
            if "time_window_values" not in G.nodes[node]:
                G.nodes[node]["time_window_values"] = []
            window_values = G.nodes[node]["time_window_values"]
            window_values.append(float(bottleneck_int))

            # サイズを超えたら古いものを削除
            while len(window_values) > self.time_window_size:
                window_values.pop(0)

            # バッファ内の最大値をBKBとして使用
            time_window_max = max(window_values) if window_values else 0
            G.nodes[node]["best_known_bottleneck"] = int(time_window_max)

        # フェロモン付加（遅延制約拡張版、有向グラフ対応）
        # 【論文の式\eqref{eq:deposit_delay}】
        # Δτ = C × B²/D_path （遅延制約が有効な場合、帯域の2乗で差別化）
        # Δτ = C × B² （遅延制約が無効な場合、帯域の2乗）
        # 【重要】有向グラフ対応: 無向グラフでは双方向付加で2倍蓄積されていたため、
        #         有向グラフでは基本量を2倍にして互換性を維持
        #         さらに、B²にすることでより強い差別化を実現
        bottleneck_squared = float(bottleneck_int) ** 2

        if self.delay_constraint_enabled and total_delay > 0:
            # 遅延を考慮したスコア: B²/D_path
            base_pheromone = bottleneck_squared / total_delay
        else:
            # 帯域の2乗: B²
            base_pheromone = bottleneck_squared

        for i in range(1, len(route)):
            u, v = route[i - 1], route[i]

            # 功績ボーナス判定（既存実装と同じ）
            k_v = node_old_bkb.get(v, 0)
            if bottleneck_int >= k_v:
                pheromone_increase = base_pheromone * self.bonus_factor
            else:
                pheromone_increase = base_pheromone

            # 双方向にフェロモン付加
            max_ph_uv = G.edges[u, v].get("max_pheromone", 10**9)
            G.edges[u, v]["pheromone"] = min(
                G.edges[u, v]["pheromone"] + pheromone_increase, max_ph_uv
            )
            max_ph_vu = G.edges[v, u].get("max_pheromone", 10**9)
            G.edges[v, u]["pheromone"] = min(
                G.edges[v, u]["pheromone"] + pheromone_increase, max_ph_vu
            )

    def _evaporate_pheromone_compatible(self) -> None:
        """
        既存実装互換のフェロモン揮発

        既存実装（aco_main_bkb_available_bandwidth.py）と同じロジックで
        フェロモンを揮発します。

        【重要】有向グラフ対応:
        G.edges()には(u, v)と(v, u)の両方が含まれるため、
        各エッジは1回だけ処理する（双方向処理は不要）
        """
        G = self.graph.graph

        for u, v in G.edges():
            # u -> v の揮発
            weight_uv = G.edges[u, v]["bandwidth"]
            bkb_u = G.nodes[u].get("best_known_bottleneck", 0)

            # 【既存実装との互換性】整数比較（既存実装ではint比較）
            if int(weight_uv) < int(bkb_u):
                rate = self.pheromone_retention_rate * self.penalty_factor
            else:
                rate = self.pheromone_retention_rate

            min_ph = G.edges[u, v].get("min_pheromone", 100)
            # 【既存実装との互換性】math.floorを使用（既存実装と同じ）
            current_pheromone = G.edges[u, v]["pheromone"]
            new_pheromone = max(math.floor(current_pheromone * rate), min_ph)
            G.edges[u, v]["pheromone"] = new_pheromone

    def _evaporate_bkb_compatible(self) -> None:
        """
        既存実装互換のBKB揮発

        既存実装（aco_main_bkb_available_bandwidth.py）と同じロジックで
        BKBを揮発します。

        【重要】既存実装との互換性:
        - 既存実装: evaporate_bkb_values(graph, 0.999, use_int_cast=False)
        - bkb *= 0.999 で99.9%が残存する
        """
        G = self.graph.graph
        for node in G.nodes():
            old_bkb = G.nodes[node].get("best_known_bottleneck", 0)
            # 既存実装: bkb *= 0.999（残存率として使用）
            G.nodes[node]["best_known_bottleneck"] = old_bkb * self.bkb_evaporation_rate

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
    ) -> Tuple[List[Dict], Tuple[List[int], List[int]]]:
        """
        ACOを実行

        Args:
            start_node: 開始ノード（スタート切り替えが有効な場合は最初のスタートノード）
            goal_node: 目的地ノード（スタート切り替えが有効な場合は最初のゴールノード）
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
        ant_log_unique_optimal = (
            []
        )  # 一意な最適解（遅延最小）に一致したか（1=最適解、-1=ゴール未到達、-2=非最適解）
        ant_log_any_optimal = (
            []
        )  # 最適解リストのいずれかに一致したか（1=最適解、-1=ゴール未到達、-2=非最適解）

        # スタートノード切り替えの設定を取得
        start_switching_config = self.config["experiment"].get("start_switching", {})
        start_switching_enabled = start_switching_config.get("enabled", False)
        switch_interval = start_switching_config.get("switch_interval", 200)

        # スタートノードリストを取得または生成
        # 【動的環境シミュレーション】ICN（Information-Centric Networking）を模擬
        # コンテンツ要求ノード（スタート）が定期的に切り替わり、
        # 以前の要求ノードがキャッシュ（ゴール）として追加されていく
        if start_switching_enabled:
            start_nodes = start_switching_config.get("start_nodes", [])
            if not start_nodes:
                # 自動生成: グラフのノード数に基づいてランダムに選択
                # 初期ゴールノードを除外してから選択（スタートとゴールが同じだと探索が無意味）
                num_nodes = self.graph.num_nodes
                num_start_nodes = 10
                all_nodes = list(range(num_nodes))
                start_node_candidates = [n for n in all_nodes if n != goal_node]
                start_nodes = random.sample(
                    start_node_candidates,
                    min(num_start_nodes, len(start_node_candidates)),
                )
        else:
            start_nodes = [start_node]

        # ゴールノードリスト（以前のスタートノードが追加されていく）
        # 【Anycast方式】複数のゴールのいずれかに到達すれば成功
        # 最初から初期ゴールノードを含める（最初のフェーズから探索可能にする）
        goal_nodes = [goal_node]

        # 現在のスタートノード（スタート切り替えが無効な場合は固定）
        current_start = start_nodes[0]

        # 帯域のみ最適化の場合、最適解を再計算するかどうか
        # 遅延制約が有効な場合、target_objectivesは["bandwidth", "delay"]に変更されているが、
        # フェロモン付加・揮発・ノード学習は["bandwidth", "delay"]と同じ（bandwidth/delayスコア）
        # 最適解判定は単一最適解として扱う（パレート最適化ではない）
        recalculate_optimal = (
            self.config["experiment"]["target_objectives"] == ["bandwidth"]
            or self.delay_constraint_enabled
        )

        # update_intervalを取得
        # update_interval == 0 の場合は変動を無効化（enabled: false と同じ）
        update_interval = self.config["graph"]["fluctuation"].get("update_interval", 1)
        if update_interval == 0:
            update_interval = float("inf")  # 変動しない（無限大に設定）

        # 現在の最適解を保持（スタートノードごとに最適解をキャッシュ）
        # 【最適解の定義】
        # - 遅延制約なし：最大ボトルネック帯域を持つ経路
        # - 遅延制約あり：帯域が最大で、その中で遅延が最小の経路（Lexicographical Order）
        optimal_bottleneck_dict: Dict[int, float] = (
            {}
        )  # 遅延制約なし：帯域値、遅延制約あり：帯域値
        optimal_delay_dict: Dict[int, Optional[float]] = (
            {}
        )  # 遅延制約ありの場合：一意な最適解（タイブレーク）の遅延値
        optimal_solutions_dict: Dict[int, List[Tuple[float, float, int]]] = (
            {}
        )  # 遅延制約ありの場合：複数の最適解のリスト

        # 前のスタートノードを記録（ゴール追加用）
        previous_start = None

        # 【修正】ループ外で最適解関連変数を初期化
        # generation > 0 でも generation == 0 で設定した値を保持するため
        current_optimal_bottleneck: Optional[float] = None
        current_optimal_delay: Optional[float] = None
        current_optimal_solutions: List[Tuple[float, float, int]] = []

        # 【デバッグ】generation 0の開始時に最適解を確認
        if optimal_solutions and len(optimal_solutions) > 0:
            print(f"[DEBUG] Generation 0開始前: optimal_solutions={optimal_solutions}")
            print(
                f"[DEBUG] Generation 0開始前: current_optimal_bottleneck={current_optimal_bottleneck}"
            )

        for generation in range(generations):
            # スタートノード切り替え処理
            if start_switching_enabled:
                phase = generation // switch_interval
                if phase >= len(start_nodes):
                    break

                # スタートノードが切り替わった時（generation % switch_interval == 0）のみ処理
                if generation % switch_interval == 0:
                    # 【キャッシュ追加】前のフェーズのスタートノードを新しいゴールとして追加
                    # ICNでは、コンテンツを要求したノードがキャッシュを持つようになる
                    if previous_start is not None:
                        if previous_start not in goal_nodes:
                            goal_nodes.append(previous_start)
                            print(
                                f"世代 {generation}: ノード {previous_start} をゴール群に追加"
                            )

                    current_start = start_nodes[phase]

                    # スタートノードが既にゴールにある場合はスキップ
                    # （既にキャッシュがあるノードから要求するケース）
                    if current_start in goal_nodes:
                        print(
                            f"警告: スタートノード {current_start} は既にゴールです。このフェーズをスキップします。"
                        )
                        optimal_bottleneck_dict[current_start] = -1.0
                        previous_start = current_start
                        continue

                    previous_start = current_start

                    print(
                        f"\n--- 世代 {generation}: スタート {current_start}, ゴール群 {goal_nodes} ---"
                    )

                    # 【最適解の再計算】複数ゴールがある場合、全ゴールへの経路を探索し、
                    # 最適解を定義（最大ボトルネック帯域を持つ全ての経路）
                    if recalculate_optimal:
                        if self.delay_constraint_enabled:
                            # 遅延制約が有効な場合：全ゴールの中で最大ボトルネック帯域を持つ全ての経路を探す
                            # 【最適解の定義】タイブレークで一意な最適解を定義：
                            # 1. 遅延制約を満たす（D ≤ D_limit）
                            # 2. ボトルネック帯域が最大（Maximize B）
                            # 3. 帯域が同じ場合、遅延が最小（Minimize D）
                            best_bottleneck_for_phase = 0.0
                            best_delay_for_phase = float("inf")
                            all_optimal_solutions_for_phase: List[
                                Tuple[float, float, int]
                            ] = []

                            # まず、全ゴールの中で最大のボトルネック帯域を求める
                            for g in goal_nodes:
                                if current_start == g:
                                    continue
                                try:
                                    optimal_path = max_load_path_with_delay_constraint(
                                        self.graph.graph,
                                        current_start,
                                        g,
                                        self.max_delay,
                                        bandwidth_weight="bandwidth",
                                        delay_weight="delay",
                                    )
                                    bottleneck = bottleneck_capacity(
                                        self.graph.graph,
                                        optimal_path,
                                        weight="bandwidth",
                                    )
                                    delay = calculate_path_delay(
                                        self.graph.graph, optimal_path
                                    )
                                    # 最大ボトルネック帯域を更新
                                    if bottleneck > best_bottleneck_for_phase:
                                        best_bottleneck_for_phase = bottleneck
                                        best_delay_for_phase = delay
                                    elif (
                                        abs(bottleneck - best_bottleneck_for_phase)
                                        < 1e-6
                                        and delay < best_delay_for_phase
                                    ):
                                        # 同じ帯域なら遅延が最小のものを更新
                                        best_delay_for_phase = delay
                                except Exception:
                                    # 経路が存在しない場合はスキップ
                                    continue

                            # 最大ボトルネック帯域を持つ全ての経路を全ゴールから探す
                            # （評価用：同じボトルネック帯域を持つ全ての経路を最適解として扱う）
                            for g in goal_nodes:
                                if current_start == g:
                                    continue
                                try:
                                    # 最大ボトルネック帯域を持つ全ての経路を取得
                                    find_func = find_all_max_bottleneck_paths_with_delay_constraint
                                    all_optimal_paths = find_func(
                                        self.graph.graph,
                                        current_start,
                                        g,
                                        self.max_delay,
                                        bandwidth_weight="bandwidth",
                                        delay_weight="delay",
                                    )
                                    for path in all_optimal_paths:
                                        bottleneck = bottleneck_capacity(
                                            self.graph.graph, path, weight="bandwidth"
                                        )
                                        delay = calculate_path_delay(
                                            self.graph.graph, path
                                        )
                                        # 最大ボトルネック帯域を持つ経路のみを追加
                                        bottleneck_diff = abs(
                                            bottleneck - best_bottleneck_for_phase
                                        )
                                        if bottleneck_diff < 1e-6:
                                            hops = len(path) - 1
                                            solution_tuple = (bottleneck, delay, hops)
                                            # 重複を避ける（同じ経路が複数ゴールで見つかる可能性があるため）
                                            if (
                                                solution_tuple
                                                not in all_optimal_solutions_for_phase
                                            ):
                                                all_optimal_solutions_for_phase.append(
                                                    solution_tuple
                                                )
                                except Exception:
                                    # 経路が存在しない場合はスキップ
                                    continue

                            optimal_bottleneck_dict[current_start] = (
                                best_bottleneck_for_phase
                            )
                            # 一意な最適解の遅延計算
                            if all_optimal_solutions_for_phase:
                                sols = all_optimal_solutions_for_phase
                                min_delay_in_solutions = min(d for _, d, _ in sols)
                                optimal_delay_dict[current_start] = (
                                    min_delay_in_solutions
                                )
                            else:
                                optimal_delay_dict[current_start] = (
                                    best_delay_for_phase
                                    if best_delay_for_phase != float("inf")
                                    else None
                                )
                            optimal_solutions_dict[current_start] = (
                                all_optimal_solutions_for_phase
                            )
                            print(
                                f"現在の最適解 (delay ≤ {self.max_delay}ms): "
                                f"Bandwidth={best_bottleneck_for_phase} Mbps, "
                                f"Delay={best_delay_for_phase:.1f} ms, "
                                f"Found {len(all_optimal_solutions_for_phase)} "
                                f"optimal path(s) with same bandwidth"
                            )
                        else:
                            # 遅延制約なし：最大ボトルネック帯域を持つ経路を最適解とする
                            best_bottleneck_for_phase = 0.0
                            for g in goal_nodes:
                                if current_start == g:
                                    continue
                                try:
                                    optimal_path = max_load_path(
                                        self.graph.graph,
                                        current_start,
                                        g,
                                        weight="bandwidth",
                                    )
                                    bottleneck = bottleneck_capacity(
                                        self.graph.graph,
                                        optimal_path,
                                        weight="bandwidth",
                                    )
                                    if bottleneck > best_bottleneck_for_phase:
                                        best_bottleneck_for_phase = bottleneck
                                except Exception:
                                    # 経路が存在しない場合はスキップ
                                    continue
                            optimal_bottleneck_dict[current_start] = (
                                best_bottleneck_for_phase
                            )
                            print(
                                f"現在の最適ボトルネック: {best_bottleneck_for_phase}"
                            )

                # スタートノードが切り替わっていない場合は、現在のフェーズのスタートノードを使用
                current_start = start_nodes[phase]
            else:
                current_start = start_node

            # 現在のゴールノード（複数ゴールがある場合は最大ボトルネックのものを使用）
            if not start_switching_enabled:
                current_goal = goal_node
            else:
                current_goal = goal_nodes[-1] if goal_nodes else goal_node

            # 帯域変動（update_intervalに応じて更新頻度を制御）
            bandwidth_updated = False
            if self.fluctuation_model is not None and update_interval != float("inf"):
                if generation % update_interval == 0:
                    self.fluctuation_model.update(self.edge_states, generation)
                    bandwidth_updated = True

            # 【最適解の取得】ログ記録用に、現在のスタートノードに対する最適解を取得
            # スタートノード切り替え時は事前計算済みの値を、帯域変動時は再計算した値を使用
            # 【注意】current_optimal_bottleneck, current_optimal_delay, current_optimal_solutions
            # はループ外で初期化されており、generation > 0 でも前回の値を保持する
            if start_switching_enabled:
                current_optimal_bottleneck = optimal_bottleneck_dict.get(
                    current_start, -1
                )
                current_optimal_delay = (
                    optimal_delay_dict.get(current_start, float("inf"))
                    if self.delay_constraint_enabled
                    else None
                )
                current_optimal_solutions = optimal_solutions_dict.get(
                    current_start, []
                )  # 複数の最適解のリスト
                if current_optimal_bottleneck <= 0:
                    continue
            else:
                # スタート切り替えが無効な場合のみ、帯域変動時に再計算
                # 【修正】generation == 0の時、optimal_solutionsが渡されている場合はそれを使用
                # 環境1（manual）では、既に計算された最適解（100Mbps）を使用する
                should_recalculate = recalculate_optimal and (
                    (
                        generation == 0 and optimal_solutions is None
                    )  # 最初の世代で最適解が渡されていない場合のみ計算
                    or (
                        self.fluctuation_model is not None
                        and update_interval != float("inf")
                        and bandwidth_updated
                    )  # 帯域が変動した時のみ再計算
                )
                if should_recalculate:
                    try:
                        # 単一ゴールの場合
                        if self.delay_constraint_enabled:
                            # 遅延制約が有効な場合：最大ボトルネック帯域を持つ全ての経路を探す
                            all_optimal_paths = (
                                find_all_max_bottleneck_paths_with_delay_constraint(
                                    self.graph.graph,
                                    current_start,
                                    current_goal,
                                    self.max_delay,
                                    bandwidth_weight="bandwidth",
                                    delay_weight="delay",
                                )
                            )
                            if all_optimal_paths:
                                # 最初の経路からボトルネック帯域を取得
                                current_optimal_bottleneck = bottleneck_capacity(
                                    self.graph.graph,
                                    all_optimal_paths[0],
                                    weight="bandwidth",
                                )
                                # 全ての最適解をリストに追加
                                current_optimal_solutions = []
                                min_delay = float("inf")
                                for path in all_optimal_paths:
                                    bottleneck = bottleneck_capacity(
                                        self.graph.graph, path, weight="bandwidth"
                                    )
                                    delay = calculate_path_delay(self.graph.graph, path)
                                    hops = len(path) - 1
                                    current_optimal_solutions.append(
                                        (bottleneck, delay, hops)
                                    )
                                    # 一意な最適解（遅延最小）を探す
                                    if delay < min_delay:
                                        min_delay = delay
                                current_optimal_delay = (
                                    min_delay if min_delay != float("inf") else None
                                )
                            else:
                                current_optimal_bottleneck = None
                                current_optimal_delay = None
                                current_optimal_solutions = []
                        else:
                            # 遅延制約なし：最大ボトルネック帯域を持つ経路を最適解とする
                            optimal_path = max_load_path(
                                self.graph.graph,
                                current_start,
                                current_goal,
                                weight="bandwidth",
                            )
                            current_optimal_bottleneck = bottleneck_capacity(
                                self.graph.graph, optimal_path, weight="bandwidth"
                            )
                            current_optimal_delay = calculate_path_delay(
                                self.graph.graph, optimal_path
                            )
                            current_optimal_hops = len(optimal_path) - 1
                            # 遅延制約なしの場合も最適解として記録（ログ用）
                            current_optimal_solutions = [
                                (
                                    current_optimal_bottleneck,
                                    current_optimal_delay,
                                    current_optimal_hops,
                                )
                            ]
                    except Exception:
                        # 経路が存在しない場合はスキップ
                        current_optimal_bottleneck = None
                        current_optimal_delay = None
                        current_optimal_solutions = []
                else:
                    # 再計算しない場合は、渡された最適解を使用するか、前回の値を保持
                    # 【重要】generation > 0 でも generation == 0 で設定した値を保持する
                    if generation == 0:
                        # 【修正】generation == 0の時、optimal_solutionsが渡されている場合はそれを使用
                        if optimal_solutions and len(optimal_solutions) > 0:
                            # 渡された最適解を使用（環境1など、既に計算済みの場合）
                            if self.delay_constraint_enabled:
                                # 遅延制約が有効な場合：複数の最適解から最小遅延を取得
                                current_optimal_solutions = optimal_solutions
                                if current_optimal_solutions:
                                    # 最初の最適解からボトルネック帯域を取得
                                    current_optimal_bottleneck = (
                                        current_optimal_solutions[0][0]
                                    )
                                    # 最小遅延を取得
                                    min_delay = min(
                                        opt_delay
                                        for opt_bw, opt_delay, opt_hops in current_optimal_solutions
                                    )
                                    current_optimal_delay = min_delay
                                else:
                                    current_optimal_bottleneck = None
                                    current_optimal_delay = None
                            else:
                                # 遅延制約なし：最初の最適解を使用
                                if len(optimal_solutions) > 0:
                                    current_optimal_bottleneck = optimal_solutions[0][0]
                                    current_optimal_delay = (
                                        optimal_solutions[0][1]
                                        if len(optimal_solutions[0]) > 1
                                        else None
                                    )
                                    current_optimal_solutions = optimal_solutions
                                    # 【デバッグ】generation 0で最適解を設定したことを確認
                                    if generation == 0:
                                        print(
                                            f"[DEBUG] Generation 0: current_optimal_bottleneck={current_optimal_bottleneck:.1f} "
                                            f"を設定しました"
                                        )
                                else:
                                    current_optimal_bottleneck = None
                                    current_optimal_delay = None
                                    current_optimal_solutions = []
                        else:
                            # 最適解が渡されていない場合はNone
                            current_optimal_bottleneck = None
                            current_optimal_delay = None
                            current_optimal_solutions = []
                    # 【重要】generation > 0 では、前回の値を保持（何もしない）
                    # current_optimal_bottleneck, current_optimal_delay, current_optimal_solutions
                    # はループ外で初期化されており、generation == 0 で設定した値が保持される

            # 【デバッグ】各世代の開始時にcurrent_optimal_bottleneckを確認
            if generation < 3:
                print(
                    f"[DEBUG] Generation {generation}開始: "
                    f"current_optimal_bottleneck={current_optimal_bottleneck}"
                )

            # 【アリの生成】各世代で指定数のアリを生成
            # 【Anycast方式】特定のゴールを選ばず、ゴール群のいずれかに到達すれば成功
            # これにより、近くのキャッシュを即座に発見できる（ICNの特性を反映）
            num_ants = self.config["experiment"]["num_ants"]
            dummy_goal = goal_nodes[0] if goal_nodes else goal_node
            ants = [
                Ant(i, current_start, dummy_goal, self.ttl) for i in range(num_ants)
            ]

            # ゴール判定用のセット（高速検索のため）
            goal_nodes_set = set(goal_nodes)

            # 【アリの探索ループ】完全分散方式：アリがゴールに到達した時点で即座にフェロモンを更新
            # これにより、他のアリの探索に即座に影響を与え、学習が加速する
            generation_solutions = []
            active_ants = list(ants)  # アクティブなアリのリスト

            while active_ants:
                # 【既存実装との互換性】アリを逆順で処理
                # 既存実装（aco_main_bkb_available_bandwidth.py line 102）:
                #   for ant in reversed(ant_list):
                for ant in reversed(list(active_ants)):
                    # 【既存実装との互換性】の順序：
                    # 1. 候補がない場合は失敗
                    # 2. 次のノードを選択して移動
                    # 3. ゴール判定
                    # 4. ゴール未到達の場合のみ、TTLチェック

                    # 【ゴール到達判定】Anycast方式：ゴール群のいずれかに到達したかチェック
                    # まず現在位置がゴールかどうかを確認（スタート=ゴールの場合を考慮）
                    has_reached_any_goal = ant.current_node in goal_nodes_set

                    # 【探索継続】ゴール未到達の場合、次のノードを選択して移動
                    if not has_reached_any_goal:
                        next_node = self._select_next_node(ant)
                        if next_node is None:
                            # 行き先がない場合（全ての隣接ノードを訪問済み）は探索失敗
                            active_ants.remove(ant)
                            # 【ログ記録】ゴール未到達 = -1
                            ant_log_unique_optimal.append(-1)
                            ant_log_any_optimal.append(-1)
                            continue

                        # エッジの属性を取得
                        edge_attr = self.graph.get_edge_attributes(
                            ant.current_node, next_node
                        )

                        # 【遅延制約チェック】_select_next_nodeで既にチェック済みだが、
                        # 念のため移動前にも確認（制約を満たさない場合は探索失敗）
                        if self.delay_constraint_enabled:
                            estimated_delay = ant.total_delay + edge_attr["delay"]
                            if estimated_delay > self.max_delay:
                                # 制約を満たさない場合は探索失敗
                                active_ants.remove(ant)
                                # 【ログ記録】制約違反 = -1（ゴール未到達）
                                ant_log_unique_optimal.append(-1)
                                ant_log_any_optimal.append(-1)
                                continue

                        # 移動（ボトルネック帯域、累積遅延、ホップ数を更新）
                        ant.move_to(
                            next_node, edge_attr["bandwidth"], edge_attr["delay"]
                        )

                        # 移動後にゴール判定（移動先がゴールだった場合）
                        has_reached_any_goal = ant.current_node in goal_nodes_set

                        # 【TTLチェック】ゴール未到達の場合のみ、TTLをチェック
                        # 既存実装: elif len(ant.route) >= TTL: ant_log.append(0)
                        # 新実装: len(route) >= TTL は (ttl - remaining_ttl + 1) >= TTL と同等
                        # routeは[start]から始まり、移動ごとに1増える
                        # remaining_ttlはttlから始まり、移動ごとに1減る
                        # len(route) = ttl - remaining_ttl + 1
                        # len(route) >= TTL ⟺ ttl - remaining_ttl + 1 >= TTL
                        #                    ⟺ remaining_ttl <= ttl - TTL + 1
                        # TTL = ttl の場合: remaining_ttl <= 1
                        # 既存実装では TTL=100, route=[start]で始まり、99回移動で len(route)=100 でTTL
                        # 新実装では ttl=100, remaining_ttl=100で始まり、移動でremaining_ttl-=1
                        # 99回移動後: remaining_ttl=1, len(route)=100
                        # つまり、remaining_ttl <= 1 でTTLに達したとみなす
                        if not has_reached_any_goal and ant.remaining_ttl <= 1:
                            active_ants.remove(ant)
                            # 【ログ記録】ゴール未到達 = -1
                            ant_log_unique_optimal.append(-1)
                            ant_log_any_optimal.append(-1)
                            continue

                    # 【ゴール到達時の処理】即座にフェロモンを更新（完全分散方式）
                    if has_reached_any_goal:
                        solution = ant.get_solution()
                        solution_delay = solution[1]  # total_delay

                        # 【遅延制約チェック】ゴール到達時に制約を確認
                        # 制約違反の場合はフェロモンを付加しない（探索失敗として扱う）
                        if self.delay_constraint_enabled:
                            if solution_delay > self.max_delay:
                                # 制約違反の場合は探索失敗として扱う
                                active_ants.remove(ant)
                                # 【ログ記録】制約違反 = -1（ゴール未到達）
                                ant_log_unique_optimal.append(-1)
                                ant_log_any_optimal.append(-1)
                                continue

                        generation_solutions.append(solution)

                        # 【既存実装互換】フェロモン更新（BKB更新も含む）
                        # 既存実装と同じロジックを使用
                        # 【デバッグ】100Mbpsのパスを通ったアリを確認
                        if generation < 10:  # 最初の10世代をデバッグ
                            route_bottleneck = (
                                min(ant.bandwidth_log) if ant.bandwidth_log else 0
                            )
                            if route_bottleneck >= 100.0 and ant.ant_id == 0:
                                # 100Mbpsの経路を見つけた場合
                                first_edge_u, first_edge_v = ant.route[0], ant.route[1]
                                pheromone_before = self.graph.graph.edges[
                                    first_edge_u, first_edge_v
                                ]["pheromone"]
                                print(
                                    f"[DEBUG] Gen {generation}, Ant {ant.ant_id}: "
                                    f"100Mbpsパスを発見！ route={ant.route[:5]}..., "
                                    f"bottleneck={route_bottleneck:.1f}, "
                                    f"最初のエッジ({first_edge_u}→{first_edge_v})のフェロモン(更新前)={pheromone_before:.0f}"
                                )
                        self._update_pheromone_compatible(
                            ant.route, ant.bandwidth_log, ant.total_delay
                        )

                        # 【ログ記録】最適解判定結果を記録
                        # 【既存実装との互換性】1 = 最適解、0 = 失敗
                        if recalculate_optimal:
                            # 【最適解判定】帯域のみ最適化または遅延制約が有効な場合
                            if current_optimal_bottleneck is not None:
                                # 【デバッグ】最初の数世代で最適解判定を確認
                                if generation < 3 and ant.ant_id == 0:
                                    print(
                                        f"[DEBUG] Gen {generation}, Ant {ant.ant_id}: "
                                        f"solution_bandwidth={solution[0]:.1f}, "
                                        f"current_optimal_bottleneck={current_optimal_bottleneck:.1f}, "
                                        f"bandwidth_ok={solution[0] >= current_optimal_bottleneck}"
                                    )
                                if self.delay_constraint_enabled:
                                    # 遅延制約が有効な場合：2つのログを記録
                                    solution_bandwidth = solution[0]
                                    solution_delay = solution[1]

                                    # 【ログ2】最適解リストのいずれかに一致したか
                                    is_any_optimal = False
                                    matched_optimal_delay = None
                                    if current_optimal_solutions:
                                        for (
                                            opt_bw,
                                            opt_delay,
                                            opt_hops,
                                        ) in current_optimal_solutions:
                                            if (
                                                abs(solution_bandwidth - opt_bw) < 1e-6
                                                and abs(solution_delay - opt_delay)
                                                < 1e-6
                                            ):
                                                is_any_optimal = True
                                                matched_optimal_delay = opt_delay
                                                break
                                    # 【ログ記録】1 = 最適解、-2 = 非最適解
                                    log_value_any = 1 if is_any_optimal else -2

                                    # 【ログ1】一意な最適解（遅延最小）に一致したか
                                    # 最適解リストの中で遅延が最小のものと一致する場合に成功
                                    is_unique_optimal = False
                                    if (
                                        is_any_optimal
                                        and matched_optimal_delay is not None
                                    ):
                                        # 最適解リストのいずれかに一致している場合
                                        # 最小遅延を計算
                                        if current_optimal_solutions:
                                            sols = current_optimal_solutions
                                            min_delay_in_solutions = min(
                                                d for _, d, _ in sols
                                            )
                                            # matched_optimal_delayが最小遅延と一致する場合
                                            if (
                                                abs(
                                                    matched_optimal_delay
                                                    - min_delay_in_solutions
                                                )
                                                < 1e-6
                                            ):
                                                is_unique_optimal = True
                                    # 【ログ記録】1 = 最適解、-2 = 非最適解
                                    log_value_unique = 1 if is_unique_optimal else -2
                                else:
                                    # 遅延制約なし：帯域で判定（両方のログは同じ値）
                                    solution_bandwidth = solution[0]
                                    bandwidth_ok = (
                                        solution_bandwidth >= current_optimal_bottleneck
                                    )
                                    # 【デバッグ】最初の数世代で最適解判定を確認
                                    if generation < 3 and ant.ant_id == 0:
                                        print(
                                            f"[DEBUG] Gen {generation}, Ant {ant.ant_id}: "
                                            f"solution_bandwidth={solution_bandwidth:.1f}, "
                                            f"current_optimal_bottleneck={current_optimal_bottleneck:.1f}, "
                                            f"bandwidth_ok={bandwidth_ok}, log_value={1 if bandwidth_ok else 0}"
                                        )
                                    # 【ログ記録】1 = 最適解、-2 = 非最適解
                                    log_value_unique = 1 if bandwidth_ok else -2
                                    log_value_any = log_value_unique
                            else:
                                # 最適解が計算されていない場合（初期世代など）
                                # 【ログ記録】-2 = 非最適解（最適解が計算されていない）
                                log_value_unique = -2
                                log_value_any = -2
                        elif optimal_solutions and metrics_calculator:
                            # 【多目的最適化】パレート最適化の場合
                            # 見つけた解がパレートフロンティアのどの解に対応するかを判定
                            optimal_index = (
                                metrics_calculator.find_optimal_solution_index(
                                    solution, optimal_solutions
                                )
                            )
                            # 【ログ記録】1 = 最適解、-2 = 非最適解
                            log_value_unique = 1 if optimal_index is not None else -2
                            log_value_any = log_value_unique  # パレート最適化では同じ
                        else:
                            # フォールバック：最適解判定ができない場合
                            # 【ログ記録】-2 = 非最適解（最適解判定ができない）
                            log_value_unique = -2
                            log_value_any = -2

                        ant_log_unique_optimal.append(log_value_unique)
                        ant_log_any_optimal.append(log_value_any)
                        active_ants.remove(ant)

            # 【既存実装互換】フェロモン揮発
            self._evaporate_pheromone_compatible()

            # 【既存実装互換】BKB揮発
            self._evaporate_bkb_compatible()

            # 結果を記録
            # interest: 現在のフェロモン分布で貪欲に1経路だけを選んだ場合の解
            interest_solution = self._greedy_pheromone_path(
                current_start, goal_nodes, self.ttl
            )

            # 各世代の最適解を記録（帯域変動が有効な場合、変動後のグラフで計算された最適解）
            current_gen_optimal_solutions = []
            if not start_switching_enabled:
                # スタート切り替えが無効な場合、現在の最適解を使用
                if current_optimal_solutions:
                    current_gen_optimal_solutions = current_optimal_solutions
                elif (
                    current_optimal_bottleneck is not None
                    and current_optimal_bottleneck > 0
                ):
                    # フォールバック：最適解リストが空の場合
                    # 【重要】manual環境では、optimal_solutionsが渡されている場合はそれを使用
                    if optimal_solutions and len(optimal_solutions) > 0:
                        current_gen_optimal_solutions = optimal_solutions
                    else:
                        current_gen_optimal_solutions = [
                            (
                                current_optimal_bottleneck,
                                current_optimal_delay or 0.0,
                                0,
                            )
                        ]
            else:
                # スタート切り替えが有効な場合、事前計算済みの最適解を使用
                if current_optimal_solutions:
                    current_gen_optimal_solutions = current_optimal_solutions
                elif (
                    current_optimal_bottleneck is not None
                    and current_optimal_bottleneck > 0
                ):
                    current_gen_optimal_solutions = [
                        (current_optimal_bottleneck, current_optimal_delay or 0.0, 0)
                    ]

            if interest_solution is not None:
                interest_bandwidth = bottleneck_capacity(
                    self.graph.graph, interest_solution, weight="bandwidth"
                )
                interest_delay = calculate_path_delay(
                    self.graph.graph, interest_solution
                )
                interest_hops = len(interest_solution) - 1
                interest_result = (
                    interest_bandwidth,
                    interest_delay,
                    interest_hops,
                )
            else:
                interest_result = None

            results.append(
                {
                    "generation": generation,
                    "solutions": generation_solutions,
                    "optimal_solutions": current_gen_optimal_solutions,  # 各世代の最適解（帯域変動を考慮）
                    "interest_solution": interest_result,
                }
            )

        return results, (ant_log_unique_optimal, ant_log_any_optimal)

    def _greedy_pheromone_path(
        self, start_node: int, goal_nodes: List[int], ttl: int
    ) -> Optional[List[int]]:
        """
        現在のフェロモン分布のみを頼りに貪欲に経路を構築する。
        同値の場合は帯域が大きい方、さらに同値なら遅延が小さい方を選ぶ。
        """
        if not goal_nodes:
            return None
        goal_set = set(goal_nodes)
        visited = set([start_node])
        path = [start_node]
        current = start_node
        steps = 0

        while current not in goal_set and steps < ttl:
            neighbors = [
                n for n in self.graph.get_neighbors(current) if n not in visited
            ]
            if not neighbors:
                return None

            def score(n: int) -> Tuple[float, float, float]:
                edge_attr = self.graph.get_edge_attributes(current, n)
                # 大きいほど良い指標でソートしたいので (-delay) で返す
                return (
                    edge_attr.get("pheromone", 0.0),
                    edge_attr.get("bandwidth", 0.0),
                    -edge_attr.get("delay", 0.0),
                )

            next_node = max(neighbors, key=score)
            edge_attr = self.graph.get_edge_attributes(current, next_node)

            # 遅延制約を満たさない場合は除外して再選択
            if self.delay_constraint_enabled:
                estimated_delay = (
                    calculate_path_delay(self.graph.graph, path + [next_node])
                    if len(path) > 1
                    else edge_attr.get("delay", float("inf"))
                )
                if estimated_delay > self.max_delay:
                    neighbors.remove(next_node)
                    if not neighbors:
                        return None
                    next_node = max(neighbors, key=score)

            path.append(next_node)
            visited.add(next_node)
            current = next_node
            steps += 1

        if current in goal_set:
            return path
        return None

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

        【ε-Greedy戦略】
        - 確率εでランダム選択（探索）：局所最適解への収束を防ぐ
        - 確率(1-ε)で確率的選択（活用）：フェロモンとヒューリスティック情報に基づく

        【遅延制約】
        遅延制約が有効な場合、制約を満たさない候補ノードは除外される。

        Args:
            ant: アリ

        Returns:
            次のノード（候補がない場合はNone）
        """
        neighbors = self.graph.get_neighbors(ant.current_node)
        candidates = [n for n in neighbors if not ant.has_visited(n)]

        if not candidates:
            return None

        # 【遅延制約チェック】制約を満たさない候補を除外
        if self.delay_constraint_enabled:
            valid_candidates = []
            for candidate in candidates:
                edge_attr = self.graph.get_edge_attributes(ant.current_node, candidate)
                estimated_delay = ant.total_delay + edge_attr["delay"]
                if estimated_delay <= self.max_delay:
                    valid_candidates.append(candidate)
            candidates = valid_candidates

        if not candidates:
            return None

        # ε-Greedy選択
        if random.random() < self.epsilon:
            # 【探索】ランダム選択：新しい経路を発見する可能性を維持
            return random.choice(candidates)
        else:
            # 【活用】フェロモンとヒューリスティックに基づく確率的選択
            return self._probabilistic_selection(ant, candidates)

    def _probabilistic_selection(self, ant: Ant, candidates: List[int]) -> int:
        """
        フェロモンとヒューリスティックに基づく確率的選択

        【Random Proportional Rule】
        選択確率 p_ij = (τ_ij^α * η_ij^β) / Σ(τ_il^α * η_il^β)
        - τ_ij: エッジ(i,j)のフェロモン量（過去の成功経験）
        - η_ij: エッジ(i,j)のヒューリスティック値（ローカル情報）
        - α: フェロモンの重要度
        - β: ヒューリスティックの重要度

        Args:
            ant: アリ
            candidates: 候補ノードのリスト

        Returns:
            選択されたノード
        """
        weights = []
        for candidate in candidates:
            edge_attr = self.graph.get_edge_attributes(ant.current_node, candidate)

            # 【フェロモン項】τ^α：過去の成功経験に基づく重み
            pheromone = edge_attr["pheromone"]
            tau = pheromone**self.alpha

            # 【ヒューリスティック項】η^β：ローカル情報（帯域、遅延）に基づく重み
            bandwidth = edge_attr["bandwidth"]
            delay = edge_attr["delay"]

            # ヒューリスティック値の計算（最適化目標に応じて変更）
            objectives = self.config["experiment"]["target_objectives"]

            if "delay" in objectives and len(objectives) > 1:
                # 【多目的最適化】帯域と遅延の両方を考慮
                # η = bandwidth^β_bandwidth * (1/delay)^β_delay
                # 帯域は大きいほど良い、遅延は小さいほど良い
                if delay > 0:
                    eta = (bandwidth**self.beta_bandwidth) * (
                        (1.0 / delay) ** self.beta_delay
                    )
                else:
                    eta = bandwidth**self.beta_bandwidth
            else:
                # 【単一目的最適化】帯域のみ考慮
                # η = bandwidth^β_bandwidth
                eta = bandwidth**self.beta_bandwidth

            # 【総合重み】フェロモンとヒューリスティックの積
            weight = tau * eta
            weights.append(weight)

        # 重みが全て0の場合はランダム選択（フェロモンが全くない初期状態など）
        if sum(weights) == 0:
            return random.choice(candidates)

        # 【確率選択】重みに比例した確率で選択
        return random.choices(candidates, weights=weights, k=1)[0]
