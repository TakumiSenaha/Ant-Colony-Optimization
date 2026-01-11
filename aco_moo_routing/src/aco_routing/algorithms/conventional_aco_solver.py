"""
ACS for MBL（Maximum Bottleneck Load）問題ソルバーモジュール

【アルゴリズム概要】
Ant Colony System (ACS) を最大ボトルネックリンク (MBL) 問題に適用した論文準拠実装。

出典: Dorigo, M., & Gambardella, L. M. (1997).
"Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem"
IEEE Transactions on Evolutionary Computation, Vol. 1, No. 1, pp. 53-66.

【TSPからMBLへの変換】
- 目的関数: 総距離の最小化 → ボトルネック帯域の最大化
- 評価方法: Σd_ij（総和） → min{w_ij}（最小値）
- ヒューリスティック: η=1/d_ij → η=w_ij/C_norm（正規化された帯域）
- 報酬: Δτ=1/L_gb → Δτ=B_gb/C_norm（正規化されたボトルネック帯域）

【ACS方式の特徴（論文準拠）】
1. 状態遷移規則（State Transition Rule）:
   - Pseudo-Random-Proportional Rule
   - 確率q₀=0.9で最良エッジを確定的に選択（Exploitation）
   - 確率(1-q₀)=0.1で確率的選択（Biased Exploration）

2. ローカル更新（Local Updating Rule）:
   - タイミング: エッジ訪問直後
   - 式: τ_ij ← (1-ξ)τ_ij + ξτ₀
   - 目的: 探索の多様性確保（同一経路への収束を防ぐ）

3. グローバル更新（Global Updating Rule）:
   - タイミング: 世代終了時
   - 対象: Global Best経路のみ（中央集権的）
   - 式: τ_ij ← (1-ρ)τ_ij + ρΔτ_ij（Δτ=ボトルネック帯域の正規化値）

【論文推奨パラメータ（Section III-D, p.56）】
- α = 1.0（フェロモン重要度）
- β = 2.0（ヒューリスティック重要度）
- q₀ = 0.9（Exploitation確率）
- ξ = 0.1（ローカル更新強度）
- ρ = 0.1（グローバル学習率）
- τ₀ = 1.0（初期フェロモン、MBL問題では楽観的初期化）

【提案手法との違い】
- ノード学習機能（BKB/BLD/BKH）: 使用しない
- 功績ボーナス/ペナルティ: 使用しない
- 更新タイミング: 世代終了時にGlobal Bestのみ更新（提案手法は全アリが即座に更新）
- 分散性: 中央集権的（全アリの解を比較して最良を選択）
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
from .single_objective_solver import (
    bottleneck_capacity,
    calculate_path_delay,
    find_all_max_bottleneck_paths_with_delay_constraint,
    max_load_path,
    max_load_path_with_delay_constraint,
)


class ConventionalACOSolver:
    """
    ACS for MBL ソルバークラス（論文準拠実装）

    出典: Dorigo, M., & Gambardella, L. M. (1997).
    "Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem"
    IEEE Transactions on Evolutionary Computation, Vol. 1, No. 1, pp. 53-66.

    【アルゴリズムの詳細】

    1. 状態遷移規則（Pseudo-Random-Proportional Rule）:
       - 確率q₀で決定論的選択: j = argmax{τ_ij^α · η_ij^β}
       - 確率(1-q₀)で確率的選択: p_ij = [τ_ij^α · η_ij^β] / Σ[τ_il^α · η_il^β]
       - ヒューリスティック η_ij = w_ij / C_norm（正規化された帯域幅）

    2. ローカル更新規則（Local Updating Rule）:
       - エッジ訪問直後に適用
       - 式: τ_ij ← (1-ξ)τ_ij + ξτ₀
       - パラメータ: ξ=0.1, τ₀=1.0

    3. グローバル更新規則（Global Updating Rule）:
       - 世代終了時、Global Best経路のみ更新
       - 揮発: τ_ij ← (1-ρ)τ_ij（全エッジ）
       - 付加: τ_ij ← τ_ij + ρΔτ（Global Best経路のみ）
       - 報酬: Δτ = B_gb / C_norm（ボトルネック帯域の正規化値）
       - パラメータ: ρ=0.1

    【論文推奨パラメータ（Section III-D, p.56）】
    - α = 1.0: フェロモン重要度
    - β = 2.0: ヒューリスティック重要度（太い回線を強く優遇）
    - q₀ = 0.9: Exploitation確率
    - ξ = 0.1: ローカル更新強度
    - ρ = 0.1: グローバル学習率
    - τ₀ = 1.0: 初期フェロモン（楽観的初期化）
    - C_norm = 100.0: 帯域正規化定数

    【MBL問題固有の設定】
    - ヒューリスティック: η_ij = w_ij / 100.0（0.1〜1.0に正規化）
    - 報酬: Δτ = B_gb / 100.0（0.1〜1.0に正規化）
    - TSPの η=1/d_ij, Δτ=1/L_gb との対応関係を維持

    Attributes:
        config (Dict): 設定辞書
        graph (RoutingGraph): ルーティンググラフ
        evaluator (SolutionEvaluator): 目的関数に基づく評価器
        fluctuation_model (Optional[BandwidthFluctuationModel]): 帯域変動モデル
        alpha (float): フェロモン重要度（論文推奨値: 1.0）
        beta_bandwidth (float): ヒューリスティック重要度（論文推奨値: 2.0）
        beta_delay (float): 遅延ヒューリスティックの重み
        q0 (float): Exploitation確率（論文推奨値: 0.9）
        ttl (int): アリの最大ステップ数
        global_best_solution (Optional[Tuple[float, float, int]]): Global Best解
        global_best_ant (Optional[Ant]): Global Bestアリ
        local_update_xi (float): ローカル更新強度（論文推奨値: 0.1）
        initial_pheromone (float): 初期フェロモン値τ₀（論文準拠: 1.0）
        bandwidth_normalization (float): 帯域正規化定数C_norm（100.0）

    Note:
        ACS論文準拠のため、フェロモン更新は直接実装しており、
        SimplePheromoneUpdater/Evaporatorは使用しない。
    """

    def __init__(self, config: Dict, graph: RoutingGraph):
        """
        Args:
            config: 設定辞書
            graph: ルーティンググラフ
        """
        self.config = config
        self.graph = graph

        # 評価関数（最適解判定用）
        self.evaluator = SolutionEvaluator(config["experiment"]["target_objectives"])

        # 注意: ACS論文準拠実装では、フェロモン更新は
        # Global Bestの経路に対して直接 τ_ij ← (1-ρ)τ_ij + ρΔτ_ij を適用
        # SimplePheromoneUpdater/Evaporatorは使用しない

        # 帯域変動モデル
        self.fluctuation_model: Optional[BandwidthFluctuationModel] = None
        self.edge_states: Dict = {}
        if config["graph"]["fluctuation"]["enabled"]:
            self._initialize_fluctuation_model()

        # === ACS for MBL 論文準拠パラメータ（全て明示的に設定）===
        # 出典: Dorigo & Gambardella (1997) - Section III-D. ACS Parameter Settings
        # 注意: config.yamlの値は使用せず、ACS論文推奨値を直接設定

        # α: フェロモン重要度（論文推奨値: 1.0）
        # 確率計算でのみ使用: p_ij ∝ τ_ij^α · η_ij^β
        self.alpha = 1.0

        # β: ヒューリスティック重要度（論文推奨値: 2.0）
        # 帯域幅を2乗することで、太い回線を強く優遇
        # 例: 100Mbps (η=1.0) → η^2=1.0, 50Mbps (η=0.5) → η^2=0.25
        # 注意: basic_aco_no_heuristic/basic_aco_with_heuristicの場合は
        #       run_experiment.pyでconfigを変更しているので、それを優先
        self.beta_bandwidth = config["aco"].get("beta_bandwidth", 2.0)
        self.beta_delay = 1.0  # 遅延制約時に使用

        # q₀: Exploitation確率（論文推奨値: 0.9）
        # 90%の確率で最良エッジを確定的に選択、10%の確率で確率的選択
        self.q0 = 0.9

        # TTL: アリの最大ステップ数（共通パラメータ、configから取得）
        self.ttl = config["aco"]["ttl"]

        # === ローカル更新用パラメータ ===
        # ξ: ローカル更新強度（論文推奨値: 0.1）
        # 式: τ_ij ← (1-ξ)τ_ij + ξτ₀
        self.local_update_xi = 0.1

        # τ₀: 初期フェロモン値（MBL問題での設定: 1.0）
        # 楽観的初期化: 未探索エッジは「最大帯域である可能性がある」と見なす
        # TSPの τ₀ = 1/(n·L_nn) とは異なるアプローチ
        self.initial_pheromone = 1.0

        # C_norm: 帯域正規化定数（MBL問題固有: 100.0）
        # η_ij = w_ij / C_norm により、ヒューリスティック値を0.1〜1.0に正規化
        self.bandwidth_normalization = 100.0

        # ρ: 大域学習率（論文推奨値: 0.1）
        # 【重要】ACS論文準拠の値を使用（configの値を上書き）
        self.evaporation_rate = 0.1  # ACS論文推奨値

        # フェロモンの範囲（正規化後のスケール）
        # 【重要】ACS論文準拠の値を使用（configの値を上書き）
        self.min_pheromone = 0.01  # 正規化後のスケール
        self.max_pheromone = 10.0  # 正規化後のスケール

        # グラフのフェロモン値をACS論文準拠の値に再初期化
        # RoutingGraphはconfig["aco"]["min_pheromone"]=100で初期化されているが、
        # ConventionalACOSolverでは正規化スケール（τ₀=1.0）を使用
        self._reinitialize_pheromones()

        # 遅延制約
        delay_constraint = config["experiment"].get("delay_constraint", {})
        self.delay_constraint_enabled = delay_constraint.get("enabled", False)
        self.max_delay = delay_constraint.get("max_delay", float("inf"))

        # Global Best（実験開始から現在までの最良解）
        self.global_best_solution: Optional[Tuple[float, float, int]] = None
        self.global_best_ant: Optional[Ant] = None
        self.global_best_generation: Optional[int] = (
            None  # グローバル解が見つかった世代
        )
        self.global_best_max_age: int = config["aco"].get(
            "global_best_max_age", 100000
        )  # グローバル解の有効期限（世代数）

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

    def _reinitialize_pheromones(self) -> None:
        """
        グラフのフェロモン値をACS論文準拠の値に再初期化

        RoutingGraphはconfig["aco"]["min_pheromone"]=100で初期化されているが、
        ConventionalACOSolverでは正規化スケール（τ₀=1.0, min=0.01, max=10.0）を使用する。
        """
        G = self.graph.graph
        for u, v in G.edges():
            # 双方向のフェロモン値を初期値τ₀に設定
            G.edges[u, v]["pheromone"] = self.initial_pheromone
            G.edges[v, u]["pheromone"] = self.initial_pheromone

            # min/max_pheromoneを正規化スケールに設定
            G.edges[u, v]["min_pheromone"] = self.min_pheromone
            G.edges[v, u]["min_pheromone"] = self.min_pheromone
            G.edges[u, v]["max_pheromone"] = self.max_pheromone
            G.edges[v, u]["max_pheromone"] = self.max_pheromone

    def run(
        self,
        start_node: int,
        goal_node: int,
        generations: int,
        optimal_solutions: Optional[List[Tuple[float, float, int]]] = None,
        metrics_calculator=None,
    ) -> Tuple[List[Dict], List[int]]:
        """
        ACOを実行（従来手法）

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
        ant_log = []  # 各アリが到達した時の成功/失敗を記録

        # 【初期フェロモン値の確認】論文準拠: τ₀ = 1.0（楽観的初期化）
        print(
            f"初期フェロモン値τ₀: {self.initial_pheromone:.2f} "
            f"（論文準拠: 楽観的初期化、未探索エッジは最大帯域を仮定）"
        )

        # 【Global Bestの初期化】実験開始時にリセット
        self.global_best_solution = None
        self.global_best_ant = None
        self.global_best_generation = None

        # スタートノード切り替えの設定を取得
        start_switching_config = self.config["experiment"].get("start_switching", {})
        start_switching_enabled = start_switching_config.get("enabled", False)
        switch_interval = start_switching_config.get("switch_interval", 200)

        # スタートノードリストを取得または生成
        if start_switching_enabled:
            start_nodes = start_switching_config.get("start_nodes", [])
            if not start_nodes:
                # 自動生成: グラフのノード数に基づいてランダムに選択
                # 初期ゴールノードを除外してから選択（スタートとゴールが同じだと探索が無意味）
                num_nodes = self.graph.num_nodes
                num_start_nodes = 10
                all_nodes = list(range(num_nodes))
                # 初期ゴールノードを除外
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

        # 帯域のみ最適化の場合、最適解を再計算するかどうか
        # 遅延制約が有効な場合、target_objectivesは["bandwidth", "delay"]に変更されているが、
        # フェロモン付加・揮発・ノード学習は["bandwidth", "delay"]と同じ（bandwidth/delayスコア）
        # 最適解判定は単一最適解として扱う（パレート最適化ではない）
        recalculate_optimal = (
            self.config["experiment"]["target_objectives"] == ["bandwidth"]
            or self.delay_constraint_enabled
        )

        # update_intervalを取得
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
        optimal_solutions_dict: Dict[int, List[Tuple[float, float, int]]] = (
            {}
        )  # 遅延制約ありの場合：複数の最適解のリスト

        # 前のスタートノードを記録（ゴール追加用）
        previous_start = None

        for generation in range(generations):
            # 【グローバル解のTTLチェック】グローバル解が古くなったら無効化
            if (
                self.global_best_generation is not None
                and generation - self.global_best_generation >= self.global_best_max_age
            ):
                self.global_best_solution = None
                self.global_best_ant = None
                self.global_best_generation = None

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
                    if current_start in goal_nodes:
                        print(
                            f"警告: スタートノード {current_start} は既にゴールです。このフェーズをスキップします。"
                        )
                        optimal_bottleneck_dict[current_start] = -1
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
                                    all_optimal_paths = find_all_max_bottleneck_paths_with_delay_constraint(
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

            # 現在のゴールノード
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
            if start_switching_enabled:
                current_optimal_bottleneck = optimal_bottleneck_dict.get(
                    current_start, -1
                )
                current_optimal_solutions = optimal_solutions_dict.get(
                    current_start, []
                )  # 複数の最適解のリスト
                if current_optimal_bottleneck <= 0:
                    continue
            else:
                # スタート切り替えが無効な場合のみ、帯域変動時に再計算
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
                                for path in all_optimal_paths:
                                    bottleneck = bottleneck_capacity(
                                        self.graph.graph, path, weight="bandwidth"
                                    )
                                    delay = calculate_path_delay(self.graph.graph, path)
                                    hops = len(path) - 1
                                    current_optimal_solutions.append(
                                        (bottleneck, delay, hops)
                                    )
                            else:
                                current_optimal_bottleneck = None
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
                        current_optimal_solutions = []
                else:
                    # 再計算しない場合は、前回の値を保持（最初の世代以外）
                    if generation == 0:
                        current_optimal_bottleneck = None
                        current_optimal_solutions = []

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

            # 【アリの探索ループ】ACS方式：解構築中にローカル更新、世代終了時にGlobal Best更新
            generation_solutions = []
            generation_ants = []  # ゴール到達したアリを保存（Global Best選択用）
            active_ants = list(ants)  # アクティブなアリのリスト

            while active_ants:
                for ant in list(active_ants):  # リストのコピーを反復（削除に対応）
                    # 【既存実装との互換性】ゴール判定 → 探索/移動 → TTLチェック の順序

                    # 【ゴール到達判定】Anycast方式：ゴール群のいずれかに到達したかチェック
                    has_reached_any_goal = ant.current_node in goal_nodes_set

                    # 【探索継続】ゴール未到達の場合、次のノードを選択して移動
                    if not has_reached_any_goal:
                        next_node = self._select_next_node(ant)
                        if next_node is None:
                            # 行き先がない場合（全ての隣接ノードを訪問済み）は探索失敗
                            active_ants.remove(ant)
                            # 【ログ記録】ゴール未到達 = -1
                            ant_log.append(-1)
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
                                ant_log.append(-1)
                                continue

                        # 移動前のノードを記録（ローカル更新用）
                        previous_node = ant.current_node

                        # 移動（ボトルネック帯域、累積遅延、ホップ数を更新）
                        ant.move_to(
                            next_node, edge_attr["bandwidth"], edge_attr["delay"]
                        )

                        # 【ローカル更新】ACS方式：訪問エッジのフェロモンを即座に減少
                        # 式: τ_ij ← (1-ξ)τ_ij + ξτ_0
                        # 移動前のノードから移動後のノードへのエッジを更新
                        self._apply_local_update(previous_node, next_node)

                        # 移動後にゴール判定（移動先がゴールだった場合）
                        has_reached_any_goal = ant.current_node in goal_nodes_set

                        # 【TTLチェック】ゴール未到達の場合のみ、TTLをチェック
                        # 既存実装: elif len(ant.route) >= TTL: ant_log.append(0)
                        if not has_reached_any_goal and ant.remaining_ttl <= 1:
                            active_ants.remove(ant)
                            # 【ログ記録】ゴール未到達 = -1
                            ant_log.append(-1)
                            continue

                    # 【ゴール到達時の処理】解を記録するだけ（フェロモン更新は世代終了時）
                    if has_reached_any_goal:
                        solution = ant.get_solution()

                        # 【遅延制約チェック】ゴール到達時にも制約を確認
                        if self.delay_constraint_enabled:
                            solution_delay = solution[1]  # total_delay
                            if solution_delay > self.max_delay:
                                # 制約違反の場合は探索失敗として扱う
                                active_ants.remove(ant)
                                # 【ログ記録】制約違反 = -1（ゴール未到達）
                                ant_log.append(-1)
                                continue

                        generation_solutions.append(solution)
                        generation_ants.append(ant)  # Global Best選択用に保存

                        # 【ログ記録】最適解判定結果を記録
                        # 【既存実装との互換性】1 = 最適解、0 = 失敗
                        if recalculate_optimal:
                            # 【最適解判定】帯域のみ最適化または遅延制約が有効な場合
                            if current_optimal_bottleneck is not None:
                                if self.delay_constraint_enabled:
                                    # 遅延制約が有効な場合：複数の最適解リストと比較
                                    solution_bandwidth = solution[0]
                                    solution_delay = solution[1]
                                    # 最適解リストのいずれかと一致するかチェック
                                    is_optimal = False
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
                                                is_optimal = True
                                                break
                                    # 【ログ記録】1 = 最適解、-2 = 非最適解
                                    log_value = 1 if is_optimal else -2
                                else:
                                    # 遅延制約なし：帯域で判定
                                    solution_bandwidth = solution[0]
                                    bandwidth_ok = (
                                        solution_bandwidth >= current_optimal_bottleneck
                                    )
                                    # 【ログ記録】1 = 最適解、-2 = 非最適解
                                    log_value = 1 if bandwidth_ok else -2
                            else:
                                # 最適解が計算されていない場合（初期世代など）
                                # 【ログ記録】-2 = 非最適解（最適解が計算されていない）
                                log_value = -2
                        elif optimal_solutions and metrics_calculator:
                            # 【多目的最適化】パレート最適化の場合
                            # 見つけた解がパレートフロンティアのどの解に対応するかを判定
                            optimal_index = (
                                metrics_calculator.find_optimal_solution_index(
                                    solution, optimal_solutions
                                )
                            )
                            # 【ログ記録】1 = 最適解、-2 = 非最適解
                            log_value = 1 if optimal_index is not None else -2
                        else:
                            # フォールバック：最適解判定ができない場合
                            # 【ログ記録】-2 = 非最適解（最適解判定ができない）
                            log_value = -2

                        ant_log.append(log_value)
                        active_ants.remove(ant)

            # === グローバル更新（Global Updating Rule）===
            # 出典: Dorigo & Gambardella (1997) - Equation (4)
            #
            # 【アルゴリズム】
            # 1. 世代内で最良のアリ（Generation Best）を選択
            # 2. Generation BestがGlobal Bestより良ければ更新
            # 3. Global Bestの経路にのみフェロモンを付加
            #
            # 【論文の式（TSP版）】
            # Δτ_ij = 1/L_gb（Global Best経路の総距離の逆数）
            #
            # 【MBL問題への適用】
            # Δτ_ij = B_gb / C_norm（Global Bestのボトルネック帯域、正規化後）
            # - 広い帯域ほどΔτが大きい（TSPの「短い経路」に対応）
            # - 正規化により、Δτは0.1〜1.0の範囲に収まる
            #
            # 【中央集権的な更新】
            # 全アリの解を比較して最良を選択する必要がある（完全分散ではない）
            if generation_ants:
                # 最良のアリを選択（MBL問題ではボトルネック帯域が最大のもの）
                best_ant = None
                if self.delay_constraint_enabled:
                    # 遅延制約あり：帯域が最大で、その中で遅延が最小
                    valid_ants = [
                        (ant, ant.get_solution())
                        for ant in generation_ants
                        if ant.get_solution()[1] <= self.max_delay
                    ]
                    if valid_ants:
                        best_ant, _ = max(
                            valid_ants,
                            key=lambda x: (x[1][0], -x[1][1]),  # (bandwidth, -delay)
                        )
                else:
                    # 遅延制約なし：ボトルネック帯域が最大
                    best_ant, _ = max(
                        [(ant, ant.get_solution()) for ant in generation_ants],
                        key=lambda x: x[1][0],  # bandwidth
                    )

                # Global Bestを更新
                if best_ant is not None:
                    best_solution = best_ant.get_solution()

                    # 現在のGlobal Bestより良い解が見つかった場合のみ更新
                    if (
                        self.global_best_solution is None
                        or best_solution[0] > self.global_best_solution[0]
                        or (
                            abs(best_solution[0] - self.global_best_solution[0]) < 1e-6
                            and self.delay_constraint_enabled
                            and best_solution[1] < self.global_best_solution[1]
                        )
                    ):
                        self.global_best_solution = best_solution
                        self.global_best_ant = best_ant
                        self.global_best_generation = (
                            generation  # グローバル解が見つかった世代を記録
                        )

            # === グローバル更新（Global Updating Rule）===
            # 出典: Dorigo & Gambardella (1997) - Equation (4)
            # 論文の式: τ_ij ← (1-ρ)τ_ij + ρΔτ_ij
            #
            # 【重要】ACSでは「Global Bestに属するエッジのみ」を更新します。
            # それ以外のエッジは一切触りません（揮発もしません）。
            #
            # 【理由】
            # - ローカル更新で、訪問済みエッジは既に減少している
            # - グローバル更新で全エッジを揮発すると「二重揮発」になる
            # - これはAnt System (AS)の仕様であり、ACSではない
            #
            # 【論文準拠の実装】
            # Global Bestのエッジのみに対して：
            #   1. 揮発: τ_ij ← (1-ρ)τ_ij
            #   2. 付加: τ_ij ← τ_ij + ρΔτ_ij
            #   → 結果: τ_ij ← (1-ρ)τ_ij + ρΔτ_ij
            #
            # ここで、Δτ_ij = B_gb / C_norm（正規化されたボトルネック帯域）

            if self.global_best_ant is not None:
                # 【報酬計算】Global Bestのボトルネック帯域を取得
                best_solution = self.global_best_ant.get_solution()
                # get_solution()は (bandwidth, delay, hops) のタプルを返す
                bottleneck = best_solution[0]

                # 【正規化】ボトルネック帯域を0〜1の範囲に変換
                # TSPの Δτ = 1/L_gb に対応するMBL問題の式
                delta_tau = bottleneck / self.bandwidth_normalization

                # 【大域学習率 ρ】論文推奨値: 0.1
                rho = self.evaporation_rate

                # 【Global Bestの経路上のエッジのみ更新】
                # 論文の式(4): τ_ij ← (1-ρ)τ_ij + ρΔτ_ij
                route_edges = self.global_best_ant.get_route_edges()
                for u, v in route_edges:
                    edge_attr = self.graph.get_edge_attributes(u, v)
                    current_tau = edge_attr["pheromone"]

                    # 論文の式を直接適用
                    new_tau = (1.0 - rho) * current_tau + rho * delta_tau

                    # フェロモンの差分を計算して更新
                    delta_pheromone = new_tau - current_tau
                    self.graph.update_pheromone(
                        u, v, delta_pheromone, bidirectional=True
                    )

            # 【ノード学習値の揮発】従来手法ではノード学習機能を使用しないため、揮発処理は不要

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
                    current_gen_optimal_solutions = [
                        (current_optimal_bottleneck, 0.0, 0)
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
                        (current_optimal_bottleneck, 0.0, 0)
                    ]

            # 結果を記録
            results.append(
                {
                    "generation": generation,
                    "solutions": generation_solutions,
                    "optimal_solutions": current_gen_optimal_solutions,  # 各世代の最適解（帯域変動を考慮）
                }
            )

        return results, ant_log

    def _select_next_node(self, ant: Ant) -> Optional[int]:
        """
        状態遷移規則（Pseudo-Random-Proportional Rule）

        出典: Dorigo & Gambardella (1997) - State Transition Rule

        【アルゴリズム】
        1. 乱数 q ∈ [0,1] を生成
        2. q ≤ q₀ の場合: Exploitation（確定的選択）
           j = argmax{τ_ij^α · η_ij^β}
        3. q > q₀ の場合: Biased Exploration（確率的選択）
           p_ij = [τ_ij^α · η_ij^β] / Σ[τ_il^α · η_il^β]

        【パラメータ（論文推奨値）】
        - q₀ = 0.9: Exploitation確率（90%の確率で最良エッジを選択）
        - α = 1.0: フェロモン重要度
        - β = 2.0: ヒューリスティック重要度

        【MBL問題での適用】
        - η_ij = w_ij / C_norm: 正規化された帯域幅（0.1〜1.0）
        - β=2.0により、太い回線を強く優遇

        【遅延制約】
        遅延制約が有効な場合、制約を満たさない候補は除外される。

        Args:
            ant: アリ

        Returns:
            次のノード（候補がない場合はNone）
        """
        # 【候補ノードの取得】未訪問の隣接ノードのみ
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

        # 【Pseudo-Random-Proportional Rule】論文の状態遷移規則
        q = random.random()  # 乱数 q ∈ [0,1]

        if q <= self.q0:
            # 【Exploitation】確率q₀で最良エッジを確定的に選択
            # 既知の良い経路を活用
            return self._select_best_edge(ant, candidates)
        else:
            # 【Biased Exploration】確率(1-q₀)で確率的選択
            # 探索の多様性を確保
            return self._probabilistic_selection(ant, candidates)

    def _probabilistic_selection(self, ant: Ant, candidates: List[int]) -> int:
        """
        確率的選択（Biased Exploration）

        出典: Dorigo & Gambardella (1997) - Pseudo-Random-Proportional Rule
        論文の式: p_ij = [τ_ij]^α · [η_ij]^β / Σ_l [τ_il]^α · [η_il]^β

        MBL問題での適用:
        - τ_ij: フェロモン（過去の成功経験）
        - η_ij: ヒューリスティック = w_ij / C_norm（正規化された帯域幅）
        - α = 1.0（論文推奨値）
        - β = 2.0（論文推奨値、太い回線を強く優遇）

        確率的選択により、フェロモンが少ないエッジも選ばれる可能性がある
        → 探索の多様性を確保（Exploration）

        Args:
            ant: アリ
            candidates: 候補ノードのリスト

        Returns:
            選択されたノード
        """
        weights = []
        for candidate in candidates:
            edge_attr = self.graph.get_edge_attributes(ant.current_node, candidate)

            # 【フェロモン項】τ^α（論文式）
            pheromone = edge_attr["pheromone"]
            weight_pheromone = pheromone**self.alpha

            # 【ヒューリスティック項】η^β（論文式）
            if self.beta_bandwidth > 0:
                # 帯域幅を正規化: η_ij = w_ij / C_norm（0.1〜1.0の範囲）
                bandwidth = edge_attr["bandwidth"]
                eta = bandwidth / self.bandwidth_normalization
                weight_heuristic = eta**self.beta_bandwidth
                weight = weight_pheromone * weight_heuristic
            else:
                # beta_bandwidth=0の場合：フェロモンのみを使用
                weight = weight_pheromone

            weights.append(weight)

        # 重みが全て0の場合はランダム選択（フォールバック）
        if sum(weights) == 0:
            return random.choice(candidates)

        # 【確率的選択】ルーレットホイール選択
        # 重みに比例した確率で候補ノードを選択
        return random.choices(candidates, weights=weights, k=1)[0]

    def _select_best_edge(self, ant: Ant, candidates: List[int]) -> int:
        """
        最良エッジを確定的に選択（Exploitation）

        出典: Dorigo & Gambardella (1997) - Pseudo-Random-Proportional Rule
        論文の式: j = argmax{τ_ij^α · η_ij^β}

        MBL問題での適用:
        - τ_ij: フェロモン（過去の成功経験）
        - η_ij: ヒューリスティック = w_ij / C_norm（正規化された帯域幅）
        - α = 1.0（論文推奨値）
        - β = 2.0（論文推奨値）

        確定的選択により、現時点で最も有望なエッジを選ぶ
        → 既知の良い経路を活用（Exploitation）

        Args:
            ant: アリ
            candidates: 候補ノードのリスト

        Returns:
            最良エッジに対応するノード
        """
        best_node = None
        best_value = -1.0

        for candidate in candidates:
            edge_attr = self.graph.get_edge_attributes(ant.current_node, candidate)

            # 【フェロモン項】τ^α（論文式）
            pheromone = edge_attr["pheromone"]
            tau = pheromone**self.alpha

            # 【ヒューリスティック項】η^β（論文式）
            bandwidth = edge_attr["bandwidth"]
            delay = edge_attr["delay"]

            # ヒューリスティック値の計算
            if self.beta_bandwidth > 0:
                # 帯域幅を正規化: η_ij = w_ij / C_norm
                eta_bandwidth = bandwidth / self.bandwidth_normalization

                if self.beta_delay > 0 and self.delay_constraint_enabled:
                    # 帯域と遅延の両方を考慮（遅延制約が有効な場合）
                    if delay > 0:
                        eta_delay = 1.0 / delay
                        eta = (eta_bandwidth**self.beta_bandwidth) * (
                            eta_delay**self.beta_delay
                        )
                    else:
                        eta = eta_bandwidth**self.beta_bandwidth
                else:
                    # 帯域のみ考慮
                    eta = eta_bandwidth**self.beta_bandwidth
            else:
                # ヒューリスティックなし（β=0の場合）
                eta = 1.0

            # 【総合値】τ^α * η^β
            value = tau * eta

            if value > best_value:
                best_value = value
                best_node = candidate

        # 最良エッジが見つからない場合（フォールバック）
        return best_node if best_node is not None else random.choice(candidates)

    def _apply_local_update(self, u: int, v: int) -> None:
        """
        ローカル更新（Local Updating Rule）

        出典: Dorigo & Gambardella (1997) - Equation (3)
        論文の式: τ_ij ← (1-ξ)τ_ij + ξτ₀

        パラメータ（論文推奨値）:
        - ξ = 0.1: ローカル更新強度
        - τ₀ = 1.0: 初期フェロモン値（MBL問題では楽観的初期化）

        【目的】
        訪問したエッジのフェロモンを τ₀ に近づけることで、後続のアリが
        同じ経路を選ぶ確率を調整し、探索の多様性を維持する。

        【動作】
        - 初期状態（τ_ij = 1.0）の場合: 変化なし
        - グローバル更新後（τ_ij > 1.0）の場合: 徐々に 1.0 に戻す（減少）
        - これにより、フェロモンが過剰に蓄積したエッジの魅力を下げる

        【TSPとの違い】
        - TSP: τ₀ ≈ 0.001（小さい値）→ 訪問済みエッジは減少
        - MBL: τ₀ = 1.0（大きい値）→ グローバル更新後のエッジが減少

        Args:
            u: 始点ノード
            v: 終点ノード
        """
        edge_attr = self.graph.get_edge_attributes(u, v)
        current_pheromone = edge_attr["pheromone"]

        # 論文の式: τ_new = (1-ξ)τ_old + ξτ₀
        new_pheromone = (
            1 - self.local_update_xi
        ) * current_pheromone + self.local_update_xi * self.initial_pheromone

        delta = new_pheromone - current_pheromone
        self.graph.update_pheromone(u, v, delta, bidirectional=True)
