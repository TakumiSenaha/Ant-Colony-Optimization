"""
フェロモン更新・揮発ロジック

【提案手法のフェロモン更新】
1. アリがゴールに到達すると、経路上の各ノードの学習値（BKB/BLD/BKH）を更新
2. 経路上の各エッジにフェロモンを付加
3. 功績ボーナス：見つけた解がノードの記憶を上回る場合、フェロモン付加量を増加

【提案手法のフェロモン揮発】
1. BKBベースのペナルティ付き揮発：エッジの帯域がノードのBKBより低い場合、揮発を促進
2. 双方向揮発：各エッジは両方向で独立に揮発（ノードごとのBKBに基づく）
3. これにより、有望でないエッジのフェロモンを急速に減少させ、探索効率を向上
"""

import math
from typing import Dict, Optional, Tuple

from ..core.ant import Ant
from ..core.graph import RoutingGraph
from .evaluator import SolutionEvaluator


class PheromoneUpdater:
    """
    フェロモン更新を管理するクラス

    Attributes:
        config (Dict): 設定辞書
        evaluator (SolutionEvaluator): 目的関数に基づく評価器
        bonus_factor (float): 功績ボーナス係数
        delay_tolerance (float): 遅延許容誤差（後方互換用）
    """

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

        【更新プロセス】
        1. 経路上の各ノードの学習値（BKB/BLD/BKH）を更新
        2. 更新前の値を記録（功績ボーナス判定に使用）
        3. 経路上の各エッジにフェロモンを付加
        4. 功績ボーナス：見つけた解がノードの記憶を上回る場合、フェロモン付加量を増加

        Args:
            ant: ゴールに到達したアリ
            graph: ルーティンググラフ
        """
        # アリの解を取得（ボトルネック帯域、累積遅延、ホップ数）
        bandwidth, delay, hops = ant.get_solution()

        # 【遅延制約チェック】制約違反のパスは学習・フェロモン付加を行わない
        delay_constraint_config = self.config["experiment"].get("delay_constraint", {})
        delay_constraint_enabled = delay_constraint_config.get("enabled", False)
        max_delay = delay_constraint_config.get("max_delay", float("inf"))

        if delay_constraint_enabled and delay > max_delay:
            # 制約違反：フェロモン付加量 = 0（学習も行わない）
            return

        # 【Step 1】各ノードの学習値（BKB/BLD/BKH）を更新し、更新前の値を記録
        # 更新前の値は、功績ボーナス判定に使用（更新後の値と比較すると常に更新されるため）
        # 【重要】制約内のパスのみ学習する（制約違反は既にreturnで除外済み）
        node_old_memory: Dict[int, Tuple[float, float, float]] = {}

        for node in ant.route:
            # 更新前の値を記録
            node_old_memory[node] = (
                graph[node].bkb,
                graph[node].bld,
                graph[node].bkh,
            )

            # 学習値を更新（帯域のみ：K_v ← max(K_v, B)）
            graph[node].update_bandwidth(bandwidth)

        # 【Step 2】経路上の各エッジにフェロモンを付加（帰還時の処理）
        # アリはゴールからスタートへ戻りながら、各エッジにフェロモンを付加
        route_edges = ant.get_route_edges()

        for i, (u, v) in enumerate(route_edges):
            # 【基本フェロモン付加量】f(B) = 10 * B
            # 【遅延制約が有効な場合】帯域/遅延のスコアを使用
            # Δτ = 10 * (B / D_path)
            if delay_constraint_enabled:
                if delay > 0:
                    base_pheromone = 10.0 * (bandwidth / delay)
                else:
                    base_pheromone = bandwidth * 10.0
            else:
                base_pheromone = bandwidth * 10.0

            # 【功績ボーナス判定】分散判断：各ノードが独立に判定
            # エッジ(u, v)の処理 = 帰還時にノードvからノードuへ戻る処理に対応
            # この時点で、アリはノードvの記憶値（更新前）を知っている
            k_v, l_v, m_v = node_old_memory[v]

            # 【功績ボーナス適用】B >= K_v の場合、ボーナス係数 B_a を適用
            if bandwidth >= k_v:
                delta_pheromone = base_pheromone * self.bonus_factor
            else:
                delta_pheromone = base_pheromone

            # 【フェロモン更新】双方向にフェロモンを付加
            # エッジ(u, v)と(v, u)の両方に同じ量を付加（対称性を保つ）
            graph.update_pheromone(u, v, delta_pheromone, bidirectional=True)


class PheromoneEvaporator:
    """
    フェロモン揮発を管理するクラス

    Attributes:
        config (Dict): 設定辞書
        evaporation_rate (float): 揮発率
        penalty_factor (float): ペナルティ係数（BKBベース揮発用）
        volatilization_mode (int): 揮発モード（3でBKBペナルティ揮発）
        evaluator (Optional[SolutionEvaluator]): 多目的時の評価器
        delay_tolerance (float): 遅延許容誤差（後方互換用）
    """

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

        【核心原理】
        エッジ単体の帯域がBKBより低ければ、そのエッジを使った経路が
        BKBを超えることは数学的に不可能（ボトルネック帯域はmin演算のため）。
        遅延は経路全体で挽回可能なため、エッジ単体でのペナルティ判定は帯域を主基準とする。

        【双方向揮発の原則】
        - 全てのエッジは必ず何かしらのノードから出ている
        - エッジ(u, v)は「ノードuから出ていくパス」→ ノードuのBKBと比較
        - エッジ(v, u)は「ノードvから出ていくパス」→ ノードvのBKBと比較
        - 全てのエッジは揮発するが、ペナルティは「自分から出ていくパス」にのみ適用
        - パレート最適化の場合でも、帯域は依然として重要な指標であるため、
          このBKBベースのペナルティは有効に機能する

        Args:
            graph: ルーティンググラフ
        """
        # 全てのエッジを双方向で処理
        for u, v in graph.graph.edges():
            # 【エッジ(u → v)の揮発計算】
            # ノードuから出ていくパスとして、ノードuのBKBと比較
            self._apply_evaporation_to_edge(graph, u, v)

            # 【エッジ(v → u)の揮発計算】
            # ノードvから出ていくパスとして、ノードvのBKBと比較
            # 注意：ノードuとノードvのBKBは異なる可能性があるため、
            #       エッジ(u → v)とエッジ(v → u)の揮発のされ方が異なる
            self._apply_evaporation_to_edge(graph, v, u)

    def _apply_evaporation_to_edge(self, graph: RoutingGraph, u: int, v: int) -> None:
        """
        単一方向のエッジ (u → v) に対して揮発処理を適用

        【重要原則】ペナルティは「自分から出ていくパス」にのみ適用
        - エッジ(u → v)は「ノードuから出ていくパス」
        - ノードuのBKBと比較してペナルティを判定
        - 「自分を通ってこの先、この品質でゴールできる」という情報に基づく

        Args:
            graph: ルーティンググラフ
            u: 始点ノード（このノードから出ていくパス）
            v: 終点ノード
        """
        # エッジの属性を取得
        edge_bandwidth = graph.graph.edges[u, v]["bandwidth"]

        # ノードuの学習値（BKB）：ノードuから出ていくパスなので、ノードuのBKBと比較
        bkb_u = graph[u].bkb

        base_evaporation = self.evaporation_rate  # 基本揮発率（例：0.02）

        # 【ペナルティ判定】
        should_penalize = False

        # ペナルティは常に帯域のみで判定（どちらのフェロモン付加方法でも）
        if edge_bandwidth < bkb_u:
            should_penalize = True

        # 【揮発率の計算】
        if should_penalize:
            # ペナルティあり：揮発を促進（残存率を減少）
            # 例：残存率 0.98 * 0.5 = 0.49（51%消える）
            # → 揮発率に換算: 1 - 0.49 = 0.51（51%消える）
            evaporation = 1.0 - (1.0 - base_evaporation) * self.penalty_factor
        else:
            # ペナルティなし：通常の揮発率
            evaporation = base_evaporation

        # 【残存率計算】残存率 = 1 - 揮発率
        retention_rate = 1.0 - evaporation

        # 【フェロモン更新】現在のフェロモン量 × 残存率
        current = graph.graph.edges[u, v]["pheromone"]
        new_pheromone = math.floor(current * retention_rate)
        # 最小フェロモン量を保証
        new_pheromone = max(new_pheromone, graph.graph.edges[u, v]["min_pheromone"])
        graph.graph.edges[u, v]["pheromone"] = new_pheromone


class SimplePheromoneUpdater:
    """
    従来手法（Conventional Method）用のフェロモン更新クラス

    【従来手法のフェロモン更新】
    - ボトルネック帯域値に比例したフェロモンを単純に付加
    - 式: Δτ_ij^k = Q * B_k（B_kはボトルネック帯域）
    - 遅延制約が有効な場合: Δτ_ij^k = Q * (B_k / D_k)（B_kは帯域、D_kは遅延）
    - ノード学習機能（BKB/BLD/BKH）は使用しない
    - 功績ボーナスは使用しない（見つけた解の品質に関わらず同じ更新式）
    - 更新タイミング：アリがゴールに到達した時点で即座に付加（オンライン更新）
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        # Q: 調整係数（フェロモン付加量のスケーリング）
        self.q_factor = config["aco"].get("q_factor", 1.0)
        # 評価関数（遅延制約が有効な場合に使用）
        target_objectives = config["experiment"]["target_objectives"]
        from .evaluator import SolutionEvaluator

        self.evaluator = SolutionEvaluator(target_objectives)

    def update_from_ant(self, ant: Ant, graph: RoutingGraph) -> None:
        """
        アリがゴールに到達した際にフェロモンを更新（従来手法）

        帯域のみ最適化: Δτ_ij^k = Q * B_k （B_kはボトルネック帯域）
        遅延制約が有効な場合: Δτ_ij^k = Q * (B_k / D_k)（B_kは帯域、D_kは遅延）

        Args:
            ant: ゴールに到達したアリ
            graph: ルーティンググラフ
        """
        # アリの解を取得（ボトルネック帯域、累積遅延、ホップ数）
        bandwidth, delay, hops = ant.get_solution()

        # 評価関数を使用してスコアを計算
        # 帯域のみ最適化: score = bandwidth
        # 遅延制約が有効な場合: score = bandwidth / delay
        score = self.evaluator.evaluate(bandwidth, delay, hops)

        # フェロモン付加量 = Q * score
        delta_pheromone = self.q_factor * score

        # 経路上の各エッジにフェロモンを付加（双方向）
        route_edges = ant.get_route_edges()
        for u, v in route_edges:
            graph.update_pheromone(u, v, delta_pheromone, bidirectional=True)


class SimplePheromoneEvaporator:
    """
    従来手法（Conventional Method）用のフェロモン揮発クラス

    【従来手法のフェロモン揮発】
    - 単純な定率揮発のみを実行
    - 式: τ_ij(t+1) = (1 - ρ) * τ_ij(t)（ρは揮発率）
    - ペナルティ付き揮発は使用しない
    - BKBベースの揮発は使用しない
    - 全エッジで同じ揮発率を適用
    - 揮発タイミング：世代終了時に全エッジを揮発
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.evaporation_rate = config["aco"]["evaporation_rate"]

    def evaporate(self, graph: RoutingGraph) -> None:
        """
        フェロモンを揮発（従来手法）

        式: τ_ij(t+1) = (1 - ρ) * τ_ij(t)

        Args:
            graph: ルーティンググラフ
        """
        # 単純な定率揮発
        graph.evaporate_pheromone(self.evaporation_rate)


class PreviousMethodPheromoneUpdater:
    """
    先行研究（Previous Method）用のフェロモン更新クラス

    【先行研究のフェロモン更新】
    - エッジベースの学習：各エッジがw_ij^minとw_ij^maxを保持
    - フェロモン付加：Δτ_ij(B)は以下の式に従う
      - B^3 if w_ij^max ≤ B
      - B × 10 if w_ij^min < B < w_ij^max
      - B otherwise
    - エッジのlocal_min_bandwidthとlocal_max_bandwidthを更新
    - 更新タイミング：アリがゴールに到達した時点で即座に付加（オンライン更新）
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config

    def calculate_pheromone_increase(
        self,
        bottleneck_bandwidth: float,
        local_min_bandwidth: float,
        local_max_bandwidth: float,
    ) -> float:
        """
        フェロモン付加量を計算（先行研究の式）

        論文の式:
        Δτ_ij(B) = {
            B^3 if w_ij^max ≤ B,
            B × 10 if w_ij^min < B < w_ij^max,
            B otherwise
        }

        Args:
            bottleneck_bandwidth: ボトルネック帯域幅（B）
            local_min_bandwidth: エッジの最小帯域幅（w_ij^min）
            local_max_bandwidth: エッジの最大帯域幅（w_ij^max）

        Returns:
            フェロモン付加量
        """
        # 論文の式に従って計算
        if bottleneck_bandwidth >= local_max_bandwidth:
            # B >= w_ij^max: 最高品質のパス
            return float(bottleneck_bandwidth**3)
        elif local_min_bandwidth < bottleneck_bandwidth < local_max_bandwidth:
            # w_ij^min < B < w_ij^max: 中間品質のパス
            return float(bottleneck_bandwidth * 10)
        else:
            # B <= w_ij^min: 最低品質のパス
            return float(bottleneck_bandwidth)

    def update_from_ant(self, ant: Ant, graph: RoutingGraph) -> None:
        """
        アリがゴールに到達した際にフェロモンとエッジの学習値を更新（先行研究）

        Args:
            ant: ゴールに到達したアリ
            graph: ルーティンググラフ
        """
        # アリの解を取得（ボトルネック帯域、累積遅延、ホップ数）
        bandwidth, delay, hops = ant.get_solution()
        bottleneck_bandwidth = bandwidth

        # 経路上の各エッジを処理
        route_edges = ant.get_route_edges()
        for u, v in route_edges:
            # エッジの現在のlocal_min/max_bandwidthを取得
            # 初期化されていない場合は帯域幅で初期化
            edge_attr = graph.graph.edges[u, v]
            if "local_min_bandwidth" not in edge_attr:
                edge_attr["local_min_bandwidth"] = edge_attr["bandwidth"]
            if "local_max_bandwidth" not in edge_attr:
                edge_attr["local_max_bandwidth"] = edge_attr["bandwidth"]

            local_min_bandwidth = graph.graph.edges[u, v]["local_min_bandwidth"]
            local_max_bandwidth = graph.graph.edges[u, v]["local_max_bandwidth"]

            # エッジが知り得た最小帯域幅を更新
            # ボトルネック帯域幅（パス全体の最小値）を記録
            graph.graph.edges[u, v]["local_min_bandwidth"] = min(
                local_min_bandwidth,
                bottleneck_bandwidth,
            )

            # エッジが知り得た最大帯域幅を更新
            # パス上の最大帯域幅を記録（アリが通過したエッジの最大値）
            # アリのbandwidth_logリスト（各エッジの帯域幅）から最大値を取得
            if hasattr(ant, "bandwidth_log") and ant.bandwidth_log:
                max_path_bandwidth = max(ant.bandwidth_log)
            else:
                # フォールバック：このエッジの帯域幅を使用
                max_path_bandwidth = graph.graph.edges[u, v]["bandwidth"]
            graph.graph.edges[u, v]["local_max_bandwidth"] = max(
                local_max_bandwidth,
                max_path_bandwidth,
            )

            # 更新後の値を取得
            updated_local_min = graph.graph.edges[u, v]["local_min_bandwidth"]
            updated_local_max = graph.graph.edges[u, v]["local_max_bandwidth"]

            # フェロモン付加量を計算
            delta_pheromone = self.calculate_pheromone_increase(
                bottleneck_bandwidth,
                updated_local_min,
                updated_local_max,
            )

            # フェロモンを更新（双方向）
            current_pheromone = graph.graph.edges[u, v]["pheromone"]
            max_pheromone = graph.graph.edges[u, v].get("max_pheromone", float("inf"))
            new_pheromone = min(current_pheromone + delta_pheromone, max_pheromone)
            graph.graph.edges[u, v]["pheromone"] = new_pheromone

            # 双方向のエッジも更新
            if graph.graph.has_edge(v, u):
                current_pheromone_rev = graph.graph.edges[v, u]["pheromone"]
                max_pheromone_rev = graph.graph.edges[v, u].get(
                    "max_pheromone", float("inf")
                )
                new_pheromone_rev = min(
                    current_pheromone_rev + delta_pheromone, max_pheromone_rev
                )
                graph.graph.edges[v, u]["pheromone"] = new_pheromone_rev


class PreviousMethodPheromoneEvaporator:
    """
    先行研究（Previous Method）用のフェロモン揮発クラス

    【先行研究のフェロモン揮発】
    - エッジベースの適応的揮発
    - 式: rate_ij = V × (w_ij - w_ij^min) / max(1, w_ij^max - w_ij^min)
    - 帯域幅が大きいエッジほど多くのフェロモンを保持
    - 帯域幅が小さいエッジほど多くのフェロモンを揮発
    - 揮発タイミング：世代終了時に全エッジを揮発
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        # V: 最大保持率（論文の定数V、通常0.98）
        self.base_evaporation_rate = config["aco"]["evaporation_rate"]
        # V = 1 - evaporation_rate（残存率）
        self.v_constant = 1.0 - self.base_evaporation_rate

    def evaporate(self, graph: RoutingGraph) -> None:
        """
        フェロモンを揮発（先行研究）

        式: rate_ij = V × (w_ij - w_ij^min) / max(1, w_ij^max - w_ij^min)
        τ_ij(t+1) = τ_ij(t) × rate_ij

        Args:
            graph: ルーティンググラフ
        """
        for u, v in graph.graph.edges():
            # エッジの属性を取得
            current_pheromone = graph.graph.edges[u, v]["pheromone"]
            weight_uv = graph.graph.edges[u, v]["bandwidth"]

            # エッジのローカル最小・最大帯域幅を取得
            # 初期化されていない場合は帯域幅で初期化
            if "local_min_bandwidth" not in graph.graph.edges[u, v]:
                graph.graph.edges[u, v]["local_min_bandwidth"] = weight_uv
            if "local_max_bandwidth" not in graph.graph.edges[u, v]:
                graph.graph.edges[u, v]["local_max_bandwidth"] = weight_uv

            local_min_bandwidth = graph.graph.edges[u, v]["local_min_bandwidth"]
            local_max_bandwidth = graph.graph.edges[u, v]["local_max_bandwidth"]

            # 揮発率を計算（論文の式）
            # rate_ij = V × (w_ij - w_ij^min) / max(1, w_ij^max - w_ij^min)
            denominator = max(1.0, local_max_bandwidth - local_min_bandwidth)
            numerator = max(0.0, weight_uv - local_min_bandwidth)  # 負の値を防ぐ
            rate = self.v_constant * numerator / denominator

            # フェロモン値を更新
            min_pheromone = graph.graph.edges[u, v].get("min_pheromone", 0)
            new_pheromone = max(
                current_pheromone * rate,
                min_pheromone,
            )
            graph.graph.edges[u, v]["pheromone"] = new_pheromone
