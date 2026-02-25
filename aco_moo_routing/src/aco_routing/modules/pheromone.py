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

        # 【重要】既存実装との互換性のため、帯域を整数として扱う
        # 既存実装（aco_main_bkb_available_bandwidth.py）では、bottleneck_bn = min(ant.width)がint
        # 新実装でも同じように扱うため、intに変換してから使用
        bandwidth_int = int(bandwidth)  # 帯域は10Mbps刻みなので整数として扱う

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
            # 【重要】既存実装との互換性のため、整数として渡す
            graph[node].update_bandwidth(float(bandwidth_int))

        # 【Step 2】経路上の各エッジにフェロモンを付加（帰還時の処理）
        # アリはゴールからスタートへ戻りながら、各エッジにフェロモンを付加
        route_edges = ant.get_route_edges()

        for i, (u, v) in enumerate(route_edges):
            # 【基本フェロモン付加量】f(B) = 10 * B
            # 【遅延制約が有効な場合】帯域/遅延のスコアを使用
            # Δτ = 10 * (B / D_path)
            # 【重要】既存実装との互換性のため、整数として扱う
            if delay_constraint_enabled:
                if delay > 0:
                    base_pheromone = 10.0 * (float(bandwidth_int) / delay)
                else:
                    base_pheromone = float(bandwidth_int) * 10.0
            else:
                base_pheromone = float(bandwidth_int) * 10.0

            # 【功績ボーナス判定】分散判断：各ノードが独立に判定
            # エッジ(u, v)の処理 = 帰還時にノードvからノードuへ戻る処理に対応
            # この時点で、アリはノードvの記憶値（更新前）を知っている
            k_v, l_v, m_v = node_old_memory[v]

            # 【功績ボーナス適用】B >= K_v の場合、ボーナス係数 B_a を適用
            # 【重要】既存実装との互換性のため、整数として比較
            if bandwidth_int >= int(k_v):
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
        # 【重要】有向グラフ対応:
        # G.edges()には(u, v)と(v, u)の両方が含まれるため、
        # 各エッジは1回だけ処理する（双方向処理は不要）
        for u, v in graph.graph.edges():
            # 【エッジ(u → v)の揮発計算】
            # ノードuから出ていくパスとして、ノードuのBKBと比較
            self._apply_evaporation_to_edge(graph, u, v)

    def _apply_evaporation_to_edge(self, graph: RoutingGraph, u: int, v: int) -> None:
        """
        単一方向のエッジ (u → v) に対して揮発処理を適用

        【重要原則】ペナルティは「自分から出ていくパス」にのみ適用
        - エッジ(u → v)は「ノードuから出ていくパス」
        - ノードuのBKBと比較してペナルティを判定
        - 「自分を通ってこの先、この品質でゴールできる」という情報に基づく

        【既存実装との互換性】
        既存実装（src/pheromone_update.py）では、base_evaporation_rateが残存率（0.98）として
        直接使用されています。新実装でも同じように、残存率として扱います。

        Args:
            graph: ルーティンググラフ
            u: 始点ノード（このノードから出ていくパス）
            v: 終点ノード
        """
        # エッジの属性を取得
        # 【既存実装との互換性】既存実装ではweight属性を使用
        # 新実装ではbandwidth属性を使用（同じものを指す）
        edge_bandwidth = graph.graph.edges[u, v]["bandwidth"]

        # ノードuの学習値（BKB）：ノードuから出ていくパスなので、ノードuのBKBと比較
        # 【既存実装との互換性】既存実装ではint型として扱う
        bkb_u = graph[u].bkb

        # 【既存実装との互換性】残存率として扱う
        # 既存実装では base_evaporation_rate = V = 0.98（残存率）として使用
        # 新実装では evaporation_rate = 0.02（揮発率）として設定されているため、
        # 残存率 = 1 - 揮発率 = 1 - 0.02 = 0.98 として計算
        retention_rate = 1.0 - self.evaporation_rate  # 残存率（例：0.98）

        # 【ペナルティ判定】
        # ペナルティは常に帯域のみで判定（どちらのフェロモン付加方法でも）
        # 【既存実装との互換性】既存実装ではint型として比較
        # 既存実装: if weight_uv < bkb_u:
        if int(edge_bandwidth) < int(bkb_u):
            # ペナルティあり：残存率を下げる（既存実装と同じ）
            # 既存実装: rate *= penalty_factor → 0.98 * 0.5 = 0.49
            retention_rate *= self.penalty_factor

        # 【フェロモン更新】現在のフェロモン量 × 残存率（既存実装と同じ）
        current = graph.graph.edges[u, v]["pheromone"]
        new_pheromone = math.floor(current * retention_rate)
        # 最小フェロモン量を保証
        new_pheromone = max(new_pheromone, graph.graph.edges[u, v]["min_pheromone"])
        graph.graph.edges[u, v]["pheromone"] = new_pheromone


class SimplePheromoneUpdater:
    """
    フェロモン更新クラス（Ant System流）

    注意: このクラスは提案手法や先行研究手法で使用されます。
          純粋なACS（1997年論文準拠）では使用しません。

    【使用される手法】
    - 提案手法（Proposed Method）: オンライン更新
    - 先行研究手法（Previous Method）: オンライン更新

    【ACS論文準拠実装との違い】
    - ACS: Global Bestのエッジのみに τ_ij ← (1-ρ)τ_ij + ρΔτ を適用
    - このクラス: 全てのアリがゴール到達時に即座にフェロモンを付加

    【大域更新規則（このクラスの仕様）】
    - タイミング: アリがゴールに到達した時点で即座に
    - 対象: そのアリの経路上の全エッジ
    - 報酬 Δτ: ボトルネック帯域の正規化値（0.1〜1.0）
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        # 正規化定数: 帯域幅を0〜1に正規化するための除数
        self.bandwidth_normalization = config["aco"].get(
            "bandwidth_normalization", 100.0
        )
        # 大域学習率 ρ（論文標準値: 0.1）
        self.rho = config["aco"]["evaporation_rate"]

    def update_from_ant(self, ant: Ant, graph: RoutingGraph) -> None:
        """
        Global Bestアリの経路にフェロモンを更新（ACS方式）

        論文の式: τ_ij ← (1-ρ)τ_ij + ρΔτ_ij
        ここで、Δτ_ij = B_gb^norm（Global Bestのボトルネック帯域、正規化後）

        MBL問題の解釈:
        - ボトルネック帯域が広いほどΔτが大きい（TSPの「短い経路ほど良い」に対応）
        - 正規化により、Δτは0.1〜1.0の範囲に収まる
        - ρ=0.1により、フェロモンは徐々に増加（爆発的な増加を防ぐ）

        Args:
            ant: Global Bestアリ（最良解を持つアリ）
            graph: ルーティンググラフ
        """
        # アリの解を取得（ボトルネック帯域、累積遅延、ホップ数）
        bandwidth, delay, hops = ant.get_solution()

        # 【正規化】ボトルネック帯域を0〜1の範囲に変換
        # η_ij = B_ij / C_norm と同じ正規化を適用
        bandwidth_normalized = bandwidth / self.bandwidth_normalization

        # 【報酬計算】Δτ = B_gb^norm
        # TSPの Δτ = 1/L_gb（短い経路ほど大きい報酬）に対応
        # MBLでは Δτ = B_gb^norm（広い帯域ほど大きい報酬）
        delta_tau = bandwidth_normalized

        # 【大域更新】Global Bestの経路上の各エッジに対して更新
        # 論文の式: τ_ij ← (1-ρ)τ_ij + ρΔτ_ij
        route_edges = ant.get_route_edges()
        for u, v in route_edges:
            # 現在のフェロモン値を取得
            edge_attr = graph.get_edge_attributes(u, v)
            current_tau = edge_attr["pheromone"]

            # 新しいフェロモン値を計算
            # τ_new = (1-ρ)τ_old + ρΔτ
            new_tau = (1 - self.rho) * current_tau + self.rho * delta_tau

            # フェロモンの差分を計算して更新
            delta_pheromone = new_tau - current_tau
            graph.update_pheromone(u, v, delta_pheromone, bidirectional=True)


class SimplePheromoneEvaporator:
    """
    フェロモン揮発クラス（Ant System流）

    注意: このクラスは提案手法や先行研究手法で使用されます。
          純粋なACS（1997年論文準拠）では使用しません。

    【使用される手法】
    - 提案手法（Proposed Method）: 世代終了時の全エッジ揮発
    - 先行研究手法（Previous Method）: 世代終了時の全エッジ揮発

    【ACS論文準拠実装との違い】
    - ACS: Global Bestのエッジのみ揮発（τ_ij ← (1-ρ)τ_ij + ρΔτ の一部）
    - このクラス: 全エッジを無差別に揮発

    【揮発規則（このクラスの仕様）】
    - タイミング: 世代終了時
    - 対象: 全てのエッジ（無差別）
    - 式: τ_ij ← (1-ρ)τ_ij
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        # ρ: 大域学習率（論文標準値: 0.1）
        self.evaporation_rate = config["aco"]["evaporation_rate"]

    def evaporate(self, graph: RoutingGraph) -> None:
        """
        フェロモンを揮発（ACS方式）

        論文の式: τ_ij ← (1-ρ)τ_ij

        MBL問題での意味:
        - 過去の経験（フェロモン）を徐々に忘却する
        - ρ=0.1なので、毎世代10%が揮発し、90%が残存
        - これにより、古い情報の影響を抑え、環境変化に対応可能

        Args:
            graph: ルーティンググラフ
        """
        # 全エッジに対して定率揮発を適用
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
