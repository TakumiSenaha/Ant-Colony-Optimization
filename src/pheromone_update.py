"""
フェロモン更新・揮発処理の共通モジュール

フェロモンの付加、揮発、ボーナス計算を一元管理します。
各ACO実装からパラメータを引数で注入して使用します。
"""

import math
from typing import Callable, Optional

import networkx as nx  # type: ignore[import-untyped]


def calculate_pheromone_increase_simple(bottleneck_bandwidth: int) -> float:
    """
    シンプルなフェロモン付加量計算

    Args:
        bottleneck_bandwidth: ボトルネック帯域幅

    Returns:
        フェロモン付加量
    """
    return float(bottleneck_bandwidth * 10)


def calculate_pheromone_increase_statistical(
    bottleneck_bandwidth: int,
    node_mean: float,
    node_var: float,
    achievement_bonus_func: Callable[[float, float, float], float],
) -> float:
    """
    統計的BKB学習を考慮したフェロモン付加量計算

    Args:
        bottleneck_bandwidth: ボトルネック帯域幅
        node_mean: ノードの平均BKB
        node_var: ノードのBKB分散
        achievement_bonus_func: 功績ボーナス計算関数

    Returns:
        フェロモン付加量
    """
    # 基本のフェロモン付加量
    base_increase = float(bottleneck_bandwidth * 10)

    # ★★★ 変動学習による動的調整 ★★★
    if node_mean > 0 and node_var > 0:
        # 変動係数（CV: Coefficient of Variation）を計算
        cv = math.sqrt(node_var) / node_mean

        # 変動が大きい（動的環境）→ より積極的な報酬
        # 変動が小さい（静的環境）→ 控えめな報酬
        if cv > 0.3:  # 高変動環境
            dynamic_multiplier = 1.5  # 50%増加
        elif cv > 0.1:  # 中変動環境
            dynamic_multiplier = 1.2  # 20%増加
        else:  # 低変動環境
            dynamic_multiplier = 1.0  # そのまま

        # 功績ボーナスも適用
        achievement_bonus = achievement_bonus_func(
            float(bottleneck_bandwidth), node_mean, node_var
        )

        return base_increase * dynamic_multiplier * achievement_bonus
    else:
        # 学習初期段階は基本値
        return base_increase


def apply_volatilization(  # noqa: C901
    graph: nx.Graph,
    u: int,
    v: int,
    volatilization_mode: int,
    base_evaporation_rate: float,
    penalty_factor: float,
    adaptive_rate_func: Optional[Callable[[nx.Graph, int, int], float]] = None,
) -> None:
    """
    指定された方向のエッジ (u → v) に対して揮発処理を適用

    揮発率は以下の3つの要因で決まります：
    1. 基本揮発率（base_evaporation_rate）：世代による一定割合の揮発
    2. BKBベースのペナルティ（penalty_factor）：BKBを下回るエッジへのペナルティ
    3. 帯域変動パターンに基づく適応的揮発（adaptive_rate_func）：エッジの可用帯域変動に応じた調整

    Args:
        graph: ネットワークグラフ
        u: 始点ノード
        v: 終点ノード
        volatilization_mode: 揮発モード (0, 1, 2, 3)
        base_evaporation_rate: 基本揮発率（残存率、世代による一定割合）
        penalty_factor: ペナルティ係数（BKBを下回るエッジへのペナルティ）
        adaptive_rate_func: 帯域変動パターンに基づく適応的揮発率調整関数
                            (graph, u, v) -> float (乗算係数)
                            例：sin関数のような周期的変動を検出して揮発を調整
                            None の場合は適用しない
    """
    # 現在のフェロモン値と帯域幅を取得
    current_pheromone = graph[u][v]["pheromone"]
    weight_uv = graph[u][v]["weight"]

    # エッジのローカル最小・最大帯域幅を取得
    local_min_bandwidth = graph[u][v]["local_min_bandwidth"]
    local_max_bandwidth = graph[u][v]["local_max_bandwidth"]

    # 揮発率の計算
    if volatilization_mode == 0:
        # 固定の揮発率
        rate = base_evaporation_rate

    elif volatilization_mode == 1:
        # 帯域幅の最小値・最大値を基準に揮発量を調整
        if local_max_bandwidth == local_min_bandwidth:
            rate = 0.98
        else:
            normalized_position = (weight_uv - local_min_bandwidth) / max(
                1, (local_max_bandwidth - local_min_bandwidth)
            )
            rate = 0.98 * normalized_position

    elif volatilization_mode == 2:
        # 平均・分散を基準に揮発量を調整
        if local_max_bandwidth == local_min_bandwidth:
            avg_bandwidth = weight_uv
            std_dev = 1
        else:
            avg_bandwidth = 0.5 * (local_min_bandwidth + local_max_bandwidth)
            std_dev = max(abs(local_max_bandwidth - avg_bandwidth), 1)

        gamma = 1.0
        rate = math.exp(-gamma * (avg_bandwidth - weight_uv) / std_dev)

    elif volatilization_mode == 3:
        # ノードのBKBに基づきペナルティを適用
        rate = base_evaporation_rate

        # === BKBベースのペナルティ（既存機能）===
        # 行き先ノードvが知っている最良のボトルネック帯域(BKB)を取得
        bkb_v = graph.nodes[v].get("best_known_bottleneck", 0)

        # このエッジの帯域幅が、行き先ノードのBKBより低い場合、ペナルティを課す
        if weight_uv < bkb_v:
            rate *= penalty_factor  # 残存率を下げることで、揮発を促進する

        # === 帯域変動パターンに基づく適応的揮発調整（新機能）===
        # エッジの可用帯域の変動パターン（例：sin関数のような周期的変動）を検出して、
        # それに応じて揮発率を調整
        if adaptive_rate_func is not None:
            adaptive_multiplier = adaptive_rate_func(graph, u, v)
            rate *= adaptive_multiplier
            # 例：周期的変動を検出した場合、次の低帯域時期を予測して
            # 揮発を促進する（adaptive_multiplier < 1.0）
            # または、安定している場合は揮発を抑制する（adaptive_multiplier > 1.0）

    else:
        raise ValueError("Invalid VOLATILIZATION_MODE. Choose 0, 1, 2 or 3.")

    # フェロモン値を計算して更新
    new_pheromone = max(
        math.floor(current_pheromone * rate), graph[u][v]["min_pheromone"]
    )
    graph[u][v]["pheromone"] = new_pheromone


def volatilize_by_width(
    graph: nx.Graph,
    volatilization_mode: int,
    base_evaporation_rate: float,
    penalty_factor: float,
    adaptive_rate_func: Optional[Callable[[nx.Graph, int, int], float]] = None,
) -> None:
    """
    各エッジのフェロモン値を双方向で揮発させる

    Args:
        graph: ネットワークグラフ
        volatilization_mode: 揮発モード (0, 1, 2, 3)
        base_evaporation_rate: 基本揮発率（残存率、世代による一定割合）
        penalty_factor: ペナルティ係数（BKBを下回るエッジへのペナルティ）
        adaptive_rate_func: 帯域変動パターンに基づく適応的揮発率調整関数
                            (graph, u, v) -> float (乗算係数)
                            None の場合は適用しない
    """
    for u, v in graph.edges():
        # u → v の揮発計算
        apply_volatilization(
            graph,
            u,
            v,
            volatilization_mode,
            base_evaporation_rate,
            penalty_factor,
            adaptive_rate_func,
        )
        # v → u の揮発計算
        apply_volatilization(
            graph,
            v,
            u,
            volatilization_mode,
            base_evaporation_rate,
            penalty_factor,
            adaptive_rate_func,
        )


def update_pheromone(
    ant,
    graph: nx.Graph,
    generation: int,
    max_pheromone: float,
    achievement_bonus: float,
    bkb_update_func: Callable[[nx.Graph, int, float, int], None],
    pheromone_increase_func: Optional[Callable[[int, float, float], float]] = None,
    observe_bandwidth_func: Optional[
        Callable[[nx.Graph, int, int, float], None]
    ] = None,
) -> None:
    """
    Antがゴールに到達したとき、経路上のフェロモンとノードのBKBを更新する
    ★★★ フェロモンは経路上のエッジに「双方向」で付加する ★★★

    Args:
        ant: Antオブジェクト
        graph: ネットワークグラフ
        generation: 現在の世代番号
        max_pheromone: フェロモンの最大値
        achievement_bonus: 功績ボーナス係数
        bkb_update_func: BKB更新関数 (graph, node, bottleneck, generation) -> None
        pheromone_increase_func: フェロモン増加量計算関数（統計的BKB学習用）
                                  Noneの場合はシンプル版を使用
        observe_bandwidth_func: エッジ帯域観測関数（帯域監視用）
                                (graph, u, v, bandwidth) -> None
                                Noneの場合は観測しない
    """
    bottleneck_bn = min(ant.width) if ant.width else 0
    if bottleneck_bn == 0:
        return

    # --- 経路上の各エッジにフェロモンを付加（BKB更新の前）---
    for i in range(1, len(ant.route)):
        u, v = ant.route[i - 1], ant.route[i]

        # === 帯域観測（帯域変動パターン学習のため）===
        # アリがエッジを通過したときに、そのエッジの帯域を観測して記録
        if observe_bandwidth_func is not None:
            # エッジの帯域幅を取得（ant.widthには各エッジの帯域が記録されている）
            if i > 0 and len(ant.width) > i - 1:
                edge_bandwidth = ant.width[i - 1]
            else:
                edge_bandwidth = graph[u][v]["weight"]
            observe_bandwidth_func(graph, u, v, float(edge_bandwidth))

        # フェロモン増加量を計算
        if pheromone_increase_func is not None:
            # 統計的BKB学習の場合
            node_mean = graph.nodes[v].get("ema_bkb", 0.0)
            node_var = graph.nodes[v].get("ema_bkb_var", 0.0)
            pheromone_increase = pheromone_increase_func(
                bottleneck_bn, node_mean, node_var
            )
        else:
            # シンプル版
            pheromone_increase = calculate_pheromone_increase_simple(bottleneck_bn)

            # 功績ボーナスの判定（シンプル版）
            current_bkb_v = graph.nodes[v].get("best_known_bottleneck", 0)
            if bottleneck_bn > current_bkb_v:
                pheromone_increase *= achievement_bonus

        # ===== ★★★ フェロモンを双方向に付加 ★★★ =====
        # 順方向 (u -> v) のフェロモンを更新
        max_pheromone_uv = graph.edges[u, v].get("max_pheromone", max_pheromone)
        graph.edges[u, v]["pheromone"] = min(
            graph.edges[u, v]["pheromone"] + pheromone_increase,
            max_pheromone_uv,
        )

        # 逆方向 (v -> u) のフェロモンも更新
        max_pheromone_vu = graph.edges[v, u].get("max_pheromone", max_pheromone)
        graph.edges[v, u]["pheromone"] = min(
            graph.edges[v, u]["pheromone"] + pheromone_increase,
            max_pheromone_vu,
        )
        # =======================================================

    # --- BKBの更新（フェロモン付加の後に行う）---
    # 経路上の各ノードのBKBを更新（コールバック関数を使用）
    for node in ant.route:
        bkb_update_func(graph, node, float(bottleneck_bn), generation)


def calculate_current_optimal_bottleneck(
    graph: nx.Graph, start_node: int, goal_node: int
) -> int:
    """
    現在のネットワーク状態での最適ボトルネック帯域を計算

    Args:
        graph: ネットワークグラフ
        start_node: 開始ノード
        goal_node: 終了ノード

    Returns:
        最適ボトルネック帯域（経路なしの場合は0）
    """
    try:
        from modified_dijkstra import max_load_path

        optimal_path = max_load_path(graph, start_node, goal_node)
        optimal_bottleneck = min(
            graph.edges[u, v]["weight"]
            for u, v in zip(optimal_path[:-1], optimal_path[1:])
        )
        return optimal_bottleneck
    except Exception:
        return 0
