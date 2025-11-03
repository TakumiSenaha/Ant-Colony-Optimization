import csv
import random

import networkx as nx  # type: ignore[import-untyped]

from bandwidth_fluctuation_config import (
    BANDWIDTH_UPDATE_INTERVAL,
    initialize_ar1_states,
    print_fluctuation_settings,
    select_fluctuating_edges,
    update_available_bandwidth_ar1,
)
from bkb_learning import (
    BKBLearningConfig,
    calculate_confidence,
    evaporate_bkb_values,  # ★BKB揮発処理を追加★
    initialize_graph_nodes_for_bkb,
    update_node_bkb_multi_scale_max,  # ★複数スケール学習を追加★
    update_node_bkb_statistics,
    update_node_bkb_three_phase,  # ★三段階学習を追加★
    update_node_bkb_time_window_max,  # ★時間区間ベース学習を追加★
)
from modified_dijkstra import max_load_path
from pheromone_update import (
    calculate_current_optimal_bottleneck,
    update_pheromone,
    volatilize_by_width,
)

# ===== シミュレーションパラメータ =====
V = 0.98  # フェロモン揮発量
MIN_F = 100  # フェロモン最小値
MAX_F = 1000000000  # フェロモン最大値
TTL = 100  # AntのTime to Live

# ===== ACOパラメータ =====
ALPHA = 1.0  # フェロモンの影響度
BETA = 1.0  # ヒューリスティック情報(帯域幅)の影響度
EPSILON = 0.1  # ランダムに行動する固定確率
ANT_NUM = 10  # 世代ごとに探索するアリの数
GENERATION = 1000  # 総世代数
SIMULATIONS = 100  # シミュレーションの試行回数

# ===== BKB統計モデル（RFC 6298 準拠）=====
# 【複数の学習速度設定】
# 動的環境の変動速度に応じて選択

# --- 標準設定（RFC 6298準拠）---
BKB_CONFIG_STANDARD = BKBLearningConfig(
    mean_alpha=1 / 8,  # 標準 SRTT 学習率 (0.125)
    var_beta=1 / 4,  # 標準 RTTVAR 学習率 (0.25)
    confidence_k=1.0,  # 信頼区間幅の係数
    achievement_bonus_base=1.5,  # シンプルな固定ボーナス係数
    achievement_bonus_max=3.0,  # （未使用）
    confidence_scaling=2.0,  # （未使用）
    penalty_factor=0.5,  # ペナルティ係数
    use_confidence_based_bonus=False,
)

# --- ★高速学習設定（標準の2倍速）★ ---
BKB_CONFIG_FAST = BKBLearningConfig(
    mean_alpha=1 / 4,  # 2倍速 SRTT 学習率 (0.25)
    var_beta=1 / 2,  # 2倍速 RTTVAR 学習率 (0.5)
    confidence_k=1.0,  # （変更なし）
    achievement_bonus_base=1.5,  # （変更なし）
    achievement_bonus_max=3.0,  # （未使用）
    confidence_scaling=2.0,  # （未使用）
    penalty_factor=0.5,  # （変更なし）
    use_confidence_based_bonus=False,
)

# --- 超高速学習設定（標準の4倍速）---
BKB_CONFIG_VERY_FAST = BKBLearningConfig(
    mean_alpha=1 / 2,  # 4倍速 SRTT 学習率 (0.5)
    var_beta=3 / 4,  # 4倍速 RTTVAR 学習率 (0.75)
    confidence_k=1.0,  # （変更なし）
    achievement_bonus_base=2.0,  # ★より積極的なボーナス★
    achievement_bonus_max=3.0,  # （未使用）
    confidence_scaling=2.0,  # （未使用）
    penalty_factor=0.3,  # ★より厳しいペナルティ★
    use_confidence_based_bonus=False,
)

# --- 即時追従設定（8倍速：ほぼ最新値を使用）---
BKB_CONFIG_INSTANT = BKBLearningConfig(
    mean_alpha=1.0,  # 即時追従（完全に最新値）
    var_beta=1.0,  # 即時追従（完全に最新値）
    confidence_k=1.0,  # （変更なし）
    achievement_bonus_base=1.5,  # （変更なし）
    achievement_bonus_max=3.0,  # （未使用）
    confidence_scaling=2.0,  # （未使用）
    penalty_factor=0.5,  # （変更なし）
    use_confidence_based_bonus=False,
)

# ===== 🎯 使用する設定を選択 =====
# BKB_CONFIG = BKB_CONFIG_FAST  # 高速学習設定（2倍速）
# BKB_CONFIG = BKB_CONFIG_STANDARD  # 標準設定
BKB_CONFIG = BKB_CONFIG_VERY_FAST  # ★超高速設定（高変動環境向け）★
# BKB_CONFIG = BKB_CONFIG_INSTANT  # 即時追従（実験用）

# ===== 学習手法の選択 =====
USE_THREE_PHASE_LEARNING = False  # 三段階学習（超短期+短期+長期）
USE_TWO_PHASE_LEARNING = False  # 二段階学習（短期+長期）
USE_TIME_WINDOW_LEARNING = True  # ★時間区間ベース学習を使用★
USE_MULTI_SCALE_LEARNING = False  # 複数スケール学習（短期+中期+長期）
# USE_TWO_PHASE_LEARNING = False  # 従来の単一EMA学習

# ===== リングバッファサイズ設定 =====
TIME_WINDOW_SIZE = 1000  # リングバッファサイズ（記憶する観測値の数）

# 後方互換性のため、個別定数も保持
BKB_CONFIDENCE_K = BKB_CONFIG.confidence_k

# ===== 時間窓学習用のパラメータ（既存のmax手法と同じ）=====
BKB_EVAPORATION_RATE = 0.999  # BKB値の揮発率（既存のmax手法と同じ）
ACHIEVEMENT_BONUS = 1.5  # BKBを更新した場合の報酬ボーナス係数（既存のmax手法と同じ）
PENALTY_FACTOR = (
    0.5 if USE_TIME_WINDOW_LEARNING else BKB_CONFIG.penalty_factor
)  # 時間窓学習の場合は既存のmax手法と同じ

# ===== 動的帯域変動パラメータ（AR(1)モデル） =====
# 帯域変動パラメータは bandwidth_fluctuation_config.py で管理


class Ant:
    def __init__(
        self, current: int, destination: int, route: list[int], width: list[int]
    ):
        self.current = current
        self.destination = destination
        self.route = route
        self.width = width

    def __repr__(self):
        return (
            f"Ant(current={self.current}, destination={self.destination}, "
            f"route={self.route}, width={self.width})"
        )


def set_pheromone_min_max_by_degree_and_width(graph: nx.Graph) -> None:
    """
    ノードの隣接数と帯域幅に基づいて
    フェロモンの最小値と最大値を双方向に設定
    """
    for u, v in graph.edges():
        # ノードuとvの隣接ノード数を取得
        degree_u = len(list(graph.neighbors(u)))
        degree_v = len(list(graph.neighbors(v)))

        # フェロモン最小値を隣接ノード数に基づいて設定
        graph[u][v]["min_pheromone"] = MIN_F * 3 // degree_u
        graph[v][u]["min_pheromone"] = MIN_F * 3 // degree_v

        # 帯域幅に基づいてフェロモン最大値を設定
        width_u_to_v = graph[u][v]["weight"]
        width_v_to_u = graph[v][u]["weight"]

        graph[u][v]["max_pheromone"] = width_u_to_v**5
        graph[v][u]["max_pheromone"] = width_v_to_u**5


VOLATILIZATION_MODE = 3


# BKB更新関数のラッパー（複数の学習モードに対応）
def _create_bkb_update_func():
    """BKB更新関数を作成（学習モードに応じて分岐）"""

    def bkb_update_func(
        graph: nx.Graph, node: int, bottleneck: float, generation: int
    ) -> None:
        if USE_TIME_WINDOW_LEARNING:
            update_node_bkb_time_window_max(
                graph, node, bottleneck, generation, time_window_size=TIME_WINDOW_SIZE
            )
        elif USE_MULTI_SCALE_LEARNING:
            update_node_bkb_multi_scale_max(
                graph,
                node,
                bottleneck,
                short_window=5,
                medium_window=20,
                long_window=100,
                short_alpha=0.7,
                medium_alpha=0.3,
                long_alpha=0.1,
            )
        elif USE_THREE_PHASE_LEARNING:
            update_node_bkb_three_phase(graph, node, bottleneck, BKB_CONFIG)
        elif USE_TWO_PHASE_LEARNING:
            from bkb_learning import update_node_bkb_two_phase

            update_node_bkb_two_phase(graph, node, bottleneck, BKB_CONFIG)
        else:
            # 従来の単一EMA学習（RFC 6298準拠）
            mean_prev = graph.nodes[node].get("ema_bkb")
            if mean_prev is None:
                graph.nodes[node]["ema_bkb"] = float(bottleneck)
                graph.nodes[node]["ema_bkb_var"] = float(bottleneck) / 2.0
            else:
                update_node_bkb_statistics(graph, node, float(bottleneck), BKB_CONFIG)

    return bkb_update_func


# ===== 定数ε-Greedy法 =====
def ant_next_node_const_epsilon(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    current_optimal_bottleneck: int,
    generation: int,
) -> None:
    """
    固定パラメータ(α, β, ε)を用いた、最もシンプルなε-Greedy法で次のノードを決定する。
    """
    for ant in reversed(ant_list):
        neighbors = list(graph.neighbors(ant.current))
        candidates = [n for n in neighbors if n not in ant.route]

        if not candidates:
            ant_list.remove(ant)
            ant_log.append(0)
            continue  # 次のアリの処理へ

        # ===== 定数ε-Greedy選択 =====
        if random.random() < EPSILON:
            # 【探索】εの確率で、重みを無視してランダムに次ノードを選択
            next_node = random.choice(candidates)
        else:
            # 【活用】1-εの確率で、フェロモンと帯域幅に基づいて次ノードを選択
            pheromones = [graph[ant.current][n]["pheromone"] for n in candidates]
            widths = [graph[ant.current][n]["weight"] for n in candidates]

            # αとβは固定値を使用
            weight_pheromone = [p**ALPHA for p in pheromones]
            weight_width = [w**BETA for w in widths]
            weights = [p * w for p, w in zip(weight_pheromone, weight_width)]

            # 重みが全て0の場合や候補がない場合のフォールバック
            if not weights or sum(weights) == 0:
                next_node = random.choice(candidates)
            else:
                next_node = random.choices(candidates, weights=weights, k=1)[0]
        # =======================

        # --- antの状態更新 ---
        next_edge_bandwidth = graph[ant.current][next_node]["weight"]
        ant.route.append(next_node)
        ant.width.append(next_edge_bandwidth)
        ant.current = next_node

        # --- ゴール判定 ---
        if ant.current == ant.destination:
            # ★★★ 共通モジュールを使用したフェロモン更新 ★★★
            update_pheromone(
                ant,
                graph,
                generation,
                max_pheromone=MAX_F,
                achievement_bonus=ACHIEVEMENT_BONUS,
                bkb_update_func=_create_bkb_update_func(),
                pheromone_increase_func=None,  # シンプル版を使用
                observe_bandwidth_func=None,  # 帯域監視は未使用
            )
            ant_log.append(1 if min(ant.width) >= current_optimal_bottleneck else 0)
            ant_list.remove(ant)
        elif len(ant.route) >= TTL:
            ant_log.append(0)
            ant_list.remove(ant)


def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """
    Barabási-Albertモデルでグラフを生成
    - 各ノードにBKB統計属性（平均・分散）を初期化
    - 各エッジに帯域幅(weight)等を初期化
    """
    graph = nx.barabasi_albert_graph(num_nodes, num_edges)

    # ===== BKB統計モデル用の属性を初期化 =====
    if USE_TIME_WINDOW_LEARNING:
        # 時間窓学習の場合：既存のmax手法と同じ初期化
        for node in graph.nodes():
            graph.nodes[node]["best_known_bottleneck"] = 0
    else:
        # 統計的BKB学習の場合：共通モジュール使用
        initialize_graph_nodes_for_bkb(graph)
    # =======================================================================

    for u, v in graph.edges():
        # リンクの帯域幅(weight)をランダムに設定
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight

        # 初期帯域幅を保存（変動の基準値として使用）
        graph[u][v]["original_weight"] = weight

        # NOTE: local_min/max_bandwidth は新しいアプローチでは使わなくなりますが、
        #       段階的な移行のため一旦残します。
        graph[u][v]["local_min_bandwidth"] = weight
        graph[u][v]["local_max_bandwidth"] = weight

        # フェロモン値を初期化
        graph[u][v]["pheromone"] = MIN_F
        graph[u][v]["max_pheromone"] = MAX_F
        graph[u][v]["min_pheromone"] = MIN_F

    return graph


def er_graph(
    num_nodes: int, edge_prob: float = 0.12, lb: int = 1, ub: int = 10
) -> nx.Graph:
    """
    Erdős–Rényi (ER)モデルでランダムグラフを生成
    - 各ノードにBKB統計属性（平均・分散）を初期化
    - 各エッジに帯域幅(weight)等を初期化
    edge_probは、BAモデルと同程度のエッジ数になるように調整してください。
    """
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)

    # BKB統計モデル用の属性を初期化
    if USE_TIME_WINDOW_LEARNING:
        # 時間窓学習の場合：既存のmax手法と同じ初期化
        for node in graph.nodes():
            graph.nodes[node]["best_known_bottleneck"] = 0
    else:
        # 統計的BKB学習の場合：共通モジュール使用
        initialize_graph_nodes_for_bkb(graph)

    for u, v in graph.edges():
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight
        graph[u][v]["local_min_bandwidth"] = weight
        graph[u][v]["local_max_bandwidth"] = weight
        graph[u][v]["pheromone"] = MIN_F
        graph[u][v]["max_pheromone"] = MAX_F
        graph[u][v]["min_pheromone"] = MIN_F

    return graph


def grid_graph(num_nodes: int, lb: int = 1, ub: int = 10) -> nx.Graph:
    """
    グリッド（格子）ネットワークを生成
    - num_nodesが平方数の場合のみ対応（例: 49, 100）
    - 各ノードにBKB統計属性（平均・分散）を初期化
    - 各エッジに帯域幅(weight)等を初期化
    """
    import math

    side = int(math.sqrt(num_nodes))
    if side * side != num_nodes:
        raise ValueError("num_nodesは平方数（例: 49, 100）である必要があります")
    graph = nx.grid_2d_graph(side, side)
    # ノードをint型に変換（0, 1, ..., num_nodes-1）
    mapping = {(i, j): i * side + j for i in range(side) for j in range(side)}
    graph = nx.relabel_nodes(graph, mapping)

    # BKB統計モデル用の属性を初期化
    if USE_TIME_WINDOW_LEARNING:
        # 時間窓学習の場合：既存のmax手法と同じ初期化
        for node in graph.nodes():
            graph.nodes[node]["best_known_bottleneck"] = 0
    else:
        # 統計的BKB学習の場合：共通モジュール使用
        initialize_graph_nodes_for_bkb(graph)
    for u, v in graph.edges():
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight
        graph[u][v]["local_min_bandwidth"] = weight
        graph[u][v]["local_max_bandwidth"] = weight
        graph[u][v]["pheromone"] = MIN_F
        graph[u][v]["max_pheromone"] = MAX_F
        graph[u][v]["min_pheromone"] = MIN_F
    return graph


# ------------------ メイン処理 ------------------
if __name__ == "__main__":  # noqa: C901
    # ===== 設定情報の表示 =====
    print("=" * 70)
    print("🚀 BKB学習設定")
    if USE_TIME_WINDOW_LEARNING:
        learning_method = "リングバッファ学習（直近1000個の観測値の最大値を記憶、外れたら削除 + 揮発率0.999）"
    elif USE_MULTI_SCALE_LEARNING:
        learning_method = "複数スケール学習（短期5世代 + 中期20世代 + 長期100世代）"
    elif USE_THREE_PHASE_LEARNING:
        learning_method = f"三段階学習（超短期α=0.95 + 短期α=0.7 + 長期α={BKB_CONFIG.mean_alpha:.4f}）"
    elif USE_TWO_PHASE_LEARNING:
        learning_method = f"二段階学習（短期α=0.5 + 長期α={BKB_CONFIG.mean_alpha:.4f}）"
    else:
        learning_method = f"単一EMA学習（α={BKB_CONFIG.mean_alpha:.4f}）"
    print(f"   学習手法: {learning_method}")
    print(f"   学習率（分散）: β = {BKB_CONFIG.var_beta:.4f}")
    print(f"   ボーナス係数: {BKB_CONFIG.achievement_bonus_base}x")
    print(f"   ペナルティ係数: {BKB_CONFIG.penalty_factor}")
    print(f"   帯域更新間隔: {BANDWIDTH_UPDATE_INTERVAL}世代ごと")
    print("   ★変動学習活用: フェロモン付加・揮発にBKB統計を反映★")
    print("=" * 70)

    # ===== ログファイルの初期化 =====
    import os

    log_filename = "./simulation_result/log_ant_available_bandwidth_rfc.csv"

    # ★★★ 詳細分析用ログファイル ★★★
    log_detailed_rfc = "./simulation_result/log_detailed_tracking_rfc.csv"

    for filename in [log_filename, log_detailed_rfc]:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"既存のログファイル '{filename}' を削除しました。")

        with open(filename, "w", newline="") as f:
            if filename == log_detailed_rfc:
                # ヘッダー行を書き込み
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "simulation",
                        "generation",
                        "optimal_bw",
                        "goal_ultra_short_bkb",
                        "goal_short_bkb",
                        "goal_long_bkb",
                        "goal_effective_bkb",
                        "goal_var",
                        "confidence",
                        "tracking_rate_ultra_short",
                        "tracking_rate_short",
                        "tracking_rate_effective",
                        "success_rate",
                    ]
                )
        print(f"ログファイル '{filename}' を初期化しました。")
    print()

    # ===== 変動設定の表示 =====
    print_fluctuation_settings()

    for sim in range(SIMULATIONS):
        # ===== シンプルな固定スタート・ゴール設定 =====
        NUM_NODES = 100
        START_NODE = random.randint(0, NUM_NODES - 1)
        GOAL_NODE = random.choice([n for n in range(NUM_NODES) if n != START_NODE])

        print(f"シミュレーション {sim+1}: スタート {START_NODE}, ゴール {GOAL_NODE}")

        # グラフはシミュレーションごとに一度だけ生成
        # graph = grid_graph(num_nodes=NUM_NODES, lb=1, ub=10)
        # graph = er_graph(num_nodes=NUM_NODES, edge_prob=0.12, lb=1, ub=10)
        graph = ba_graph(num_nodes=NUM_NODES, num_edges=6, lb=1, ub=15)

        set_pheromone_min_max_by_degree_and_width(graph)

        # ★変動エッジを選択 (設定に応じて自動選択)★
        fluctuating_edges = select_fluctuating_edges(graph)

        # ★変動対象エッジのみ AR(1)状態を初期化★
        edge_states = initialize_ar1_states(graph, fluctuating_edges)

        # ★初回の帯域更新も変動対象のみに適用される★
        update_available_bandwidth_ar1(graph, edge_states, 0)

        # 動的環境での初期最適解の計算（比較用）
        try:
            initial_optimal = calculate_current_optimal_bottleneck(
                graph, START_NODE, GOAL_NODE
            )
            print(f"動的環境での初期最適ボトルネック帯域: {initial_optimal}")
        except (nx.NetworkXNoPath, Exception):
            print("経路が存在しません。スキップします。")
            continue

        ant_log: list[int] = []
        bandwidth_change_log: list[int] = []  # 帯域変動の記録
        bandwidth_change_count = 0  # 帯域変動の累計回数

        for generation in range(GENERATION):
            # === AR(1)モデルによる帯域変動 ===
            bandwidth_changed = update_available_bandwidth_ar1(
                graph, edge_states, generation
            )
            bandwidth_change_log.append(1 if bandwidth_changed else 0)
            if bandwidth_changed:
                bandwidth_change_count += 1

            # === 最適解の再計算 ===
            current_optimal = calculate_current_optimal_bottleneck(
                graph, START_NODE, GOAL_NODE
            )
            if current_optimal == 0:
                # 経路が存在しない場合はスキップ
                continue

            # 帯域変動があった場合は通知
            if bandwidth_changed and generation % 50 == 0:
                avg_utilization = sum(edge_states.values()) / len(edge_states)
                print(
                    f"世代 {generation}: AR(1)帯域変動発生 - "
                    f"新しい最適値: {current_optimal}, "
                    f"平均利用率: {avg_utilization:.3f}"
                )

            # === アリの探索 ===
            ants = [
                Ant(START_NODE, GOAL_NODE, [START_NODE], []) for _ in range(ANT_NUM)
            ]

            temp_ant_list = list(ants)
            while temp_ant_list:
                ant_next_node_const_epsilon(
                    temp_ant_list, graph, ant_log, current_optimal, generation
                )

            # フェロモンの揮発
            # ★★★ 共通モジュールを使用したフェロモン揮発 ★★★
            volatilize_by_width(
                graph,
                volatilization_mode=VOLATILIZATION_MODE,
                base_evaporation_rate=V,
                penalty_factor=PENALTY_FACTOR,
                adaptive_rate_func=None,  # 帯域変動パターンに基づく適応的揮発は未使用
            )
            # BKB値の揮発処理（共通モジュール使用）
            evaporate_bkb_values(
                graph, BKB_EVAPORATION_RATE, use_int_cast=USE_TIME_WINDOW_LEARNING
            )

            # ★★★ 詳細ログ記録（10世代ごと） ★★★
            if generation % 10 == 0:
                goal_ultra_short = float(
                    graph.nodes[GOAL_NODE].get("ultra_short_ema_bkb") or 0.0
                )
                goal_short = float(graph.nodes[GOAL_NODE].get("short_ema_bkb") or 0.0)
                goal_long = float(graph.nodes[GOAL_NODE].get("long_ema_bkb") or 0.0)
                goal_effective = float(graph.nodes[GOAL_NODE].get("ema_bkb") or 0.0)
                goal_var_log = float(graph.nodes[GOAL_NODE].get("ema_bkb_var") or 0.0)
                confidence_log = calculate_confidence(goal_effective, goal_var_log)

                tracking_ultra_short = (
                    goal_ultra_short / current_optimal
                    if current_optimal > 0 and goal_ultra_short
                    else 0
                )
                tracking_short = (
                    goal_short / current_optimal
                    if current_optimal > 0 and goal_short
                    else 0
                )
                tracking_effective = (
                    goal_effective / current_optimal
                    if current_optimal > 0 and goal_effective
                    else 0
                )
                recent_success = (
                    sum(ant_log[-10:]) / min(len(ant_log), 10) if ant_log else 0
                )

                with open(log_detailed_rfc, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            sim + 1,
                            generation,
                            current_optimal,
                            goal_ultra_short,
                            goal_short,
                            goal_long,
                            goal_effective,
                            goal_var_log,
                            confidence_log,
                            tracking_ultra_short,
                            tracking_short,
                            tracking_effective,
                            recent_success,
                        ]
                    )

            # 進捗表示
            if generation % 100 == 0:
                recent_success_rate = (
                    sum(ant_log[-100:]) / min(len(ant_log), 100) if ant_log else 0
                )

                # ===== 確信度の計算（ゴールノードの統計）共通モジュール使用 =====
                goal_ultra_short_disp = float(
                    graph.nodes[GOAL_NODE].get("ultra_short_ema_bkb") or 0.0
                )
                goal_short_disp = float(
                    graph.nodes[GOAL_NODE].get("short_ema_bkb") or 0.0
                )
                goal_long_disp = float(
                    graph.nodes[GOAL_NODE].get("long_ema_bkb") or 0.0
                )
                goal_mean = float(graph.nodes[GOAL_NODE].get("ema_bkb") or 0.0)
                goal_var = float(graph.nodes[GOAL_NODE].get("ema_bkb_var") or 0.0)

                confidence = calculate_confidence(goal_mean, goal_var)
                goal_mean_display = goal_mean

                if USE_THREE_PHASE_LEARNING:
                    bkb_display = (
                        f"ゴールBKB[超短期={goal_ultra_short_disp:.1f}, "
                        f"短期={goal_short_disp:.1f}, 長期={goal_long_disp:.1f}, "
                        f"実効={goal_mean_display:.1f}]Mbps"
                    )
                elif USE_TWO_PHASE_LEARNING:
                    bkb_display = (
                        f"ゴールBKB[短期={goal_short_disp:.1f}, "
                        f"長期={goal_long_disp:.1f}, 実効={goal_mean_display:.1f}]Mbps"
                    )
                else:
                    bkb_display = f"ゴール平均BKB = {goal_mean_display:.1f}Mbps"

                print(
                    f"世代 {generation}: 成功率 = {recent_success_rate:.3f}, "
                    f"最適値 = {current_optimal}Mbps, "
                    f"{bkb_display}, "
                    f"確信度 = {confidence:.3f}"
                )

                # 最適解の詳細出力
                try:
                    optimal_path = max_load_path(graph, START_NODE, GOAL_NODE)
                    print(f"  最適経路: {' -> '.join(map(str, optimal_path))}")
                    print(f"  最適経路のボトルネック帯域: {current_optimal}Mbps")
                except nx.NetworkXNoPath:
                    print("  最適経路: 経路なし")

        # --- 結果の保存 ---
        with open(log_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        # 最終成功率の表示
        final_success_rate = sum(ant_log) / len(ant_log) if ant_log else 0

        # 最終確信度の計算（共通モジュール使用）
        goal_ultra_short_final = float(
            graph.nodes[GOAL_NODE].get("ultra_short_ema_bkb") or 0.0
        )
        goal_short_final = float(graph.nodes[GOAL_NODE].get("short_ema_bkb") or 0.0)
        goal_long_final = float(graph.nodes[GOAL_NODE].get("long_ema_bkb") or 0.0)
        goal_mean_final = float(graph.nodes[GOAL_NODE].get("ema_bkb") or 0.0)
        goal_var_final = float(graph.nodes[GOAL_NODE].get("ema_bkb_var") or 0.0)

        confidence_final = calculate_confidence(goal_mean_final, goal_var_final)
        goal_mean_final_display = goal_mean_final

        print(
            f"✅ シミュレーション {sim+1}/{SIMULATIONS} 完了 - "
            f"成功率: {final_success_rate:.3f}, "
            f"最終確信度: {confidence_final:.3f}, "
            f"ゴール平均BKB: {goal_mean_final_display:.1f}Mbps"
        )

    print(f"\n🎉 全{SIMULATIONS}回のシミュレーション完了！")
