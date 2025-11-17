import csv
import random
from typing import Callable, Optional

import networkx as nx  # type: ignore[import-untyped]

from bandwidth_fluctuation_config import (
    BANDWIDTH_UPDATE_INTERVAL,
    initialize_ar1_states,
    print_fluctuation_settings,
    select_fluctuating_edges,
    update_available_bandwidth_ar1,
)
from bandwidth_monitoring import (
    calculate_adaptive_evaporation_rate,
    calculate_predictive_heuristic,
    observe_all_edges_bandwidth,
    update_patterns_for_all_edges,
)
from bkb_learning import (
    evaporate_bkb_values,  # ★BKB揮発処理を追加★
    update_node_bkb_time_window_max,  # ★リングバッファ学習★
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

# ===== BKB学習設定（リングバッファ学習）=====
TIME_WINDOW_SIZE = 10  # リングバッファサイズ（記憶する観測値の数）
BKB_EVAPORATION_RATE = 0.999  # BKB値の揮発率
PENALTY_FACTOR = 0.5  # BKBを下回るエッジへのペナルティ

# ===== 適応的揮発モデル設定 =====
USE_ADAPTIVE_EVAPORATION = True  # ★帯域変動パターンに基づく適応的揮発を有効化★
ADAPTIVE_PATTERN_UPDATE_INTERVAL = 10  # パターン学習の更新間隔（世代数）
ADAPTIVE_MIN_SAMPLES = 10  # パターン学習に必要な最小サンプル数
USE_WAVELET_PERIODICITY = (
    False  # True: ウェーブレット周期性検出, False: 自己相関周期性検出
)
ADAPTIVE_PREDICTION_METHOD = "ma"  # 適応的揮発での予測手法（"ar1", "ma", "ema"）
USE_PREDICTION_VARIABILITY = True  # 予測変動性に基づく適応的揮発調整を有効化

# ===== 予測的ヒューリスティック設定（研究コンペンディウム推奨: Phase 3）=====
USE_PREDICTIVE_HEURISTIC = True  # ★予測的ヒューリスティックを有効化★
PREDICTIVE_HEURISTIC_METHOD = "ma"  # 予測手法（"ar1", "ma", "ema"）
GAMMA = 1.0  # 予測ヒューリスティックの重み（研究コンペンディウムでは通常1.0）

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


# BKB更新関数（リングバッファベースのBKB計算）
def _bkb_update_func(
    graph: nx.Graph, node: int, bottleneck: float, generation: int
) -> None:
    """BKB更新関数（リングバッファベースのBKB計算）"""
    update_node_bkb_time_window_max(
        graph, node, bottleneck, generation, time_window_size=TIME_WINDOW_SIZE
    )


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

            # ★★★ 研究コンペンディウム推奨: 予測的ヒューリスティックの統合（Phase 3）★★★
            # エッジ$(i, j)$の予測される将来の帯域幅に基づくヒューリスティック成分$\eta_{pred}(ij)$
            if USE_PREDICTIVE_HEURISTIC:
                weight_predicted = [
                    calculate_predictive_heuristic(
                        graph,
                        ant.current,
                        n,
                        prediction_method=PREDICTIVE_HEURISTIC_METHOD,
                        gamma=GAMMA,
                    )
                    for n in candidates
                ]
                # 状態遷移確率: P_{ij} ∝ τ_{ij}^α * η_{distance}(ij)^β * η_{pred}(ij)^γ
                weights = [
                    p * w * pred
                    for p, w, pred in zip(
                        weight_pheromone, weight_width, weight_predicted
                    )
                ]
            else:
                # 既存の手法（予測的ヒューリスティックなし）
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
            # 帯域観測関数を設定（適応的揮発が有効な場合のみ）
            # 全エッジ監視を使用するため、アリ経路での観測は不要
            observe_func = None  # observe_all_edges_bandwidth() で全エッジを監視
            update_pheromone(
                ant,
                graph,
                generation,
                max_pheromone=MAX_F,
                bkb_update_func=_bkb_update_func,
                pheromone_increase_func=None,  # シンプル版を使用
                observe_bandwidth_func=observe_func,  # ★帯域観測を有効化★
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

    # ===== BKB初期化（リングバッファ学習）=====
    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0

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

    # ===== BKB初期化（リングバッファ学習）=====
    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0

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

    # ===== BKB初期化（リングバッファ学習）=====
    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0
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
    print(
        f"   学習手法: リングバッファ学習（直近{TIME_WINDOW_SIZE}個の観測値の最大値を記憶）"
    )
    print(f"   BKB揮発率: {BKB_EVAPORATION_RATE}")
    print(f"   ペナルティ係数: {PENALTY_FACTOR}")
    print(f"   帯域更新間隔: {BANDWIDTH_UPDATE_INTERVAL}世代ごと")
    if USE_ADAPTIVE_EVAPORATION:
        print("   ★適応的揮発: 帯域変動パターン学習に基づく揮発率調整を有効化★")
        print(f"      パターン更新間隔: {ADAPTIVE_PATTERN_UPDATE_INTERVAL}世代")
        print(f"      最小サンプル数: {ADAPTIVE_MIN_SAMPLES}")
        print(
            f"      周期性検出: {'ウェーブレット' if USE_WAVELET_PERIODICITY else '自己相関'}"
        )
        print(f"      適応的揮発の予測手法: {ADAPTIVE_PREDICTION_METHOD.upper()}")
        print(
            f"      予測変動性ベース調整: {'有効' if USE_PREDICTION_VARIABILITY else '無効'}"
        )
    if USE_PREDICTIVE_HEURISTIC:
        print("   ★予測的ヒューリスティック: 有効★")
        print(f"      予測手法: {PREDICTIVE_HEURISTIC_METHOD.upper()}")
        print(f"      重み係数 (γ): {GAMMA}")
    print("=" * 70)

    # ===== ログファイルの初期化 =====
    import os

    log_filename = "./simulation_result/log_ant_available_bandwidth_ma_autocorr.csv"

    if os.path.exists(log_filename):
        os.remove(log_filename)
        print(f"既存のログファイル '{log_filename}' を削除しました。")

    with open(log_filename, "w", newline="") as f:
        pass  # 空のファイルを作成
    print(f"ログファイル '{log_filename}' を初期化しました。")
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

            # === ★★★ 全エッジの帯域を観測（毎世代）★★★ ===
            # 研究コンペンディウム推奨: Phase 1 - 全エッジの継続的監視
            # アリに依存せず、毎世代すべてのエッジの帯域を記録
            if USE_ADAPTIVE_EVAPORATION:
                observe_all_edges_bandwidth(graph)

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

            # ★★★ 帯域変動パターンの学習（適応的揮発が有効な場合のみ）★★★
            if USE_ADAPTIVE_EVAPORATION:
                update_patterns_for_all_edges(
                    graph,
                    min_samples=ADAPTIVE_MIN_SAMPLES,
                    update_interval=ADAPTIVE_PATTERN_UPDATE_INTERVAL,
                    generation=generation,
                    use_wavelet=USE_WAVELET_PERIODICITY,  # ★周期性検出手法を指定★
                )

            # フェロモンの揮発
            # ★★★ 共通モジュールを使用したフェロモン揮発 ★★★
            # 適応的揮発関数を設定（適応的揮発が有効な場合のみ）
            adaptive_func: Optional[Callable[[nx.Graph, int, int], float]]
            if USE_ADAPTIVE_EVAPORATION:
                # ラッパー関数を作成（設定値を渡す）
                def adaptive_func(g, u, v):
                    return calculate_adaptive_evaporation_rate(
                        g,
                        u,
                        v,
                        use_prediction_variability=USE_PREDICTION_VARIABILITY,
                        prediction_method=ADAPTIVE_PREDICTION_METHOD,
                    )

            else:
                adaptive_func = None
            volatilize_by_width(
                graph,
                volatilization_mode=VOLATILIZATION_MODE,
                base_evaporation_rate=V,
                penalty_factor=PENALTY_FACTOR,
                adaptive_rate_func=adaptive_func,  # ★帯域変動パターンに基づく適応的揮発★
            )
            # BKB値の揮発処理（共通モジュール使用）
            evaporate_bkb_values(graph, BKB_EVAPORATION_RATE, use_int_cast=False)

            # 進捗表示
            if generation % 100 == 0:
                recent_success_rate = (
                    sum(ant_log[-100:]) / min(len(ant_log), 100) if ant_log else 0
                )

                # ===== ゴールノードのBKB値を取得 =====
                goal_bkb = float(
                    graph.nodes[GOAL_NODE].get("best_known_bottleneck") or 0.0
                )
                tracking_rate = (
                    goal_bkb / current_optimal
                    if current_optimal > 0 and goal_bkb > 0
                    else 0.0
                )

                print(
                    f"世代 {generation}: 成功率 = {recent_success_rate:.3f}, "
                    f"最適値 = {current_optimal}Mbps, "
                    f"ゴールBKB = {goal_bkb:.1f}Mbps, "
                    f"追従率 = {tracking_rate:.3f}"
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
        goal_bkb_final = float(
            graph.nodes[GOAL_NODE].get("best_known_bottleneck") or 0.0
        )

        print(
            f"✅ シミュレーション {sim+1}/{SIMULATIONS} 完了 - "
            f"成功率: {final_success_rate:.3f}, "
            f"最終ゴールBKB: {goal_bkb_final:.1f}Mbps"
        )

    print(f"\n🎉 全{SIMULATIONS}回のシミュレーション完了！")
