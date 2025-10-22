import csv
import math
import random
from typing import Dict, Tuple

import networkx as nx  # type: ignore[import-untyped]

from modified_dijkstra import max_load_path

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
GENERATION = 10000  # 総世代数
SIMULATIONS = 100  # シミュレーションの試行回数

# ===== BKB統計モデル（RFC 6298 準拠）=====
BKB_MEAN_ALPHA = 1 / 8  # SRTTの学習率 (0.125) - RFC 6298標準
BKB_VAR_BETA = 1 / 4  # RTTVARの学習率 (0.25) - RFC 6298標準
BKB_CONFIDENCE_K = 1.0  # 信頼区間幅の係数（平均 - K*分散）
ACHIEVEMENT_BONUS = 1.5  # BKB「平均」を更新した場合の報酬ボーナス係数
PENALTY_FACTOR = 0.5  # BKB「信頼下限」を下回るエッジへのペナルティ

# ===== 動的帯域変動パラメータ（AR(1)モデル） =====
BANDWIDTH_UPDATE_INTERVAL = 100  # 何世代ごとに帯域を更新するか

MEAN_UTILIZATION: float = 0.4  # (根拠: ISPの一般的な運用マージン)
AR_COEFFICIENT: float = 0.95  # (根拠: ネットワークトラフィックの高い自己相関)
NOISE_VARIANCE: float = 0.000975  # (根拠: 上記2値から逆算した値)


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


def volatilize_by_width(graph: nx.Graph) -> None:
    """
    各エッジのフェロモン値を双方向で揮発させる
    - VOLATILIZATION_MODE が 0 の場合: 固定の揮発率を適用
    - VOLATILIZATION_MODE が 1 の場合: エッジのlocal_min/max帯域幅を基準に揮発量を調整
    - VOLATILIZATION_MODE が 2 の場合: エッジの帯域幅の平均/分散を基準に揮発量を計算
    - VOLATILIZATION_MODE が 3 の場合: ノードのBKB統計（平均・分散）に基づきペナルティを適用

    ★BKBの「忘却」はEMA（指数移動平均）が担うため、
    　従来のBKB_EVAPORATION_RATEによる揮発処理は不要となり、削除。
    """
    for u, v in graph.edges():
        # u → v の揮発計算
        _apply_volatilization(graph, u, v)
        # v → u の揮発計算
        _apply_volatilization(graph, v, u)


def _apply_volatilization(graph: nx.Graph, u: int, v: int) -> None:
    """
    指定された方向のエッジ (u → v) に対して揮発処理を適用
    """
    # 現在のフェロモン値と帯域幅を取得
    current_pheromone = graph[u][v]["pheromone"]
    weight_uv = graph[u][v]["weight"]

    # エッジのローカル最小・最大帯域幅を取得
    local_min_bandwidth = graph[u][v]["local_min_bandwidth"]
    local_max_bandwidth = graph[u][v]["local_max_bandwidth"]

    # 揮発率の計算
    if VOLATILIZATION_MODE == 0:
        # --- 既存の揮発式 ---
        # 最大帯域幅100Mbpsを基準に固定値で揮発率を計算
        rate = V

    # 0.99に設定する方が，最適解既知でないときに如実に良くなる．
    elif VOLATILIZATION_MODE == 1:
        # --- 帯域幅の最小値・最大値を基準に揮発量を調整 ---
        # エッジの帯域幅が、ローカルな最小・最大帯域幅のどの位置にあるかを計算
        if local_max_bandwidth == local_min_bandwidth:
            # 未使用エッジの場合：帯域幅が大きいほど rate が 1 に近づく
            rate = 0.98
        else:
            # 使用済みエッジの場合：帯域幅の相対位置を基準に揮発量を調整
            normalized_position = (weight_uv - local_min_bandwidth) / max(
                1, (local_max_bandwidth - local_min_bandwidth)
            )
            rate = 0.98 * normalized_position

    # FIXME: OverflowError: cannot convert float infinity to integer
    elif VOLATILIZATION_MODE == 2:
        # --- 平均・分散を基準に揮発量を調整 ---
        # 平均帯域幅と標準偏差を計算し、それを基に揮発率を算出
        if local_max_bandwidth == local_min_bandwidth:
            # 未使用エッジの場合：帯域幅が大きいほど rate が 1 に近づく
            avg_bandwidth = weight_uv
            std_dev = 1  # デフォルト値
        else:
            # 使用済みエッジの場合
            avg_bandwidth = 0.5 * (local_min_bandwidth + local_max_bandwidth)
            std_dev = max(abs(local_max_bandwidth - avg_bandwidth), 1)

        # 平均・分散に基づいて揮発率を計算
        gamma = 1.0  # 減衰率の調整パラメータ
        rate = math.exp(-gamma * (avg_bandwidth - weight_uv) / std_dev)

    elif VOLATILIZATION_MODE == 3:
        # --- ノードのBKB統計（平均・分散）に基づきペナルティを適用 ---
        # 基本の残存率を設定
        rate = V

        # 行き先ノードvのBKB統計（平均と分散）を取得
        bkb_mean = graph.nodes[v].get("ema_bkb")
        bkb_var = graph.nodes[v].get("ema_bkb_var", 0.0)

        if bkb_mean is None:
            # まだ学習していないノードはペナルティ対象外
            bkb_mean, bkb_var = 0.0, 0.0

        # 信頼区間の下限（平均 - K * 分散）を計算
        lower_bound = bkb_mean - BKB_CONFIDENCE_K * bkb_var

        # このエッジの帯域幅が、信頼できる期待値（下限）より低い場合、ペナルティを課す
        if weight_uv < lower_bound:
            rate *= PENALTY_FACTOR  # 残存率を下げることで、揮発を促進する

    else:
        raise ValueError("Invalid VOLATILIZATION_MODE. Choose 0, 1, 2 or 3.")

    # フェロモン値を計算して更新
    new_pheromone = max(
        math.floor(current_pheromone * rate), graph[u][v]["min_pheromone"]
    )
    graph[u][v]["pheromone"] = new_pheromone


def calculate_pheromone_increase(bottleneck_bandwidth: int) -> float:
    """
    フェロモン付加量を計算する。
    """
    # ボトルネック帯域が大きいほど、指数的に報酬を増やす
    # ただし、過大にならないよう2乗程度に抑える
    return float(bottleneck_bandwidth * 10)


def initialize_ar1_states(graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    """
    各エッジのAR(1)モデルの初期利用率を設定する
    """
    edge_states = {}
    for u, v in graph.edges():
        # u -> v / v -> u の初期利用率
        util_uv = random.uniform(0.3, 0.5)
        util_vu = random.uniform(0.3, 0.5)
        edge_states[(u, v)] = util_uv
        edge_states[(v, u)] = util_vu

        # 標準的な可用帯域計算: キャパシティ × (1 - 使用率)
        capacity = graph[u][v]["original_weight"]
        avg_util = 0.5 * (util_uv + util_vu)
        initial_available = int(round(capacity * (1.0 - avg_util)))
        # 10Mbps刻みに丸め
        initial_available = ((initial_available + 5) // 10) * 10
        graph[u][v]["weight"] = initial_available
        graph[u][v]["local_min_bandwidth"] = initial_available
        graph[u][v]["local_max_bandwidth"] = initial_available
    return edge_states


def update_available_bandwidth_ar1(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], float], generation: int
) -> bool:
    """
    AR(1)モデルによる帯域変動
    - BANDWIDTH_UPDATE_INTERVAL世代ごとにのみ更新
    """
    # 更新間隔でない世代は変化なし
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), current_utilization in edge_states.items():
        # AR(1)モデル: X(t) = c + φ*X(t-1) + ε(t)
        noise = random.gauss(0, math.sqrt(NOISE_VARIANCE))

        new_utilization = (
            (1 - AR_COEFFICIENT) * MEAN_UTILIZATION  # 平均への回帰
            + AR_COEFFICIENT * current_utilization  # 過去の値への依存
            + noise  # ランダムノイズ
        )

        # 利用率を0.05 - 0.95の範囲にクリップ
        new_utilization = max(0.05, min(0.95, new_utilization))

        # 状態を更新
        edge_states[(u, v)] = new_utilization

        # 標準的な可用帯域計算: キャパシティ × (1 - 使用率)
        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        # 10Mbps刻みに丸め
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        # 変化があったかチェック
        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        # グラフのweight属性を更新
        graph[u][v]["weight"] = available_bandwidth

        # local_min/max_bandwidth も更新
        graph[u][v]["local_min_bandwidth"] = graph[u][v]["weight"]
        graph[u][v]["local_max_bandwidth"] = graph[u][v]["weight"]

    return bandwidth_changed


def calculate_current_optimal_bottleneck(
    graph: nx.Graph, start_node: int, goal_node: int
) -> int:
    """
    現在のネットワーク状態での最適ボトルネック帯域を計算
    """
    try:
        optimal_path = max_load_path(graph, start_node, goal_node)
        optimal_bottleneck = min(
            graph.edges[u, v]["weight"]
            for u, v in zip(optimal_path[:-1], optimal_path[1:])
        )
        return optimal_bottleneck
    except nx.NetworkXNoPath:
        return 0


def update_pheromone(ant: Ant, graph: nx.Graph) -> None:
    """
    ★★★ RFC 6298準拠の統計的BKB学習モデル ★★★
    Antがゴールした時、経路上のBKB統計情報（平均・分散）を更新し、
    フェロモンを付加する。
    """
    bottleneck_bn = min(ant.width) if ant.width else 0
    if bottleneck_bn == 0:
        return

    # --- ステップ1: ノード側のBKB統計（平均・分散）の更新（RFC 6298 準拠）---
    for node in ant.route:
        mean_prev = graph.nodes[node].get("ema_bkb")
        var_prev = graph.nodes[node].get("ema_bkb_var", 0.0)

        if mean_prev is None:
            # 最初のサンプル (Karn's Algorithm)
            mean_new = float(bottleneck_bn)
            var_new = float(bottleneck_bn) / 2.0  # TCPのRTO初期値計算に準拠
        else:
            # 2回目以降 (RFC 6298)
            # 信頼度（ばらつき）の更新 (RTTVARの計算)
            deviation = abs(bottleneck_bn - mean_prev)
            var_new = (1 - BKB_VAR_BETA) * var_prev + BKB_VAR_BETA * deviation
            # 平均値の更新 (SRTTの計算)
            mean_new = (1 - BKB_MEAN_ALPHA) * mean_prev + BKB_MEAN_ALPHA * bottleneck_bn

        graph.nodes[node]["ema_bkb"] = mean_new
        graph.nodes[node]["ema_bkb_var"] = var_new

        # 互換維持：古いBKB最大値も（平均値で）更新しておく
        graph.nodes[node]["best_known_bottleneck"] = max(
            graph.nodes[node].get("best_known_bottleneck", 0), int(mean_new)
        )

    # --- ステップ2: フェロモン付加（功績ボーナスは「平均」基準に変更）---
    for i in range(1, len(ant.route)):
        u, v = ant.route[i - 1], ant.route[i]

        pheromone_increase = calculate_pheromone_increase(bottleneck_bn)

        # ボーナス判定: アントの帯域が、行き先ノードvの「平均BKB」より大きいか？
        bkb_v_mean = graph.nodes[v].get("ema_bkb") or 0.0
        if bottleneck_bn > bkb_v_mean:
            pheromone_increase *= ACHIEVEMENT_BONUS

        # ===== ★★★ フェロモンを双方向に付加 ★★★ =====
        # 順方向 (u -> v) のフェロモンを更新
        max_pheromone_uv = graph.edges[u, v].get("max_pheromone", MAX_F)
        graph.edges[u, v]["pheromone"] = min(
            graph.edges[u, v]["pheromone"] + pheromone_increase,
            max_pheromone_uv,
        )

        # 逆方向 (v -> u) のフェロモンも更新
        max_pheromone_vu = graph.edges[v, u].get("max_pheromone", MAX_F)
        graph.edges[v, u]["pheromone"] = min(
            graph.edges[v, u]["pheromone"] + pheromone_increase,
            max_pheromone_vu,
        )
        # =======================================================


# ===== 定数ε-Greedy法 =====
def ant_next_node_const_epsilon(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    current_optimal_bottleneck: int,
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
            update_pheromone(ant, graph)
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
    for node in graph.nodes():
        graph.nodes[node]["ema_bkb"] = None  # 平均（SRTT相当）
        graph.nodes[node]["ema_bkb_var"] = 0.0  # 分散（RTTVAR相当）
        graph.nodes[node]["best_known_bottleneck"] = 0  # 互換維持用
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

    for node in graph.nodes():
        graph.nodes[node]["ema_bkb"] = None  # 平均（SRTT相当）
        graph.nodes[node]["ema_bkb_var"] = 0.0  # 分散（RTTVAR相当）
        graph.nodes[node]["best_known_bottleneck"] = 0  # 互換維持用

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
    for node in graph.nodes():
        graph.nodes[node]["ema_bkb"] = None  # 平均（SRTT相当）
        graph.nodes[node]["ema_bkb_var"] = 0.0  # 分散（RTTVAR相当）
        graph.nodes[node]["best_known_bottleneck"] = 0  # 互換維持用
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
    # ===== ログファイルの初期化 =====
    import os

    log_filename = "./simulation_result/log_ant_available_bandwidth_rfc.csv"
    if os.path.exists(log_filename):
        os.remove(log_filename)
        print(f"既存のログファイル '{log_filename}' を削除しました。")

    with open(log_filename, "w", newline="") as f:
        pass  # 空のファイルを作成
    print(f"ログファイル '{log_filename}' を初期化しました。")

    for sim in range(SIMULATIONS):
        # ===== シンプルな固定スタート・ゴール設定 =====
        NUM_NODES = 100
        START_NODE = random.randint(0, NUM_NODES - 1)
        GOAL_NODE = random.choice([n for n in range(NUM_NODES) if n != START_NODE])

        print(f"シミュレーション {sim+1}: スタート {START_NODE}, ゴール {GOAL_NODE}")

        # グラフはシミュレーションごとに一度だけ生成
        # graph = grid_graph(num_nodes=NUM_NODES, lb=1, ub=10)
        # graph = er_graph(num_nodes=NUM_NODES, edge_prob=0.12, lb=1, ub=10)
        graph = ba_graph(num_nodes=NUM_NODES, num_edges=6, lb=1, ub=10)

        set_pheromone_min_max_by_degree_and_width(graph)

        # AR(1)状態初期化
        edge_states = initialize_ar1_states(graph)

        # 初回のAR(1)帯域更新を適用（世代0として呼び出し）
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
                    temp_ant_list, graph, ant_log, current_optimal
                )

            # フェロモンの揮発
            volatilize_by_width(graph)

            # 進捗表示
            if generation % 100 == 0:
                recent_success_rate = (
                    sum(ant_log[-100:]) / min(len(ant_log), 100) if ant_log else 0
                )
                bandwidth_change_rate = (
                    sum(bandwidth_change_log[-100:])
                    / min(len(bandwidth_change_log), 100)
                    if bandwidth_change_log
                    else 0
                )
                avg_utilization = sum(edge_states.values()) / len(edge_states)
                print(
                    f"世代 {generation}: 成功率 = {recent_success_rate:.3f}, "
                    f"帯域変化率 = {bandwidth_change_rate:.3f}, "
                    f"平均利用率 = {avg_utilization:.3f}, "
                    f"最適値 = {current_optimal}, "
                    f"累計変動回数 = {bandwidth_change_count}"
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
        total_bandwidth_changes = sum(bandwidth_change_log)
        print(
            f"✅ シミュレーション {sim+1}/{SIMULATIONS} 完了 - "
            f"成功率: {final_success_rate:.3f}, "
            f"帯域変動回数: {total_bandwidth_changes}/{GENERATION} "
            f"({total_bandwidth_changes/GENERATION*100:.1f}%)"
        )

    print(f"\n🎉 全{SIMULATIONS}回のシミュレーション完了！")
