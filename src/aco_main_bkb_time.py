import csv
import math
import random
import time
from statistics import mean, stdev

import networkx as nx

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
GENERATION = 1000  # 総世代数
SIMULATIONS = 100  # シミュレーションの試行回数

# ===== BKBモデル用パラメータ =====
PENALTY_FACTOR = 0.5  # BKBを下回るエッジへのペナルティ(0.0-1.0)
BKB_EVAPORATION_RATE = 0.999  # BKB値の揮発率


class Ant:
    def __init__(
        self, current: int, destination: int, route: list[int], width: list[int]
    ):
        self.current = current
        self.destination = destination
        self.route = route
        self.width = width

    def __repr__(self):
        return f"Ant(current={self.current}, destination={self.destination}, route={self.route}, width={self.width})"


def set_pheromone_min_max_by_degree_and_width(graph: nx.Graph) -> None:
    """ノードの隣接数と帯域幅に基づいてフェロモンの最小値と最大値を双方向に設定"""
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
    - VOLATILIZATION_MODE が 3 の場合: ノードのbest_known_bottleneck(BKB)に基づきペナルティを適用
    """
    for u, v in graph.edges():
        # u → v の揮発計算
        _apply_volatilization(graph, u, v)
        # v → u の揮発計算
        _apply_volatilization(graph, v, u)

    for node in graph.nodes():
        if "best_known_bottleneck" in graph.nodes[node]:
            graph.nodes[node]["best_known_bottleneck"] *= BKB_EVAPORATION_RATE


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
        # --- ノードのBKBに基づきペナルティを適用 ---
        # 基本の残存率を設定
        rate = V

        # 現在のノードuが知っている最良のボトルネック帯域(BKB)を取得
        bkb_u = graph.nodes[u].get("best_known_bottleneck", 0)

        # このエッジの帯域幅が、現在のノードuのBKBより低い場合、ペナルティを課す
        # 理由: ノードuが既に𝐾_uという最適値を知っているなら、
        #       それより小さい帯域のエッジは使わない方が良い（そのノードを通って
        #       この値でゴールできるはずなのに、その道を通るわけはない）
        if weight_uv < bkb_u:
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


# ===== 新しいパラメータ（功績ボーナス）=====


def update_pheromone(ant: Ant, graph: nx.Graph) -> None:
    """
    Antがゴールに到達したとき、経路上のフェロモンとノードのBKBを更新する。
    ★★★ フェロモンは経路上のエッジに「双方向」で付加する ★★★
    """
    bottleneck_bn = min(ant.width) if ant.width else 0
    if bottleneck_bn == 0:
        return

    # --- 経路上の各エッジにフェロモンを付加 ---
    for i in range(1, len(ant.route)):
        u, v = ant.route[i - 1], ant.route[i]

        # ステップ1：基本のフェロモン増加量を計算
        pheromone_increase = calculate_pheromone_increase(bottleneck_bn)

        # ステップ2：功績ボーナスの判定（使用しない）

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

    # --- BKBの更新（フェロモン付加の後に行う）---
    # 経路上の各ノードのBKBを更新
    for node in ant.route:
        current_bkb = graph.nodes[node].get("best_known_bottleneck", 0)
        graph.nodes[node]["best_known_bottleneck"] = max(current_bkb, bottleneck_bn)


# ===== 定数ε-Greedy法 =====
def ant_next_node_const_epsilon(
    ant_list: list[Ant],
    graph: nx.Graph,
    ant_log: list[int],
    current_optimal_bottleneck: int,
    timing_stats: dict,
) -> None:
    """
    固定パラメータ(α, β, ε)を用いた、最もシンプルなε-Greedy法で次のノードを決定する。
    """
    for ant in reversed(ant_list):
        # アリ個別の処理開始時刻を記録
        ant_step_start = time.time()

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
            # アリがゴールに到達した純粋な計算時間を記録
            ant_arrival_time = time.time() - ant_step_start
            timing_stats["ant_arrivals"].append(ant_arrival_time)

            update_pheromone(ant, graph)
            ant_log.append(1 if min(ant.width) >= current_optimal_bottleneck else 0)
            ant_list.remove(ant)
        elif len(ant.route) >= TTL:
            ant_log.append(0)
            ant_list.remove(ant)


def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """
    Barabási-Albertモデルでグラフを生成
    - 各ノードに best_known_bottleneck を初期化
    - 各エッジに帯域幅(weight)等を初期化
    """
    graph = nx.barabasi_albert_graph(num_nodes, num_edges)

    # ===== 全てのノードに best_known_bottleneck 属性を初期値 0 で追加 =====
    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0
    # =======================================================================

    for u, v in graph.edges():
        # リンクの帯域幅(weight)をランダムに設定
        weight = random.randint(lb, ub) * 10
        graph[u][v]["weight"] = weight

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
    - 各ノードに best_known_bottleneck を初期化
    - 各エッジに帯域幅(weight)等を初期化
    edge_probは、BAモデルと同程度のエッジ数になるように調整してください。
    """
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)

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
    - 各ノードに best_known_bottleneck を初期化
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
if __name__ == "__main__":
    # ===== ログファイルの初期化 =====
    import os

    log_filename = "./simulation_result/log_ant.csv"
    if os.path.exists(log_filename):
        os.remove(log_filename)
        print(f"既存のログファイル '{log_filename}' を削除しました。")

    with open(log_filename, "w", newline="") as f:
        pass  # 空のファイルを作成
    print(f"ログファイル '{log_filename}' を初期化しました。")

    # ===== 時間測定用の統計データ =====
    all_simulation_times = []
    all_generation_times = []
    all_ant_arrival_times = []
    all_graph_generation_times = []
    all_optimal_calculation_times = []
    all_pheromone_evaporation_times = []

    print(f"\n🚀 {SIMULATIONS}回のシミュレーション開始（詳細時間測定付き）")
    print("=" * 60)

    total_start_time = time.time()

    for sim in range(SIMULATIONS):
        # I/O処理（時間測定対象外）
        print(f"\nシミュレーション {sim+1}/{SIMULATIONS}: スタート準備中...")

        simulation_start_time = time.time()

        # ===== シンプルな固定スタート・ゴール設定 =====
        NUM_NODES = 100
        START_NODE = random.randint(0, NUM_NODES - 1)
        GOAL_NODE = random.choice([n for n in range(NUM_NODES) if n != START_NODE])

        # I/O処理（時間測定対象外）
        print(f"  スタート {START_NODE}, ゴール {GOAL_NODE}")

        # ===== グラフ生成時間の測定 =====
        graph_gen_start = time.time()
        graph = ba_graph(num_nodes=NUM_NODES, num_edges=6, lb=1, ub=10)
        set_pheromone_min_max_by_degree_and_width(graph)
        graph_gen_time = time.time() - graph_gen_start
        all_graph_generation_times.append(graph_gen_time)

        # ===== 最適解計算時間の測定 =====
        optimal_calc_start = time.time()
        try:
            optimal_path = max_load_path(graph, START_NODE, GOAL_NODE)
            optimal_bottleneck = min(
                graph.edges[u, v]["weight"]
                for u, v in zip(optimal_path[:-1], optimal_path[1:])
            )
            optimal_calc_time = time.time() - optimal_calc_start
            all_optimal_calculation_times.append(optimal_calc_time)

            # I/O処理（時間測定対象外）
            print(
                f"  最適ボトルネック帯域: {optimal_bottleneck} (計算時間: {optimal_calc_time:.4f}s)"
            )
        except nx.NetworkXNoPath:
            print("  経路が存在しません。スキップします。")
            continue

        ant_log: list[int] = []
        generation_times = []

        # シミュレーション内の時間統計
        timing_stats = {
            "ant_arrivals": [],
            "generation_times": [],
            "evaporation_times": [],
        }

        # I/O処理時間を除外するため、実際の処理開始時刻を記録
        actual_simulation_start = time.time()

        for generation in range(GENERATION):
            generation_start_time = time.time()

            ants = [
                Ant(START_NODE, GOAL_NODE, [START_NODE], []) for _ in range(ANT_NUM)
            ]

            temp_ant_list = list(ants)
            while temp_ant_list:
                ant_next_node_const_epsilon(
                    temp_ant_list, graph, ant_log, optimal_bottleneck, timing_stats
                )

            # ===== フェロモン揮発時間の測定 =====
            evaporation_start = time.time()
            volatilize_by_width(graph)
            evaporation_time = time.time() - evaporation_start
            timing_stats["evaporation_times"].append(evaporation_time)

            generation_time = time.time() - generation_start_time
            generation_times.append(generation_time)

            # I/O処理（時間測定対象外） - 進捗表示（100世代ごと）
            if generation % 100 == 0:
                # I/O処理時間を一時停止
                io_start = time.time()
                recent_success_rate = (
                    sum(ant_log[-100:]) / min(len(ant_log), 100) if ant_log else 0
                )
                avg_gen_time = mean(generation_times[-100:]) if generation_times else 0
                print(
                    f"    世代 {generation}: 成功率 = {recent_success_rate:.3f}, "
                    f"平均世代時間 = {avg_gen_time:.4f}s"
                )
                io_time = time.time() - io_start
                # I/O時間を実際のシミュレーション開始時刻に加算して除外
                actual_simulation_start += io_time

        # シミュレーション終了時の統計（I/O処理を除外して計算）
        pure_simulation_time = time.time() - actual_simulation_start
        all_simulation_times.append(pure_simulation_time)
        all_generation_times.extend(generation_times)
        all_ant_arrival_times.extend(timing_stats["ant_arrivals"])
        all_pheromone_evaporation_times.extend(timing_stats["evaporation_times"])

        # I/O処理（時間測定対象外） - 結果の保存
        with open("./simulation_result/log_ant.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ant_log)

        # I/O処理（時間測定対象外） - 最終成功率の表示
        final_success_rate = sum(ant_log) / len(ant_log) if ant_log else 0
        avg_ant_arrival = (
            mean(timing_stats["ant_arrivals"]) if timing_stats["ant_arrivals"] else 0
        )
        avg_generation_time = mean(generation_times) if generation_times else 0

        print(
            f"  ✅ 完了 - 成功率: {final_success_rate:.3f}, "
            f"純粋計算時間: {pure_simulation_time:.2f}s"
        )
        print(
            f"     平均世代時間: {avg_generation_time:.4f}s, "
            f"平均アリ到達時間: {avg_ant_arrival:.6f}s"
        )

    total_time = time.time() - total_start_time

    # ===== 最終統計の表示 =====
    print("\n" + "=" * 60)
    print("🎯 最終統計結果（100回シミュレーション平均）")
    print("=" * 60)

    print(f"📊 全体実行時間: {total_time:.2f}秒")
    print()

    if all_simulation_times:
        print(f"🔄 シミュレーション時間統計:")
        print(f"   平均: {mean(all_simulation_times):.3f}s")
        print(
            f"   標準偏差: {stdev(all_simulation_times) if len(all_simulation_times) > 1 else 0:.3f}s"
        )
        print(f"   最小: {min(all_simulation_times):.3f}s")
        print(f"   最大: {max(all_simulation_times):.3f}s")
        print()

    if all_generation_times:
        print(f"🧬 世代時間統計:")
        print(f"   平均: {mean(all_generation_times):.6f}s")
        print(
            f"   標準偏差: {stdev(all_generation_times) if len(all_generation_times) > 1 else 0:.6f}s"
        )
        print(f"   最小: {min(all_generation_times):.6f}s")
        print(f"   最大: {max(all_generation_times):.6f}s")
        print(f"   総世代数: {len(all_generation_times)}")
        print()

    if all_ant_arrival_times:
        print(f"🐜 アリ到達時間統計:")
        print(f"   平均: {mean(all_ant_arrival_times):.8f}s")
        print(
            f"   標準偏差: {stdev(all_ant_arrival_times) if len(all_ant_arrival_times) > 1 else 0:.8f}s"
        )
        print(f"   最小: {min(all_ant_arrival_times):.8f}s")
        print(f"   最大: {max(all_ant_arrival_times):.8f}s")
        print(f"   総到達回数: {len(all_ant_arrival_times)}")
        print()

    if all_graph_generation_times:
        print(f"🕸️  グラフ生成時間統計:")
        print(f"   平均: {mean(all_graph_generation_times):.6f}s")
        print(
            f"   標準偏差: {stdev(all_graph_generation_times) if len(all_graph_generation_times) > 1 else 0:.6f}s"
        )
        print(f"   最小: {min(all_graph_generation_times):.6f}s")
        print(f"   最大: {max(all_graph_generation_times):.6f}s")
        print()

    if all_optimal_calculation_times:
        print(f"🎯 最適解計算時間統計:")
        print(f"   平均: {mean(all_optimal_calculation_times):.6f}s")
        print(
            f"   標準偏差: {stdev(all_optimal_calculation_times) if len(all_optimal_calculation_times) > 1 else 0:.6f}s"
        )
        print(f"   最小: {min(all_optimal_calculation_times):.6f}s")
        print(f"   最大: {max(all_optimal_calculation_times):.6f}s")
        print()

    if all_pheromone_evaporation_times:
        print(f"💨 フェロモン揮発時間統計:")
        print(f"   平均: {mean(all_pheromone_evaporation_times):.8f}s")
        print(
            f"   標準偏差: {stdev(all_pheromone_evaporation_times) if len(all_pheromone_evaporation_times) > 1 else 0:.8f}s"
        )
        print(f"   最小: {min(all_pheromone_evaporation_times):.8f}s")
        print(f"   最大: {max(all_pheromone_evaporation_times):.8f}s")
        print()

    print("🎉 全シミュレーション完了！")
    print("=" * 60)
