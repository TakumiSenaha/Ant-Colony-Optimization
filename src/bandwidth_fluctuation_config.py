"""
帯域変動パラメータの共通設定ファイル

このファイルで帯域変動の設定を一元管理し、
複数のACO実装ファイルで共有します。
"""

import math
import random
from typing import Dict, List, Tuple

import networkx as nx  # type: ignore[import-untyped]

# ===== 帯域変動設定パラメータ =====
# ★★★ メイン設定：ここを変更するだけで全ファイルに反映されます ★★★
BANDWIDTH_UPDATE_INTERVAL = 100000  # 何世代ごとに帯域を更新するか（1=毎世代）

# ===== エッジ選択方法 =====
# "hub": ハブノード選択（推奨）
# "hub_partial": ハブノードの隣接エッジを部分的に選択
# "hub_random": ハブノードの隣接エッジをランダムに選択
# "random": 完全ランダムでエッジを選択
# "betweenness": 媒介中心性選択
EDGE_SELECTION_METHOD = "hub"  # ★ハブノード選択を推奨★

# ===== 統一パラメータ =====
FLUCTUATION_PERCENTAGE = 0.1  # 統一パラメータ：選択方法に応じて自動解釈
# - ハブノード選択: ハブノードとして選択するノードの割合
# - ランダム選択: ランダムに選択するエッジの割合
# - 媒介中心性選択: 媒介中心性で選択するエッジの割合

# ===== 選択方法別の詳細パラメータ =====
# ハブノード選択用
HUB_NEIGHBOR_EDGE_RATIO = 1.0  # ハブノードの隣接エッジのうち変動させる割合（0.0-1.0）
HUB_NEIGHBOR_SELECTION_METHOD = "degree"  # "degree": 次数順, "random": ランダム

# ランダム選択用
RANDOM_SELECTION_TIMING = (
    "fixed"  # "fixed": シミュレーション開始時固定, "dynamic": 毎回ランダム
)

# ===== 変動モデル選択 =====
# "ar1": AR(1)モデル（現在実装済み、推奨）
# "ar2": AR(2)モデル（2次自己回帰）
# "random_walk": ランダムウォークモデル
# "ou_process": オルンシュタイン・ウーレンベック過程
# "sinusoidal": 正弦波モデル（周期的変動）
# "step_function": 階段関数モデル（段階的変動）
# "garch": GARCHモデル（ボラティリティクラスタリング）
# "markov_chain": マルコフ連鎖モデル（状態遷移）
# "mixed": 混合モデル（AR(1) + 正弦波）
# "exponential_decay": 指数関数的減衰モデル
FLUCTUATION_MODEL = "ar1"  # ★変動モデルを選択★

# ===== AR(1)モデルパラメータ =====
MEAN_UTILIZATION: float = 0.4  # (根拠: ISPの一般的な運用マージン)
AR_COEFFICIENT: float = 0.95  # (根拠: ネットワークトラフィックの高い自己相関)
NOISE_VARIANCE: float = 0.000975  # (根拠: 上記2値から逆算した値)

# ===== AR(2)モデルパラメータ =====
AR1_COEFFICIENT: float = 0.7  # 1次係数
AR2_COEFFICIENT: float = 0.2  # 2次係数

# ===== ランダムウォークモデルパラメータ =====
RW_NOISE_VARIANCE: float = 0.001  # ノイズ分散

# ===== OU過程パラメータ =====
OU_MEAN_REVERSION_SPEED: float = 0.1  # 平均回帰速度

# ===== 正弦波モデルパラメータ =====
SINUSOIDAL_AMPLITUDE: float = 0.2  # 振幅
SINUSOIDAL_PERIOD: int = 50  # 周期（世代数）
SINUSOIDAL_NOISE_VARIANCE: float = 0.0001  # ノイズ分散

# ===== 階段関数モデルパラメータ =====
STEP_INTERVAL: int = 20  # レベル変更間隔（世代数）
STEP_SIZE: float = 0.1  # レベル変更幅
STEP_NOISE_VARIANCE: float = 0.0001  # ノイズ分散

# ===== GARCHモデルパラメータ =====
GARCH_ALPHA0: float = 0.0001
GARCH_ALPHA1: float = 0.1
GARCH_BETA: float = 0.85

# ===== マルコフ連鎖モデルパラメータ =====
MARKOV_STATES: List[float] = [0.2, 0.4, 0.6]  # 低/中/高利用率
# 遷移確率行列（行: 現在の状態、列: 次の状態）
MARKOV_TRANSITION_MATRIX: List[List[float]] = [
    [0.7, 0.2, 0.1],  # 低→低/中/高
    [0.2, 0.6, 0.2],  # 中→低/中/高
    [0.1, 0.2, 0.7],  # 高→低/中/高
]
MARKOV_NOISE_VARIANCE: float = 0.0001  # ノイズ分散

# ===== 混合モデルパラメータ =====
# AR(1)と正弦波のパラメータを使用

# ===== 指数関数的減衰モデルパラメータ =====
DECAY_RATE: float = 0.05  # 減衰率
DECAY_NOISE_VARIANCE: float = 0.0001  # ノイズ分散


def select_hub_edges(
    graph: nx.Graph, percentage: float = FLUCTUATION_PERCENTAGE
) -> List[Tuple[int, int]]:
    """
    ハブノード（隣接ノード数が多いノード）に接続するエッジを選択する（従来版）

    Args:
        graph: ネットワークグラフ
        percentage: 変動させるエッジの割合（0.0-1.0）

    Returns:
        変動対象となるエッジのリスト（双方向）
    """
    # 100%の場合は全エッジを返す
    if percentage >= 1.0:
        print("全エッジを変動対象として選択します。")
        all_edges = []
        for u, v in graph.edges():
            all_edges.append((u, v))
            all_edges.append((v, u))  # 双方向
        print(f"全{len(graph.edges())}ペアのエッジを変動対象として選択しました。")
        return all_edges

    print("ハブノード（次数の高いノード）を計算中...")

    # 各ノードの次数（隣接ノード数）を計算
    node_degrees = dict(graph.degree())

    # 次数の高い順にソート
    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

    # 上位percentageのノードを選択
    num_nodes = int(len(graph.nodes()) * percentage)
    hub_nodes = [node for node, _ in sorted_nodes[:num_nodes]]

    print(f"ハブノード選択: 上位{percentage*100:.0f}% ({len(hub_nodes)}ノード)")
    for node, degree in sorted_nodes[:num_nodes]:
        print(f"  ノード{node}: 次数{degree}")

    # ハブノードに接続するエッジを収集
    hub_edges = []
    for node in hub_nodes:
        for neighbor in graph.neighbors(node):
            hub_edges.append((node, neighbor))

    print(f"ハブノードに接続する{len(hub_edges)}エッジを変動対象として選択しました。")
    return hub_edges


def select_hub_edges_partial(
    graph: nx.Graph,
    hub_percentage: float = FLUCTUATION_PERCENTAGE,
    neighbor_edge_ratio: float = HUB_NEIGHBOR_EDGE_RATIO,
    selection_method: str = HUB_NEIGHBOR_SELECTION_METHOD,
) -> List[Tuple[int, int]]:
    """
    ハブノードの隣接エッジを部分的に選択する（柔軟版）

    Args:
        graph: ネットワークグラフ
        hub_percentage: ハブノードとして選択するノードの割合（0.0-1.0）
        neighbor_edge_ratio: ハブノードの隣接エッジのうち変動させる割合（0.0-1.0）
        selection_method: 隣接エッジの選択方法（"degree": 次数順, "random": ランダム）

    Returns:
        変動対象となるエッジのリスト（双方向）
    """
    print("柔軟なハブノード選択を実行中...")
    print(f"  ハブノード割合: {hub_percentage*100:.0f}%")
    print(f"  隣接エッジ変動割合: {neighbor_edge_ratio*100:.0f}%")
    print(f"  隣接エッジ選択方法: {selection_method}")

    # 各ノードの次数（隣接ノード数）を計算
    node_degrees = dict(graph.degree())

    # 次数の高い順にソート
    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

    # 上位hub_percentageのノードを選択
    num_hub_nodes = int(len(graph.nodes()) * hub_percentage)
    hub_nodes = [node for node, _ in sorted_nodes[:num_hub_nodes]]

    print(f"ハブノード選択: 上位{hub_percentage*100:.0f}% ({len(hub_nodes)}ノード)")
    for node, degree in sorted_nodes[:num_hub_nodes]:
        print(f"  ノード{node}: 次数{degree}")

    # ハブノードの隣接エッジを部分的に選択
    selected_edges = []
    total_neighbor_edges = 0
    selected_neighbor_edges = 0

    for hub_node in hub_nodes:
        neighbors = list(graph.neighbors(hub_node))
        total_neighbor_edges += len(neighbors)

        # 隣接エッジの選択
        if selection_method == "degree":
            # 隣接ノードを次数順でソート（高い順）
            neighbor_degrees = [
                (neighbor, node_degrees[neighbor]) for neighbor in neighbors
            ]
            neighbor_degrees.sort(key=lambda x: x[1], reverse=True)
            sorted_neighbors = [neighbor for neighbor, _ in neighbor_degrees]
        elif selection_method == "random":
            # ランダムにシャッフル
            sorted_neighbors = neighbors.copy()
            random.shuffle(sorted_neighbors)
        else:
            raise ValueError(f"Invalid selection_method: {selection_method}")

        # 隣接エッジのうち指定割合を選択
        num_edges_to_select = max(1, int(len(neighbors) * neighbor_edge_ratio))
        selected_neighbors = sorted_neighbors[:num_edges_to_select]
        selected_neighbor_edges += len(selected_neighbors)

        # 選択されたエッジを追加
        for neighbor in selected_neighbors:
            selected_edges.append((hub_node, neighbor))

    print("隣接エッジ選択結果:")
    print(f"  総隣接エッジ数: {total_neighbor_edges}")
    print(f"  選択された隣接エッジ数: {selected_neighbor_edges}")
    print(f"  実際の選択割合: {selected_neighbor_edges/total_neighbor_edges*100:.1f}%")

    print(f"合計{len(selected_edges)}エッジを変動対象として選択しました。")
    return selected_edges


def select_hub_edges_random(
    graph: nx.Graph,
    hub_percentage: float = FLUCTUATION_PERCENTAGE,
    neighbor_edge_ratio: float = HUB_NEIGHBOR_EDGE_RATIO,
) -> List[Tuple[int, int]]:
    """
    ハブノードの隣接エッジをランダムに選択する

    Args:
        graph: ネットワークグラフ
        hub_percentage: ハブノードとして選択するノードの割合（0.0-1.0）
        neighbor_edge_ratio: ハブノードの隣接エッジのうち変動させる割合（0.0-1.0）

    Returns:
        変動対象となるエッジのリスト（双方向）
    """
    return select_hub_edges_partial(
        graph, hub_percentage, neighbor_edge_ratio, "random"
    )


def select_random_edges(
    graph: nx.Graph,
    percentage: float = FLUCTUATION_PERCENTAGE,
    timing: str = RANDOM_SELECTION_TIMING,
) -> List[Tuple[int, int]]:
    """
    完全ランダムでエッジを選択する

    Args:
        graph: ネットワークグラフ
        percentage: ランダムに選択するエッジの割合（0.0-1.0）
        timing: 選択タイミング（"fixed": 固定, "dynamic": 毎回ランダム）

    Returns:
        変動対象となるエッジのリスト（双方向）
    """
    # 100%の場合は全エッジを返す
    if percentage >= 1.0:
        print("全エッジを変動対象として選択します。")
        all_edges = []
        for u, v in graph.edges():
            all_edges.append((u, v))
            all_edges.append((v, u))  # 双方向
        print(f"全{len(graph.edges())}ペアのエッジを変動対象として選択しました。")
        return all_edges

    print("完全ランダム選択を実行中...")
    print(f"  選択割合: {percentage*100:.0f}%")
    print(f"  選択タイミング: {timing}")

    # 全エッジを取得
    all_edges = list(graph.edges())

    # 指定割合のエッジ数を計算
    num_edges_to_select = max(1, int(len(all_edges) * percentage))

    if timing == "fixed":
        # シミュレーション開始時に固定（シードを使用）
        print(f"  固定選択: {num_edges_to_select}エッジを選択")
        random.seed(42)  # 固定シードで再現可能
        selected_edges = random.sample(all_edges, num_edges_to_select)
        random.seed()  # シードをリセット
    elif timing == "dynamic":
        # 毎回ランダム
        print(f"  動的選択: {num_edges_to_select}エッジを選択")
        selected_edges = random.sample(all_edges, num_edges_to_select)
    else:
        raise ValueError(f"Invalid timing: {timing}")

    print("ランダム選択結果:")
    print(f"  総エッジ数: {len(all_edges)}")
    print(f"  選択されたエッジ数: {len(selected_edges)}")
    print(f"  実際の選択割合: {len(selected_edges)/len(all_edges)*100:.1f}%")

    # 選択されたエッジの双方向を追加
    fluctuating_edges = []
    for u, v in selected_edges:
        fluctuating_edges.append((u, v))
        fluctuating_edges.append((v, u))  # 双方向

    print(f"合計{len(fluctuating_edges)}エッジを変動対象として選択しました。")
    return fluctuating_edges


def select_high_betweenness_edges(
    graph: nx.Graph, percentage: float = FLUCTUATION_PERCENTAGE
) -> List[Tuple[int, int]]:
    """
    エッジ媒介中心性(Edge Betweenness Centrality)に基づいて
    上位percentageのエッジを選択する

    Args:
        graph: ネットワークグラフ
        percentage: 変動させるエッジの割合（0.0-1.0）

    Returns:
        変動対象となるエッジのリスト（双方向）
    """
    # 100%の場合は媒介中心性計算をスキップして全エッジを返す
    if percentage >= 1.0:
        print("全エッジを変動対象として選択します。")
        all_edges = []
        for u, v in graph.edges():
            all_edges.append((u, v))
            all_edges.append((v, u))  # 双方向
        print(f"全{len(graph.edges())}ペアのエッジを変動対象として選択しました。")
        return all_edges

    print("エッジ媒介中心性を計算中...")
    # NetworkXの関数でエッジ媒介中心性を計算
    # weight=None とすることでホップ数ベースの最短経路で計算される
    # normalize=True で 0~1 の範囲に正規化される
    edge_centrality: Dict[Tuple[int, int], float] = nx.edge_betweenness_centrality(
        graph, weight=None, normalized=True
    )
    print("計算完了。")

    # 中心性の値でエッジを降順ソート
    sorted_edge_items = sorted(
        edge_centrality.items(), key=lambda item: item[1], reverse=True
    )

    # 上位percentageのエッジを選択
    num_edges_to_select = int(len(graph.edges()) * percentage)  # 全エッジ数を基準にする
    selected_edges_directed = [
        edge for edge, centrality in sorted_edge_items[:num_edges_to_select]
    ]

    # 選択されたエッジの逆方向も追加（双方向で変動させるため）
    fluctuating_edges_set = set()
    num_pairs_selected = 0
    added_pairs = set()  # (u,v)と(v,u)のペアを管理

    for u, v in selected_edges_directed:
        # すでにペアとして追加済みでなければ追加
        pair = tuple(sorted((u, v)))
        if pair not in added_pairs:
            fluctuating_edges_set.add((u, v))
            if graph.has_edge(v, u):  # 無向グラフの場合や逆方向エッジが存在する場合
                fluctuating_edges_set.add((v, u))
            added_pairs.add(pair)
            num_pairs_selected += 1
            if num_pairs_selected >= num_edges_to_select:
                break  # 目的のペア数に達したら終了

    print(
        f"媒介中心性上位 約{percentage*100:.0f}% ({num_pairs_selected}ペア) のエッジを変動対象として選択しました。"
    )
    return list(fluctuating_edges_set)


def initialize_ar1_states(
    graph: nx.Graph, fluctuating_edges: List[Tuple[int, int]] | None = None
) -> Dict[Tuple[int, int], float]:
    """
    選択されたエッジのAR(1)モデルの初期利用率を設定する

    Args:
        graph: ネットワークグラフ
        fluctuating_edges: 変動対象となるエッジのリスト（Noneの場合は全エッジ）

    Returns:
        エッジのAR(1)状態辞書
    """
    edge_states = {}

    # 変動対象エッジが指定されていない場合は全エッジを対象とする
    if fluctuating_edges is None:
        target_edges = list(graph.edges())
    else:
        target_edges = fluctuating_edges

    print(f"AR(1)状態初期化: {len(target_edges)}エッジを対象とします")

    for u, v in target_edges:
        if not graph.has_edge(u, v):
            continue

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
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """
    AR(1)モデルによる帯域変動（選択されたエッジのみ）
    - BANDWIDTH_UPDATE_INTERVAL世代ごとにのみ更新

    Args:
        graph: ネットワークグラフ
        edge_states: エッジの状態辞書（Dict形式）
        generation: 現在の世代

    Returns:
        帯域が変更されたかどうか
    """
    # 更新間隔でない世代は変化なし
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        # エッジが存在しない場合はスキップ
        if not graph.has_edge(u, v):
            continue

        # 後方互換性: 古い形式（float）もサポート
        if isinstance(state, dict):
            current_utilization = state.get("utilization", MEAN_UTILIZATION)
        else:
            # 古い形式（float）の場合
            current_utilization = state
            state = {"utilization": current_utilization}
            edge_states[(u, v)] = state

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
        state["utilization"] = new_utilization

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


def initialize_fluctuation_states(
    graph: nx.Graph, fluctuating_edges: List[Tuple[int, int]] | None = None
) -> Dict[Tuple[int, int], Dict]:
    """
    選択されたエッジの変動モデルの初期状態を設定する（モデルに応じて）

    Args:
        graph: ネットワークグラフ
        fluctuating_edges: 変動対象となるエッジのリスト（Noneの場合は全エッジ）

    Returns:
        エッジの状態辞書（モデルに応じた状態を保存）
    """
    edge_states: Dict[Tuple[int, int], Dict] = {}

    # 変動対象エッジが指定されていない場合は全エッジを対象とする
    if fluctuating_edges is None:
        target_edges = list(graph.edges())
    else:
        target_edges = fluctuating_edges

    print(
        f"変動モデル '{FLUCTUATION_MODEL}' の状態初期化: {len(target_edges)}エッジを対象とします"
    )

    for u, v in target_edges:
        if not graph.has_edge(u, v):
            continue

        # モデルに応じた初期化
        if FLUCTUATION_MODEL == "ar1":
            # AR(1)モデル: 利用率のみ
            util_uv = random.uniform(0.3, 0.5)
            util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {"utilization": util_uv}
            edge_states[(v, u)] = {"utilization": util_vu}

        elif FLUCTUATION_MODEL == "ar2":
            # AR(2)モデル: 利用率と1つ前の利用率
            util_uv = random.uniform(0.3, 0.5)
            util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {
                "utilization": util_uv,
                "prev_utilization": util_uv,
            }
            edge_states[(v, u)] = {
                "utilization": util_vu,
                "prev_utilization": util_vu,
            }

        elif FLUCTUATION_MODEL == "random_walk":
            # ランダムウォーク: 利用率のみ
            util_uv = random.uniform(0.3, 0.5)
            util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {"utilization": util_uv}
            edge_states[(v, u)] = {"utilization": util_vu}

        elif FLUCTUATION_MODEL == "ou_process":
            # OU過程: 利用率のみ
            util_uv = random.uniform(0.3, 0.5)
            util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {"utilization": util_uv}
            edge_states[(v, u)] = {"utilization": util_vu}

        elif FLUCTUATION_MODEL == "sinusoidal":
            # 正弦波: 利用率と世代カウンタ
            util_uv = random.uniform(0.3, 0.5)
            util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {"utilization": util_uv, "generation": 0}
            edge_states[(v, u)] = {"utilization": util_vu, "generation": 0}

        elif FLUCTUATION_MODEL == "step_function":
            # 階段関数: 利用率とレベル変更カウンタ
            util_uv = random.uniform(0.3, 0.5)
            util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {
                "utilization": util_uv,
                "step_counter": 0,
            }
            edge_states[(v, u)] = {
                "utilization": util_vu,
                "step_counter": 0,
            }

        elif FLUCTUATION_MODEL == "garch":
            # GARCH: 利用率とボラティリティ
            util_uv = random.uniform(0.3, 0.5)
            util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {
                "utilization": util_uv,
                "volatility": math.sqrt(NOISE_VARIANCE),
                "prev_noise": 0.0,
            }
            edge_states[(v, u)] = {
                "utilization": util_vu,
                "volatility": math.sqrt(NOISE_VARIANCE),
                "prev_noise": 0.0,
            }

        elif FLUCTUATION_MODEL == "markov_chain":
            # マルコフ連鎖: 現在の状態インデックス
            state_idx_uv = random.randint(0, len(MARKOV_STATES) - 1)
            state_idx_vu = random.randint(0, len(MARKOV_STATES) - 1)
            edge_states[(u, v)] = {
                "state_index": state_idx_uv,
                "utilization": MARKOV_STATES[state_idx_uv],
            }
            edge_states[(v, u)] = {
                "state_index": state_idx_vu,
                "utilization": MARKOV_STATES[state_idx_vu],
            }

        elif FLUCTUATION_MODEL == "mixed":
            # 混合モデル: AR(1)と正弦波の両方
            util_uv = random.uniform(0.3, 0.5)
            util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {
                "utilization": util_uv,
                "generation": 0,
            }
            edge_states[(v, u)] = {
                "utilization": util_vu,
                "generation": 0,
            }

        elif FLUCTUATION_MODEL == "exponential_decay":
            # 指数関数的減衰: 利用率と初期値
            initial_util_uv = random.uniform(0.3, 0.5)
            initial_util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {
                "utilization": initial_util_uv,
                "initial_utilization": initial_util_uv,
            }
            edge_states[(v, u)] = {
                "utilization": initial_util_vu,
                "initial_utilization": initial_util_vu,
            }

        else:
            # デフォルト: AR(1)と同じ
            util_uv = random.uniform(0.3, 0.5)
            util_vu = random.uniform(0.3, 0.5)
            edge_states[(u, v)] = {"utilization": util_uv}
            edge_states[(v, u)] = {"utilization": util_vu}

        # 標準的な可用帯域計算: キャパシティ × (1 - 使用率)
        capacity = graph[u][v]["original_weight"]
        avg_util = 0.5 * (
            edge_states[(u, v)]["utilization"] + edge_states[(v, u)]["utilization"]
        )
        initial_available = int(round(capacity * (1.0 - avg_util)))
        # 10Mbps刻みに丸め
        initial_available = ((initial_available + 5) // 10) * 10
        graph[u][v]["weight"] = initial_available
        graph[u][v]["local_min_bandwidth"] = initial_available
        graph[u][v]["local_max_bandwidth"] = initial_available

    return edge_states


def update_available_bandwidth(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """
    選択された変動モデルに応じて帯域変動を更新する

    Args:
        graph: ネットワークグラフ
        edge_states: エッジの状態辞書
        generation: 現在の世代

    Returns:
        帯域が変更されたかどうか
    """
    if FLUCTUATION_MODEL == "ar1":
        return update_available_bandwidth_ar1(graph, edge_states, generation)
    elif FLUCTUATION_MODEL == "ar2":
        return update_available_bandwidth_ar2(graph, edge_states, generation)
    elif FLUCTUATION_MODEL == "random_walk":
        return update_available_bandwidth_random_walk(graph, edge_states, generation)
    elif FLUCTUATION_MODEL == "ou_process":
        return update_available_bandwidth_ou(graph, edge_states, generation)
    elif FLUCTUATION_MODEL == "sinusoidal":
        return update_available_bandwidth_sinusoidal(graph, edge_states, generation)
    elif FLUCTUATION_MODEL == "step_function":
        return update_available_bandwidth_step(graph, edge_states, generation)
    elif FLUCTUATION_MODEL == "garch":
        return update_available_bandwidth_garch(graph, edge_states, generation)
    elif FLUCTUATION_MODEL == "markov_chain":
        return update_available_bandwidth_markov(graph, edge_states, generation)
    elif FLUCTUATION_MODEL == "mixed":
        return update_available_bandwidth_mixed(graph, edge_states, generation)
    elif FLUCTUATION_MODEL == "exponential_decay":
        return update_available_bandwidth_decay(graph, edge_states, generation)
    else:
        # デフォルト: AR(1)
        return update_available_bandwidth_ar1(graph, edge_states, generation)


def update_available_bandwidth_ar2(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """AR(2)モデルによる帯域変動"""
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        if not graph.has_edge(u, v):
            continue

        current_util = state["utilization"]
        prev_util = state.get("prev_utilization", current_util)

        # AR(2)モデル: X(t) = c + φ₁*X(t-1) + φ₂*X(t-2) + ε(t)
        noise = random.gauss(0, math.sqrt(NOISE_VARIANCE))
        c = (1 - AR1_COEFFICIENT - AR2_COEFFICIENT) * MEAN_UTILIZATION

        new_utilization = (
            c + AR1_COEFFICIENT * current_util + AR2_COEFFICIENT * prev_util + noise
        )

        new_utilization = max(0.05, min(0.95, new_utilization))

        # 状態を更新
        state["prev_utilization"] = current_util
        state["utilization"] = new_utilization

        # 可用帯域を更新
        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        graph[u][v]["weight"] = available_bandwidth
        graph[u][v]["local_min_bandwidth"] = available_bandwidth
        graph[u][v]["local_max_bandwidth"] = available_bandwidth

    return bandwidth_changed


def update_available_bandwidth_random_walk(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """ランダムウォークモデルによる帯域変動"""
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        if not graph.has_edge(u, v):
            continue

        current_util = state["utilization"]

        # ランダムウォーク: X(t+1) = X(t) + ε(t)
        noise = random.gauss(0, math.sqrt(RW_NOISE_VARIANCE))
        new_utilization = current_util + noise

        new_utilization = max(0.05, min(0.95, new_utilization))

        state["utilization"] = new_utilization

        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        graph[u][v]["weight"] = available_bandwidth
        graph[u][v]["local_min_bandwidth"] = available_bandwidth
        graph[u][v]["local_max_bandwidth"] = available_bandwidth

    return bandwidth_changed


def update_available_bandwidth_ou(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """オルンシュタイン・ウーレンベック過程による帯域変動"""
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        if not graph.has_edge(u, v):
            continue

        current_util = state["utilization"]

        # OU過程: X(t+1) = X(t) + θ × (μ - X(t)) + ε(t)
        noise = random.gauss(0, math.sqrt(NOISE_VARIANCE))
        new_utilization = (
            current_util
            + OU_MEAN_REVERSION_SPEED * (MEAN_UTILIZATION - current_util)
            + noise
        )

        new_utilization = max(0.05, min(0.95, new_utilization))

        state["utilization"] = new_utilization

        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        graph[u][v]["weight"] = available_bandwidth
        graph[u][v]["local_min_bandwidth"] = available_bandwidth
        graph[u][v]["local_max_bandwidth"] = available_bandwidth

    return bandwidth_changed


def update_available_bandwidth_sinusoidal(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """正弦波モデルによる帯域変動"""
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        if not graph.has_edge(u, v):
            continue

        gen = state.get("generation", generation)

        # 正弦波: X(t) = μ + A × sin(2π × t / T) + ε(t)
        noise = random.gauss(0, math.sqrt(SINUSOIDAL_NOISE_VARIANCE))
        sinusoidal_component = SINUSOIDAL_AMPLITUDE * math.sin(
            2 * math.pi * gen / SINUSOIDAL_PERIOD
        )
        new_utilization = MEAN_UTILIZATION + sinusoidal_component + noise

        new_utilization = max(0.05, min(0.95, new_utilization))

        state["utilization"] = new_utilization
        state["generation"] = gen + 1

        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        graph[u][v]["weight"] = available_bandwidth
        graph[u][v]["local_min_bandwidth"] = available_bandwidth
        graph[u][v]["local_max_bandwidth"] = available_bandwidth

    return bandwidth_changed


def update_available_bandwidth_step(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """階段関数モデルによる帯域変動"""
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        if not graph.has_edge(u, v):
            continue

        current_util = state["utilization"]
        step_counter = state.get("step_counter", 0)

        # 一定間隔ごとにレベルを変更
        if step_counter >= STEP_INTERVAL:
            # レベルを変更（±STEP_SIZE）
            change = random.choice([-STEP_SIZE, STEP_SIZE])
            new_base_util = current_util + change
            new_base_util = max(0.1, min(0.9, new_base_util))
            state["utilization"] = new_base_util
            state["step_counter"] = 0
        else:
            state["step_counter"] = step_counter + 1

        # ノイズを追加
        noise = random.gauss(0, math.sqrt(STEP_NOISE_VARIANCE))
        new_utilization = state["utilization"] + noise

        new_utilization = max(0.05, min(0.95, new_utilization))

        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        graph[u][v]["weight"] = available_bandwidth
        graph[u][v]["local_min_bandwidth"] = available_bandwidth
        graph[u][v]["local_max_bandwidth"] = available_bandwidth

    return bandwidth_changed


def update_available_bandwidth_garch(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """GARCHモデルによる帯域変動"""
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        if not graph.has_edge(u, v):
            continue

        volatility = state.get("volatility", math.sqrt(NOISE_VARIANCE))
        prev_noise = state.get("prev_noise", 0.0)

        # GARCH(1,1): σ²(t+1) = α₀ + α₁ × ε²(t) + β × σ²(t)
        new_volatility_sq = (
            GARCH_ALPHA0 + GARCH_ALPHA1 * (prev_noise**2) + GARCH_BETA * (volatility**2)
        )
        new_volatility = math.sqrt(max(0.0001, new_volatility_sq))

        # ノイズを生成（ボラティリティを考慮）
        noise = random.gauss(0, new_volatility)

        # 利用率を更新
        new_utilization = MEAN_UTILIZATION + noise
        new_utilization = max(0.05, min(0.95, new_utilization))

        # 状態を更新
        state["utilization"] = new_utilization
        state["volatility"] = new_volatility
        state["prev_noise"] = noise

        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        graph[u][v]["weight"] = available_bandwidth
        graph[u][v]["local_min_bandwidth"] = available_bandwidth
        graph[u][v]["local_max_bandwidth"] = available_bandwidth

    return bandwidth_changed


def update_available_bandwidth_markov(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """マルコフ連鎖モデルによる帯域変動"""
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        if not graph.has_edge(u, v):
            continue

        current_state_idx = state["state_index"]

        # 遷移確率行列に基づいて次の状態を決定
        transition_probs = MARKOV_TRANSITION_MATRIX[current_state_idx]
        rand = random.random()
        cumulative = 0.0
        next_state_idx = current_state_idx

        for i, prob in enumerate(transition_probs):
            cumulative += prob
            if rand <= cumulative:
                next_state_idx = i
                break

        # 状態に対応する利用率を取得
        base_utilization = MARKOV_STATES[next_state_idx]

        # ノイズを追加
        noise = random.gauss(0, math.sqrt(MARKOV_NOISE_VARIANCE))
        new_utilization = base_utilization + noise
        new_utilization = max(0.05, min(0.95, new_utilization))

        # 状態を更新
        state["state_index"] = next_state_idx
        state["utilization"] = new_utilization

        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        graph[u][v]["weight"] = available_bandwidth
        graph[u][v]["local_min_bandwidth"] = available_bandwidth
        graph[u][v]["local_max_bandwidth"] = available_bandwidth

    return bandwidth_changed


def update_available_bandwidth_mixed(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """混合モデル（AR(1) + 正弦波）による帯域変動"""
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        if not graph.has_edge(u, v):
            continue

        current_util = state["utilization"]
        gen = state.get("generation", generation)

        # AR(1)成分
        noise_ar = random.gauss(0, math.sqrt(NOISE_VARIANCE))
        ar_component = (
            (1 - AR_COEFFICIENT) * MEAN_UTILIZATION
            + AR_COEFFICIENT * current_util
            + noise_ar
        )

        # 正弦波成分
        sinusoidal_component = SINUSOIDAL_AMPLITUDE * math.sin(
            2 * math.pi * gen / SINUSOIDAL_PERIOD
        )

        # 混合
        new_utilization = ar_component + 0.3 * sinusoidal_component
        new_utilization = max(0.05, min(0.95, new_utilization))

        state["utilization"] = new_utilization
        state["generation"] = gen + 1

        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        graph[u][v]["weight"] = available_bandwidth
        graph[u][v]["local_min_bandwidth"] = available_bandwidth
        graph[u][v]["local_max_bandwidth"] = available_bandwidth

    return bandwidth_changed


def update_available_bandwidth_decay(
    graph: nx.Graph, edge_states: Dict[Tuple[int, int], Dict], generation: int
) -> bool:
    """指数関数的減衰モデルによる帯域変動"""
    if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
        return False

    bandwidth_changed = False

    for (u, v), state in edge_states.items():
        if not graph.has_edge(u, v):
            continue

        initial_util = state.get("initial_utilization", MEAN_UTILIZATION)

        # 指数関数的減衰: X(t) = μ + (X₀ - μ) × exp(-λ × t) + ε(t)
        noise = random.gauss(0, math.sqrt(DECAY_NOISE_VARIANCE))
        decay_component = (initial_util - MEAN_UTILIZATION) * math.exp(
            -DECAY_RATE * generation
        )
        new_utilization = MEAN_UTILIZATION + decay_component + noise

        new_utilization = max(0.05, min(0.95, new_utilization))

        state["utilization"] = new_utilization

        capacity = graph[u][v]["original_weight"]
        available_bandwidth = int(round(capacity * (1.0 - new_utilization)))
        available_bandwidth = ((available_bandwidth + 5) // 10) * 10

        if graph[u][v]["weight"] != available_bandwidth:
            bandwidth_changed = True

        graph[u][v]["weight"] = available_bandwidth
        graph[u][v]["local_min_bandwidth"] = available_bandwidth
        graph[u][v]["local_max_bandwidth"] = available_bandwidth

    return bandwidth_changed


def select_fluctuating_edges(graph: nx.Graph) -> List[Tuple[int, int]]:
    """
    設定に応じてエッジ選択方法を切り替える

    Args:
        graph: ネットワークグラフ

    Returns:
        変動対象となるエッジのリスト（双方向）
    """
    if EDGE_SELECTION_METHOD == "hub":
        return select_hub_edges(graph, FLUCTUATION_PERCENTAGE)
    elif EDGE_SELECTION_METHOD == "hub_partial":
        return select_hub_edges_partial(
            graph,
            FLUCTUATION_PERCENTAGE,
            HUB_NEIGHBOR_EDGE_RATIO,
            HUB_NEIGHBOR_SELECTION_METHOD,
        )
    elif EDGE_SELECTION_METHOD == "hub_random":
        return select_hub_edges_random(
            graph, FLUCTUATION_PERCENTAGE, HUB_NEIGHBOR_EDGE_RATIO
        )
    elif EDGE_SELECTION_METHOD == "random":
        return select_random_edges(
            graph, FLUCTUATION_PERCENTAGE, RANDOM_SELECTION_TIMING
        )
    elif EDGE_SELECTION_METHOD == "betweenness":
        return select_high_betweenness_edges(graph, FLUCTUATION_PERCENTAGE)
    else:
        raise ValueError(f"Invalid EDGE_SELECTION_METHOD: {EDGE_SELECTION_METHOD}")


def print_fluctuation_settings():
    """
    帯域変動設定を表示する
    """
    print("=" * 80)
    print("🚀 帯域変動設定")
    print(f"   変動モデル: {FLUCTUATION_MODEL}")
    print(f"   変動間隔: {BANDWIDTH_UPDATE_INTERVAL}世代ごと")
    print(f"   エッジ選択方法: {EDGE_SELECTION_METHOD}")

    if EDGE_SELECTION_METHOD == "hub":
        fluctuation_type = f"ハブノード({FLUCTUATION_PERCENTAGE*100:.0f}%)接続エッジ"
    elif EDGE_SELECTION_METHOD == "hub_partial":
        fluctuation_type = (
            f"ハブノード({FLUCTUATION_PERCENTAGE*100:.0f}%)の"
            f"隣接エッジ({HUB_NEIGHBOR_EDGE_RATIO*100:.0f}%)"
        )
    elif EDGE_SELECTION_METHOD == "hub_random":
        fluctuation_type = (
            f"ハブノード({FLUCTUATION_PERCENTAGE*100:.0f}%)の"
            f"隣接エッジ({HUB_NEIGHBOR_EDGE_RATIO*100:.0f}%)ランダム"
        )
    elif EDGE_SELECTION_METHOD == "random":
        fluctuation_type = f"完全ランダム({FLUCTUATION_PERCENTAGE*100:.0f}%)"
    elif EDGE_SELECTION_METHOD == "betweenness":
        fluctuation_type = f"媒介中心性上位({FLUCTUATION_PERCENTAGE*100:.0f}%)"
    else:
        fluctuation_type = "不明"

    if EDGE_SELECTION_METHOD in ["hub", "hub_partial", "hub_random"]:
        print(f"   ハブノード割合: {FLUCTUATION_PERCENTAGE*100:.0f}%")
        if EDGE_SELECTION_METHOD in ["hub_partial", "hub_random"]:
            print(f"   隣接エッジ変動割合: {HUB_NEIGHBOR_EDGE_RATIO*100:.0f}%")
        if EDGE_SELECTION_METHOD == "hub_partial":
            print(f"   隣接エッジ選択方法: {HUB_NEIGHBOR_SELECTION_METHOD}")
    elif EDGE_SELECTION_METHOD == "random":
        print(f"   ランダム選択割合: {FLUCTUATION_PERCENTAGE*100:.0f}%")
        print(f"   選択タイミング: {RANDOM_SELECTION_TIMING}")
    elif EDGE_SELECTION_METHOD == "betweenness":
        print(f"   媒介中心性選択割合: {FLUCTUATION_PERCENTAGE*100:.0f}%")

    strategy_type = (
        "完全ランダム選択"
        if EDGE_SELECTION_METHOD == "random"
        else (
            "柔軟なハブノード選択"
            if EDGE_SELECTION_METHOD in ["hub_partial", "hub_random"]
            else (
                "ハブノード選択"
                if EDGE_SELECTION_METHOD == "hub"
                else "媒介中心性ベース選択"
            )
        )
    )
    print(f"   変動戦略: {strategy_type}")
    print(f"   変動対象: {fluctuation_type}")
    print("=" * 80)
