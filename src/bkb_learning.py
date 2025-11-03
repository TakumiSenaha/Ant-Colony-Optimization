"""
BKB（Best Known Bottleneck）学習モジュール

RFC 6298準拠の統計的BKB学習手法を提供します。
このモジュールは、複数のACO実装で共通して使用できます。
"""

import math
from typing import Optional

import networkx as nx  # type: ignore[import-untyped]

# ===== デフォルトパラメータ =====
# これらは呼び出し側でオーバーライド可能
DEFAULT_BKB_MEAN_ALPHA = 1 / 8  # SRTTの学習率 (0.125) - RFC 6298標準
DEFAULT_BKB_VAR_BETA = 1 / 4  # RTTVARの学習率 (0.25) - RFC 6298標準
DEFAULT_BKB_CONFIDENCE_K = 1.0  # 信頼区間幅の係数（平均 - K*分散）
DEFAULT_ACHIEVEMENT_BONUS_BASE = 1.5  # 基本の報酬ボーナス係数
DEFAULT_ACHIEVEMENT_BONUS_MAX = 3.0  # ボーナスの最大値（静的環境で一点集中）
DEFAULT_CONFIDENCE_SCALING = 2.0  # 確信度に基づくボーナススケーリング係数
DEFAULT_PENALTY_FACTOR = 0.5  # BKB「信頼下限」を下回るエッジへのペナルティ


class BKBLearningConfig:
    """BKB学習のパラメータ設定クラス"""

    def __init__(
        self,
        mean_alpha: float = DEFAULT_BKB_MEAN_ALPHA,
        var_beta: float = DEFAULT_BKB_VAR_BETA,
        confidence_k: float = DEFAULT_BKB_CONFIDENCE_K,
        achievement_bonus_base: float = DEFAULT_ACHIEVEMENT_BONUS_BASE,
        achievement_bonus_max: float = DEFAULT_ACHIEVEMENT_BONUS_MAX,
        confidence_scaling: float = DEFAULT_CONFIDENCE_SCALING,
        penalty_factor: float = DEFAULT_PENALTY_FACTOR,
        use_confidence_based_bonus: bool = True,
    ):
        """
        Args:
            mean_alpha: 平均値の学習率（RFC 6298のα）
            var_beta: 分散の学習率（RFC 6298のβ）
            confidence_k: 信頼区間幅の係数
            achievement_bonus_base: 基本の報酬ボーナス係数
            achievement_bonus_max: ボーナスの最大値
            confidence_scaling: 確信度スケーリング係数
            penalty_factor: ペナルティ係数
            use_confidence_based_bonus: 確信度ベースのボーナスを使用するか
        """
        self.mean_alpha = mean_alpha
        self.var_beta = var_beta
        self.confidence_k = confidence_k
        self.achievement_bonus_base = achievement_bonus_base
        self.achievement_bonus_max = achievement_bonus_max
        self.confidence_scaling = confidence_scaling
        self.penalty_factor = penalty_factor
        self.use_confidence_based_bonus = use_confidence_based_bonus


def update_node_bkb_statistics(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    config: Optional[BKBLearningConfig] = None,
) -> None:
    """
    ノードのBKB統計（平均・分散）を更新する（RFC 6298準拠）

    Args:
        graph: ネットワークグラフ
        node: 更新対象のノード
        bottleneck: 新しく観測されたボトルネック帯域
        config: BKB学習の設定（Noneの場合はデフォルト）
    """
    if config is None:
        config = BKBLearningConfig()

    mean_prev = graph.nodes[node].get("ema_bkb")
    var_prev = graph.nodes[node].get("ema_bkb_var", 0.0)

    if mean_prev is None:
        # 最初のサンプル (Karn's Algorithm)
        mean_new = float(bottleneck)
        var_new = float(bottleneck) / 2.0  # TCPのRTO初期値計算に準拠
    else:
        # 2回目以降 (RFC 6298)
        # 信頼度（ばらつき）の更新 (RTTVARの計算)
        deviation = abs(bottleneck - mean_prev)
        var_new = (1 - config.var_beta) * var_prev + config.var_beta * deviation

        # 平均値の更新 (SRTTの計算)
        mean_new = (1 - config.mean_alpha) * mean_prev + config.mean_alpha * bottleneck

    graph.nodes[node]["ema_bkb"] = mean_new
    graph.nodes[node]["ema_bkb_var"] = var_new

    # 互換維持：古いBKB最大値も（平均値で）更新しておく
    graph.nodes[node]["best_known_bottleneck"] = max(
        graph.nodes[node].get("best_known_bottleneck", 0), int(mean_new)
    )


def calculate_achievement_bonus_simple(
    bottleneck: float, node_mean: float, bonus: float = DEFAULT_ACHIEVEMENT_BONUS_BASE
) -> float:
    """
    シンプルな功績ボーナス計算（従来手法）

    Args:
        bottleneck: アントが発見したボトルネック帯域
        node_mean: ノードの平均BKB
        bonus: ボーナス係数

    Returns:
        ボーナス係数（1.0 または bonus）
    """
    if bottleneck > node_mean:
        return bonus
    return 1.0


def calculate_achievement_bonus_confidence_based(
    bottleneck: float,
    node_mean: float,
    node_var: float,
    config: Optional[BKBLearningConfig] = None,
) -> float:
    """
    確信度ベースの功績ボーナスを計算（改良版 v2）

    **改良ポイント（v2）**：
    - 基本ボーナス（achievement_bonus_base）を常に保証
    - 確信度が高い場合は追加ボーナスを付与
    - 初期段階でもシンプル手法並みの収束速度を実現
    - 静的環境では確信度が上がり、さらに強力なボーナスに

    戦略:
    - 初期段階（低確信度）：基本ボーナス（1.5x）で素早く収束
    - 静的環境（高確信度）：追加ボーナスで一点集中（最大3.0x）
    - 動的環境（低確信度維持）：基本ボーナスで柔軟な探索

    Args:
        bottleneck: アントが発見したボトルネック帯域
        node_mean: ノードの平均BKB
        node_var: ノードのBKB分散
        config: BKB学習の設定（Noneの場合はデフォルト）

    Returns:
        ボーナス係数（1.0 または achievement_bonus_base ~ achievement_bonus_max）
    """
    if config is None:
        config = BKBLearningConfig()

    # 基本的な功績判定：平均以下ならボーナスなし
    if bottleneck <= node_mean:
        return 1.0

    # 平均超過の度合い（相対的な改善率）
    if node_mean > 0:
        excess_ratio = (bottleneck - node_mean) / node_mean
    else:
        excess_ratio = 1.0  # 初期状態

    # 確信度の計算（分散が小さいほど確信度が高い）
    # 分散が0に近い → 確信度 ≈ 1.0（静的環境）
    # 分散が大きい → 確信度 ≈ 0.0（動的環境）
    if node_mean > 0 and node_var > 0:
        # 変動係数（CV: Coefficient of Variation）の逆数を使用
        # CV = std / mean = sqrt(var) / mean
        # 確信度 = 1 / (1 + CV)
        cv = math.sqrt(node_var) / node_mean
        confidence = 1.0 / (1.0 + cv)
    else:
        confidence = 0.5  # デフォルト（中間的な確信度）

    # ★★★ 改良：基本ボーナスを保証 ★★★
    # 初期段階（低確信度）でもシンプル手法並みの収束速度を実現
    # 基本ボーナス：achievement_bonus_base（例：1.5）
    # 追加ボーナス：確信度が高いほど大きくなる
    base_bonus = config.achievement_bonus_base
    additional_bonus = confidence * excess_ratio * config.confidence_scaling

    # 最終ボーナス = 基本 + 追加
    bonus = base_bonus + additional_bonus

    # 最大値でクリップ
    return min(bonus, config.achievement_bonus_max)


def calculate_achievement_bonus(
    bottleneck: float,
    node_mean: float,
    node_var: float,
    config: Optional[BKBLearningConfig] = None,
) -> float:
    """
    功績ボーナスを計算（設定に応じて手法を切り替え）

    Args:
        bottleneck: アントが発見したボトルネック帯域
        node_mean: ノードの平均BKB
        node_var: ノードのBKB分散
        config: BKB学習の設定（Noneの場合はデフォルト）

    Returns:
        ボーナス係数
    """
    if config is None:
        config = BKBLearningConfig()

    if config.use_confidence_based_bonus:
        return calculate_achievement_bonus_confidence_based(
            bottleneck, node_mean, node_var, config
        )
    else:
        return calculate_achievement_bonus_simple(
            bottleneck, node_mean, config.achievement_bonus_base
        )


def calculate_confidence(node_mean: float, node_var: float) -> float:
    """
    ノードの確信度を計算

    Args:
        node_mean: ノードの平均BKB
        node_var: ノードのBKB分散

    Returns:
        確信度（0.0 ~ 1.0）
        - 1.0に近い：高い確信度（静的環境）
        - 0.0に近い：低い確信度（動的環境）
    """
    if node_mean and node_mean > 0 and node_var > 0:
        cv = math.sqrt(node_var) / node_mean
        confidence = 1.0 / (1.0 + cv)
    else:
        confidence = 0.0
    return confidence


def get_confidence_lower_bound(
    node_mean: float, node_var: float, k: float = DEFAULT_BKB_CONFIDENCE_K
) -> float:
    """
    信頼区間の下限を計算

    Args:
        node_mean: ノードの平均BKB
        node_var: ノードのBKB分散
        k: 信頼区間幅の係数

    Returns:
        信頼下限（mean - k * var）
    """
    return node_mean - k * node_var


def initialize_graph_nodes_for_bkb(graph: nx.Graph) -> None:
    """
    グラフの全ノードにBKB統計属性を初期化

    Args:
        graph: ネットワークグラフ
    """
    for node in graph.nodes():
        graph.nodes[node]["ema_bkb"] = None  # 平均（SRTT相当）
        graph.nodes[node]["ema_bkb_var"] = 0.0  # 分散（RTTVAR相当）
        graph.nodes[node]["best_known_bottleneck"] = 0  # 互換維持用


# ===== 単純BKB学習（最大値ベース）=====


def update_node_bkb_simple(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    evaporation_rate: float = 0.999,
) -> None:
    """
    ノードのBKBを単純な最大値で更新（従来手法）

    Args:
        graph: ネットワークグラフ
        node: 更新対象のノード
        bottleneck: 新しく観測されたボトルネック帯域
        evaporation_rate: 揮発率（0.999推奨）
    """
    current_bkb = graph.nodes[node].get("best_known_bottleneck", 0)
    graph.nodes[node]["best_known_bottleneck"] = max(current_bkb, int(bottleneck))


def evaporate_all_bkb(graph: nx.Graph, evaporation_rate: float = 0.999) -> None:
    """
    全ノードのBKB値を揮発させる（従来手法）

    Args:
        graph: ネットワークグラフ
        evaporation_rate: 揮発率（0.999推奨）
    """
    for node in graph.nodes():
        if "best_known_bottleneck" in graph.nodes[node]:
            graph.nodes[node]["best_known_bottleneck"] *= evaporation_rate


def evaporate_bkb_values(
    graph: nx.Graph,
    evaporation_rate: float = 0.999,
    use_int_cast: bool = False,
) -> None:
    """
    全ノードのBKB値を揮発させる（統一関数）

    ★★★ BKBに関わる忘却処理を統一管理 ★★★

    Args:
        graph: ネットワークグラフ
        evaporation_rate: 揮発率（0.999推奨）
        use_int_cast: Trueの場合、結果をint型に変換（時間窓学習など）
    """
    for node in graph.nodes():
        if "best_known_bottleneck" in graph.nodes[node]:
            new_value = graph.nodes[node]["best_known_bottleneck"] * evaporation_rate
            if use_int_cast:
                graph.nodes[node]["best_known_bottleneck"] = int(new_value)
            else:
                graph.nodes[node]["best_known_bottleneck"] = new_value


def initialize_graph_nodes_for_simple_bkb(graph: nx.Graph) -> None:
    """
    グラフの全ノードにシンプルなBKB属性を初期化

    Args:
        graph: ネットワークグラフ
    """
    for node in graph.nodes():
        graph.nodes[node]["best_known_bottleneck"] = 0


# ===== 二段階BKB学習（高速追従+安定性）=====


def update_node_bkb_two_phase(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    config: BKBLearningConfig = BKBLearningConfig(),
) -> None:
    """
    二段階BKB学習：短期記憶（高速）+ 長期記憶（安定）

    分散的に機能：各ノードが独立に動作し、環境全体の情報は不要

    戦略:
    - 短期記憶（α=0.5）：最近の変動を素早く捉える
    - 長期記憶（α=0.125）：安定した基準を保持
    - 実効BKB = max(短期, 長期)：両方の利点を活用

    Args:
        graph: ネットワークグラフ
        node: 更新対象のノード
        bottleneck: 新しく観測されたボトルネック帯域
        config: BKB学習の設定（mean_alphaを長期記憶の学習率として使用）
    """
    # ★★★ 短期記憶（高速学習：α=0.5） ★★★
    short_mean = graph.nodes[node].get("short_ema_bkb")
    if short_mean is None:
        short_mean = float(bottleneck)
    else:
        # 50%の重みで最新値を反映
        short_mean = 0.5 * short_mean + 0.5 * bottleneck

    # ★★★ 長期記憶（標準学習：configで指定） ★★★
    long_mean = graph.nodes[node].get("long_ema_bkb")
    if long_mean is None:
        long_mean = float(bottleneck)
    else:
        # configの学習率を使用（例：α=0.125）
        long_alpha = config.mean_alpha
        long_mean = (1 - long_alpha) * long_mean + long_alpha * bottleneck

    # ★★★ 実効的なBKB = 短期と長期の最大値 ★★★
    # これにより、素早い追従と安定性を両立
    effective_mean = max(short_mean, long_mean)

    # 分散は短期記憶から計算（変動をよく捉える）
    var_prev = graph.nodes[node].get("ema_bkb_var", 0.0)
    deviation = abs(bottleneck - short_mean)
    var_new = (1 - config.var_beta) * var_prev + config.var_beta * deviation

    # 保存
    graph.nodes[node]["short_ema_bkb"] = short_mean
    graph.nodes[node]["long_ema_bkb"] = long_mean
    graph.nodes[node]["ema_bkb"] = effective_mean  # フェロモン判定に使用
    graph.nodes[node]["ema_bkb_var"] = var_new

    # 互換維持
    graph.nodes[node]["best_known_bottleneck"] = max(
        graph.nodes[node].get("best_known_bottleneck", 0), int(effective_mean)
    )


# ===== 三段階BKB学習（超短期+短期+長期）=====
def update_node_bkb_three_phase(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    config: BKBLearningConfig = BKBLearningConfig(),
) -> None:
    """
    三段階BKB学習：超短期記憶（超高速）+ 短期記憶（高速）+ 長期記憶（安定）
    分散的に機能：各ノードが独立に動作し、環境全体の情報は不要
    戦略:
    - 超短期記憶（α=0.9）：最新1-2世代の変化を即座に反映
    - 短期記憶（α=0.5）：直近5-10世代を反映
    - 長期記憶（α=0.125）：安定した基準を保持
    - 実効BKB = max(超短期, 短期, 長期)：3つの利点を活用
    Args:
        graph: ネットワークグラフ
        node: 更新対象のノード
        bottleneck: 新しく観測されたボトルネック帯域
        config: BKB学習の設定（mean_alphaを長期記憶の学習率として使用）
    """
    # ★★★ 超短期記憶（超高速学習：α=0.95） ★★★
    ultra_short = graph.nodes[node].get("ultra_short_ema_bkb")
    if ultra_short is None:
        ultra_short = float(bottleneck)
    else:
        # 95%の重みで最新値を反映（最新1世代を極めて強く反映）
        ultra_short = 0.05 * ultra_short + 0.95 * bottleneck

    # ★★★ 短期記憶（高速学習：α=0.7） ★★★
    short_mean = graph.nodes[node].get("short_ema_bkb")
    if short_mean is None:
        short_mean = float(bottleneck)
    else:
        # 70%の重みで最新値を反映（より積極的）
        short_mean = 0.3 * short_mean + 0.7 * bottleneck

    # ★★★ 長期記憶（標準学習：configで指定） ★★★
    long_mean = graph.nodes[node].get("long_ema_bkb")
    if long_mean is None:
        long_mean = float(bottleneck)
    else:
        # configの学習率を使用（例：α=0.125）
        long_alpha = config.mean_alpha
        long_mean = (1 - long_alpha) * long_mean + long_alpha * bottleneck

    # ★★★ 実効的なBKB = 3つの最大値 ★★★
    # これにより、超高速追従、高速追従、安定性を全て活用
    effective_mean = max(ultra_short, short_mean, long_mean)

    # 分散は超短期記憶から計算（最も変動をよく捉える）
    var_prev = graph.nodes[node].get("ema_bkb_var", 0.0)
    deviation = abs(bottleneck - ultra_short)
    var_new = (1 - config.var_beta) * var_prev + config.var_beta * deviation

    # 保存
    graph.nodes[node]["ultra_short_ema_bkb"] = ultra_short
    graph.nodes[node]["short_ema_bkb"] = short_mean
    graph.nodes[node]["long_ema_bkb"] = long_mean
    graph.nodes[node]["ema_bkb"] = effective_mean  # フェロモン判定に使用
    graph.nodes[node]["ema_bkb_var"] = var_new

    # 互換維持
    graph.nodes[node]["best_known_bottleneck"] = max(
        graph.nodes[node].get("best_known_bottleneck", 0), int(effective_mean)
    )


# ===== 時間区間ベース最大帯域学習 =====


def update_node_bkb_time_window_max(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    generation: int,
    time_window_size: int = 10,
    alpha: Optional[float] = None,  # 未使用（後方互換性のため残す）
) -> None:
    """
    時間区間ベースの最大帯域学習（リングバッファ版）

    ★★★ 核心: リングバッファで直近N個の観測値を記憶 ★★★

    戦略:
    - リングバッファで直近N個の観測値を保持
    - 新しい観測値が来たら追加
    - N個を超えたら古いものをpop(0)で削除（FIFO）
    - バッファ内の最大値をBKBとして使用

    Args:
        graph: ネットワークグラフ
        node: 更新対象のノード
        bottleneck: 新しく観測されたボトルネック帯域
        generation: 現在の世代番号（互換性のため保持、使用しない）
        time_window_size: リングバッファサイズ（記憶する観測値の個数）
        alpha: 未使用（後方互換性のため残す）
    """
    # リングバッファの初期化
    if "time_window_values" not in graph.nodes[node]:
        graph.nodes[node]["time_window_values"] = []

    window_values = graph.nodes[node]["time_window_values"]

    # 新しい観測値を追加
    window_values.append(bottleneck)

    # ★★★ リングバッファ: サイズを超えたら古いものを削除（FIFO）★★★
    while len(window_values) > time_window_size:
        window_values.pop(0)  # 古いものを先頭から削除

    # ★★★ バッファ内の最大値をBKBとして使用 ★★★
    time_window_max = max(window_values) if window_values else 0

    # BKB値を更新
    graph.nodes[node]["best_known_bottleneck"] = int(time_window_max)

    # デバッグ用：バッファの状態を保存
    graph.nodes[node]["time_window_max"] = time_window_max
    graph.nodes[node]["time_window_size"] = len(window_values)


def update_time_scale_max(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    scale_name: str,
    window_size: int,
    alpha: float,
) -> None:
    """
    単一時間スケールの最大値更新（内部関数）

    Args:
        graph: ネットワークグラフ
        node: 更新対象のノード
        bottleneck: 新しく観測されたボトルネック帯域
        scale_name: スケール名（"short_max", "medium_max", "long_max"等）
        window_size: 時間窓サイズ
        alpha: 指数平滑の学習率
    """
    # 時間窓の値を管理
    window_key = f"{scale_name}_values"
    if window_key not in graph.nodes[node]:
        graph.nodes[node][window_key] = []

    window_values = graph.nodes[node][window_key]
    window_values.append(bottleneck)

    # 窓サイズを超えたら古い値を削除
    if len(window_values) > window_size:
        window_values.pop(0)

    # 時間窓内の最大値
    time_window_max = max(window_values)

    # 指数平滑で古い最大記憶を更新
    old_max = graph.nodes[node].get(scale_name, 0)
    new_max = (1 - alpha) * old_max + alpha * time_window_max

    graph.nodes[node][scale_name] = int(new_max)


def update_node_bkb_multi_scale_max(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    short_window: int = 5,
    medium_window: int = 20,
    long_window: int = 100,
    short_alpha: float = 0.5,
    medium_alpha: float = 0.2,
    long_alpha: float = 0.1,
) -> None:
    """
    複数時間スケールの最大帯域学習

    戦略:
    - 短期（5世代）、中期（20世代）、長期（100世代）の最大値
    - 各スケールで指数平滑
    - 実効最大値 = max(短期, 中期, 長期)

    Args:
        graph: ネットワークグラフ
        node: 更新対象のノード
        bottleneck: 新しく観測されたボトルネック帯域
        short_window: 短期時間窓サイズ
        medium_window: 中期時間窓サイズ
        long_window: 長期時間窓サイズ
        short_alpha: 短期指数平滑学習率
        medium_alpha: 中期指数平滑学習率
        long_alpha: 長期指数平滑学習率
    """
    # 短期最大値（5世代）
    update_time_scale_max(
        graph, node, bottleneck, "short_max", short_window, short_alpha
    )

    # 中期最大値（20世代）
    update_time_scale_max(
        graph, node, bottleneck, "medium_max", medium_window, medium_alpha
    )

    # 長期最大値（100世代）
    update_time_scale_max(graph, node, bottleneck, "long_max", long_window, long_alpha)

    # 実効最大値
    short_max = graph.nodes[node].get("short_max", 0)
    medium_max = graph.nodes[node].get("medium_max", 0)
    long_max = graph.nodes[node].get("long_max", 0)

    effective_max = max(short_max, medium_max, long_max)
    graph.nodes[node]["best_known_bottleneck"] = int(effective_max)

    # デバッグ用：各スケールの値を保存
    graph.nodes[node]["effective_max"] = effective_max
