"""
帯域監視・学習モジュール

エッジの利用可能帯域を常時監視し、帯域変動パターンを学習します。
学習結果は `pheromone_update.py` の揮発処理で使用されます。
"""

import math
from typing import Optional

import networkx as nx  # type: ignore[import-untyped]


def observe_edge_bandwidth(
    graph: nx.Graph,
    u: int,
    v: int,
    current_bandwidth: float,
    max_history_size: int = 100,
) -> None:
    """
    エッジの帯域を観測し、履歴に記録する

    アリがエッジを通過するたびに、そのエッジの利用可能帯域を観測して
    時系列データとして保存します。

    Args:
        graph: ネットワークグラフ
        u: 始点ノード
        v: 終点ノード
        current_bandwidth: 現在の利用可能帯域
        max_history_size: 保持する履歴の最大サイズ（リングバッファ）
    """
    # エッジ属性の初期化
    if "bandwidth_history" not in graph[u][v]:
        graph[u][v]["bandwidth_history"] = []

    # 履歴に追加（リングバッファ：古いデータを削除）
    history = graph[u][v]["bandwidth_history"]
    history.append(current_bandwidth)
    if len(history) > max_history_size:
        history.pop(0)  # 最古のデータを削除


def learn_bandwidth_pattern(
    graph: nx.Graph,
    u: int,
    v: int,
    min_samples: int = 10,
) -> Optional[dict]:
    """
    エッジの帯域変動パターンを学習する

    観測された時系列データから、変動パターン（統計特性、周期性など）を
    推定して学習します。

    Args:
        graph: ネットワークグラフ
        u: 始点ノード
        v: 終点ノード
        min_samples: 学習に必要な最小サンプル数

    Returns:
        学習した変動パターンの辞書、または None（サンプル数不足の場合）
    """
    # 履歴を取得
    history = graph[u][v].get("bandwidth_history", [])
    if len(history) < min_samples:
        return None

    # 基本統計量を計算
    mean = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / len(history)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean if mean > 0 else 0.0  # 変動係数

    # 周期的変動の検出（簡易版：自己相関を計算）
    periodicity = detect_periodicity(history)

    # AR(1)係数の推定（簡易版）
    ar_coefficient = estimate_ar1_coefficient(history)

    # トレンドの検出
    trend = detect_trend(history)

    # 次の低帯域時期の予測（周期的変動が検出された場合）
    next_low_period = None
    if periodicity is not None:
        next_low_period = predict_next_low_period(history, periodicity)

    # パターンをエッジ属性に保存
    pattern = {
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "cv": cv,  # 変動係数
        "ar_coefficient": ar_coefficient,  # AR(1)係数
        "trend": trend,  # "increasing", "decreasing", "stable"
        "periodicity": periodicity,  # 周期（観測回数単位）、Noneの場合は非周期的
        "next_low_period": next_low_period,  # 次の低帯域時期までの観測回数
    }

    graph[u][v]["bandwidth_pattern"] = pattern
    return pattern


def calculate_adaptive_evaporation_rate(
    graph: nx.Graph,
    u: int,
    v: int,
    base_rate: float = 1.0,
) -> float:
    """
    帯域変動パターンに基づく適応的揮発率を計算する

    学習した変動パターンに基づいて、揮発率の乗算係数を返します。
    この関数は `pheromone_update.py` の `apply_volatilization` から呼ばれます。

    Args:
        graph: ネットワークグラフ
        u: 始点ノード
        v: 終点ノード
        base_rate: ベースとなる乗算係数（通常は1.0）

    Returns:
        適応的揮発率の乗算係数
        - < 1.0: 揮発を促進（古い情報を早く忘れる）
        - > 1.0: 揮発を抑制（長期的な情報を保持）
        - = 1.0: 変化なし
    """
    pattern = graph[u][v].get("bandwidth_pattern")
    if pattern is None:
        # パターンを学習していない場合は変化なし
        return base_rate

    multiplier = base_rate

    # 変動係数に基づく調整
    cv = pattern.get("cv", 0.0)
    if cv > 0.3:  # 高変動環境
        multiplier *= 0.95  # 5%多く揮発（古い情報を早く忘れる）
    elif cv > 0.1:  # 中変動環境
        multiplier *= 0.98  # 2%多く揮発
    else:  # 低変動環境
        multiplier *= 1.02  # 2%少なく揮発（長期的な情報を保持）

    # 周期的変動に基づく調整
    periodicity = pattern.get("periodicity")
    next_low_period = pattern.get("next_low_period")
    if periodicity is not None and next_low_period is not None:
        # 次の低帯域時期が近い場合は、揮発を促進してその経路を選ばれにくくする
        if next_low_period < periodicity * 0.3:  # 周期の30%以内
            multiplier *= 0.90  # 10%多く揮発
        elif next_low_period < periodicity * 0.5:  # 周期の50%以内
            multiplier *= 0.95  # 5%多く揮発

    # トレンドに基づく調整
    trend = pattern.get("trend", "stable")
    if trend == "decreasing":  # 減少傾向
        multiplier *= 0.95  # 揮発を促進（劣化している経路を避ける）
    elif trend == "increasing":  # 増加傾向
        multiplier *= 1.01  # 揮発を抑制（改善している経路を保持）

    return multiplier


# ===== 内部関数（変動パターンの検出）=====


def detect_periodicity(history: list[float], max_period: int = 50) -> Optional[int]:
    """
    時系列データから周期性を検出する（簡易版：自己相関を使用）

    Args:
        history: 時系列データ
        max_period: 検出する最大周期

    Returns:
        周期（観測回数単位）、Noneの場合は非周期的
    """
    if len(history) < max_period * 2:
        return None

    best_period = None
    best_correlation = 0.0

    for period in range(2, min(max_period, len(history) // 2)):
        # 自己相関を計算
        correlation = calculate_autocorrelation(history, period)
        if correlation > best_correlation and correlation > 0.5:  # 閾値
            best_correlation = correlation
            best_period = period

    return best_period


def calculate_autocorrelation(history: list[float], lag: int) -> float:
    """
    自己相関を計算する

    Args:
        history: 時系列データ
        lag: ラグ（周期候補）

    Returns:
        自己相関係数（-1.0 ～ 1.0）
    """
    if len(history) < lag * 2:
        return 0.0

    mean = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / len(history)

    if variance == 0:
        return 0.0

    # 自己共分散を計算
    covariance = 0.0
    for i in range(len(history) - lag):
        covariance += (history[i] - mean) * (history[i + lag] - mean)
    covariance /= len(history) - lag

    # 自己相関係数 = 自己共分散 / 分散
    return covariance / variance


def estimate_ar1_coefficient(history: list[float]) -> float:
    """
    AR(1)係数を推定する（簡易版：最小二乗法）

    Args:
        history: 時系列データ

    Returns:
        AR(1)係数（-1.0 ～ 1.0）
    """
    if len(history) < 2:
        return 0.0

    # y_t = a * y_{t-1} + e_t の形で回帰
    X = history[:-1]  # y_{t-1}
    Y = history[1:]  # y_t

    mean_X = sum(X) / len(X)
    mean_Y = sum(Y) / len(Y)

    numerator = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(len(X)))
    denominator = sum((X[i] - mean_X) ** 2 for i in range(len(X)))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def detect_trend(history: list[float]) -> str:
    """
    時系列データのトレンドを検出する

    Args:
        history: 時系列データ

    Returns:
        "increasing", "decreasing", "stable"
    """
    if len(history) < 3:
        return "stable"

    # 線形回帰で傾きを計算
    n = len(history)
    x_mean = (n - 1) / 2
    y_mean = sum(history) / n

    numerator = sum((i - x_mean) * (history[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return "stable"

    slope = numerator / denominator

    # 閾値で判定
    threshold = (max(history) - min(history)) / n * 0.1
    if slope > threshold:
        return "increasing"
    elif slope < -threshold:
        return "decreasing"
    else:
        return "stable"


def predict_next_low_period(history: list[float], periodicity: int) -> Optional[int]:
    """
    周期的変動が検出された場合、次の低帯域時期を予測する

    Args:
        history: 時系列データ
        periodicity: 検出された周期

    Returns:
        次の低帯域時期までの観測回数、または None
    """
    if len(history) < periodicity:
        return None

    # 最近の周期内での最小値を探す
    recent_period = history[-periodicity:]
    min_index_in_period = recent_period.index(min(recent_period))

    # 次の低帯域時期までの観測回数を計算
    next_low_period = periodicity - min_index_in_period

    return next_low_period
