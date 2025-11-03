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


def observe_all_edges_bandwidth(
    graph: nx.Graph,
    max_history_size: int = 100,
) -> None:
    """
    全エッジの現在の帯域幅を観測し、履歴に記録する

    研究コンペンディウム推奨: Phase 1 - 全エッジの継続的監視
    アリに依存せず、毎世代すべてのエッジの帯域を記録します。
    帯域変動更新（update_available_bandwidth_ar1）の直後に呼び出す。

    Args:
        graph: ネットワークグラフ
        max_history_size: 保持する履歴の最大サイズ（リングバッファ）

    Based on: 研究コンペンディウム Chapter 1.1
    パッシブ監視を主要データソースとして使用し、
    全エッジにおける帯域幅を各世代で監視する。
    """
    for u, v in graph.edges():
        current_bandwidth = graph[u][v]["weight"]
        observe_edge_bandwidth(graph, u, v, current_bandwidth, max_history_size)


def learn_bandwidth_pattern(
    graph: nx.Graph,
    u: int,
    v: int,
    min_samples: int = 10,
    use_wavelet: bool = False,
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
        use_wavelet: Trueの場合、ウェーブレット変換による周期性検出を使用
                     研究コンペンディウム推奨（Phase 2）

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

    # 周期的変動の検出
    # 研究コンペンディウム推奨: ウェーブレット変換を使用（Phase 2）
    if use_wavelet:
        periodicity = detect_periodicity_wavelet(history)
    else:
        periodicity = detect_periodicity(history)  # 既存の自己相関ベース

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


def calculate_adaptive_evaporation_rate(  # noqa: C901
    graph: nx.Graph,
    u: int,
    v: int,
    base_rate: float = 1.0,
    use_prediction_variability: bool = True,
    prediction_method: str = "ar1",
) -> float:
    """
    帯域変動パターンに基づく適応的揮発率を計算する

    学習した変動パターンに基づいて、揮発率の乗算係数を返します。
    この関数は `pheromone_update.py` の `apply_volatilization` から呼ばれます。

    研究コンペンディウム推奨: Phase 3 - 予測変動性に基づく適応型蒸発率
    ルール1（高変動）: 予測される帯域変動が高い場合 → 蒸発率ρを増加
    ルール2（低変動）: 予測される帯域が安定している場合 → 蒸発率ρを減少

    Args:
        graph: ネットワークグラフ
        u: 始点ノード
        v: 終点ノード
        base_rate: ベースとなる乗算係数（通常は1.0）
        use_prediction_variability: Trueの場合、予測変動性も考慮（研究コンペンディウム推奨）
        prediction_method: 予測に使用する手法（"ar1", "ma", "ema"）

    Returns:
        適応的揮発率の乗算係数
        - < 1.0: 揮発を促進（古い情報を早く忘れる）
        - > 1.0: 揮発を抑制（長期的な情報を保持）
        - = 1.0: 変化なし

    Based on: 研究コンペンディウム Chapter 3.2
    予測変動性への蒸発率の連動は、モデルの将来の状態に関する信頼度を
    ACOアルゴリズムの探索戦略に直接マッピングします。
    """
    pattern = graph[u][v].get("bandwidth_pattern")
    if pattern is None:
        # パターンを学習していない場合は変化なし
        return base_rate

    multiplier = base_rate

    # === 研究コンペンディウム推奨: 予測変動性に基づく調整（Phase 3）===
    if use_prediction_variability:
        history = graph[u][v].get("bandwidth_history", [])
        if len(history) >= 5:  # 予測には最低限の履歴が必要
            # 複数ステップ先を予測して変動性を計算
            predicted_values = []
            current_history = history.copy()  # コピーを作成（元の履歴を変更しない）
            for _ in range(5):  # 5ステップ先まで予測
                if len(current_history) >= 2:
                    predicted = predict_next_bandwidth(
                        current_history, method=prediction_method
                    )
                    predicted_values.append(predicted)
                    # 仮想的に履歴に追加（次の予測のために）
                    current_history = current_history + [predicted]

            if len(predicted_values) >= 3:
                # 予測値の分散を計算（予測変動性）
                pred_mean = sum(predicted_values) / len(predicted_values)
                pred_variance = sum(
                    (x - pred_mean) ** 2 for x in predicted_values
                ) / len(predicted_values)
                pred_cv = math.sqrt(pred_variance) / pred_mean if pred_mean > 0 else 0.0

                # ルール1（高変動）: 予測変動が高い → 蒸発率を増加（探索促進）
                # ルール2（低変動）: 予測変動が低い → 蒸発率を減少（活用促進）
                if pred_cv > 0.3:  # 高変動予測
                    multiplier *= 0.90  # 10%多く揮発（探索を促進）
                elif pred_cv > 0.15:  # 中高変動予測
                    multiplier *= 0.94  # 6%多く揮発
                elif pred_cv > 0.05:  # 中変動予測
                    multiplier *= 0.97  # 3%多く揮発
                else:  # 低変動予測（安定）
                    multiplier *= 1.02  # 2%少なく揮発（記憶を強化、活用促進）

    # === 変動係数（CV）に基づく調整（既存の手法）===
    # CVが高い = 変動が激しい = 古い情報を早く忘れる必要がある
    cv = pattern.get("cv", 0.0)
    if cv > 0.3:  # 高変動環境（変動が激しい）
        multiplier *= 0.92  # 8%多く揮発（古い情報を早く忘れる）
    elif cv > 0.15:  # 中高変動環境
        multiplier *= 0.96  # 4%多く揮発
    elif cv > 0.05:  # 中変動環境
        multiplier *= 0.98  # 2%多く揮発
    else:  # 低変動環境（安定している）
        multiplier *= 1.01  # 1%少なく揮発（長期的な情報を保持）

    # === 周期的変動に基づく調整 ===
    # 次の低帯域時期が近い場合、その経路を選ばれにくくするため揮発を促進
    periodicity = pattern.get("periodicity")
    next_low_period = pattern.get("next_low_period")
    if periodicity is not None and next_low_period is not None:
        # 次の低帯域時期までの残り時間の割合を計算
        period_ratio = next_low_period / periodicity if periodicity > 0 else 1.0
        if period_ratio < 0.2:  # 周期の20%以内（非常に近い）
            multiplier *= 0.88  # 12%多く揮発
        elif period_ratio < 0.3:  # 周期の30%以内（近い）
            multiplier *= 0.92  # 8%多く揮発
        elif period_ratio < 0.5:  # 周期の50%以内（やや近い）
            multiplier *= 0.96  # 4%多く揮発

    # === トレンドに基づく調整 ===
    # 減少傾向のエッジは劣化しているため揮発を促進、増加傾向は改善しているため保持
    trend = pattern.get("trend", "stable")
    if trend == "decreasing":  # 減少傾向（劣化中）
        multiplier *= 0.94  # 6%多く揮発（劣化している経路を避ける）
    elif trend == "increasing":  # 増加傾向（改善中）
        multiplier *= 1.02  # 2%少なく揮発（改善している経路を保持）
    # "stable" の場合は調整なし

    # === AR(1)係数に基づく調整 ===
    # AR(1)係数が高い = 過去の値に強く依存 = 変動が予測可能 = 揮発を抑制
    ar_coeff = pattern.get("ar_coefficient", 0.0)
    if ar_coeff > 0.7:  # 高い自己相関（予測可能な変動）
        multiplier *= 1.01  # 1%少なく揮発（予測可能なので保持）
    elif ar_coeff < 0.3:  # 低い自己相関（予測困難な変動）
        multiplier *= 0.98  # 2%多く揮発（予測困難なので早く忘れる）

    # 乗算係数の範囲を制限（極端な値にならないように）
    multiplier = max(0.80, min(1.10, multiplier))

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


def detect_periodicity_wavelet(
    history: list[float], max_period: int = 50
) -> Optional[int]:
    """
    ウェーブレット変換による周期性検出

    研究コンペンディウム推奨: Phase 2 - ウェーブレット変換による周期性検出
    FFTではなくウェーブレット変換を使用することで、時間と周波数の両方で
    局在化した分析が可能になり、バースト検出に適している。

    Args:
        history: 時系列データ
        max_period: 検出する最大周期

    Returns:
        周期（観測回数単位）、Noneの場合は非周期的

    Based on: 研究コンペンディウム Chapter 2.3
    ウェーブレット変換は時間と周波数（スケール）の両方で局在化した
    「ウェーブレット」に分解する。これにより時間周波数分析が可能になり、
    過渡的または周期的なイベントがいつ発生したかを特定できる。

    Note: 現在は簡易実装（離散ウェーブレット変換の簡易版）
    将来的には PyWavelets などのライブラリを使用可能
    """
    if len(history) < max_period * 2:
        return None

    # 簡易的な離散ウェーブレット変換（Haarウェーブレットの簡易版）
    # 時間とスケール（周期）の両方を考慮した分析
    best_period = None
    best_strength = 0.0

    # 各周期候補について、ウェーブレット係数の分散を計算
    # 分散が高い = その周期で強いパターンが存在
    for period in range(2, min(max_period, len(history) // 2)):
        # Haarウェーブレットの簡易版: 差分を計算
        # 周期periodでのパターンの強度を評価
        strength = 0.0
        count = 0

        # 周期periodでの繰り返しパターンの強度を計算
        for i in range(period, len(history)):
            # 現在の値と周期前の値の差分（ウェーブレット係数の簡易版）
            diff = abs(history[i] - history[i - period])
            strength += diff * diff  # エネルギーの簡易版
            count += 1

        if count > 0:
            strength = strength / count  # 正規化

            # 周期が存在する場合、強度が閾値を超える
            if strength > best_strength and strength > 0.1:  # 閾値調整可能
                best_strength = strength
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


# ===== 帯域予測関数（複数の手法を提供）=====


def predict_next_bandwidth_ar1(history: list[float]) -> float:
    """
    AR(1)モデルによる1ステップ先予測

    時系列予測の基本的な手法。過去の値に基づいて次の値を予測する。

    Args:
        history: 時系列データ（帯域幅の観測値のリスト）

    Returns:
        予測される次の帯域幅
    """
    if len(history) < 2:
        return history[-1] if history else 0.0

    # AR(1)係数を推定
    ar_coeff = estimate_ar1_coefficient(history)

    # 平均値を計算
    mean = sum(history) / len(history)

    # AR(1)予測: y_{t+1} = mean + ar_coeff * (y_t - mean)
    # これは定常過程を仮定したAR(1)モデルの予測式
    last_value = history[-1]
    predicted = mean + ar_coeff * (last_value - mean)

    return max(0.0, predicted)  # 負の値は0にクリップ


def predict_next_bandwidth_ma(history: list[float], window: int = 5) -> float:
    """
    移動平均（Moving Average）による予測（最もシンプルな手法）

    最近の観測値の平均を予測値とする。

    Args:
        history: 時系列データ
        window: 使用する観測値の数（デフォルト: 5）

    Returns:
        予測される次の帯域幅
    """
    if not history:
        return 0.0

    # ウィンドウサイズを履歴長に制限
    window = min(window, len(history))

    # 最近の観測値の平均を計算
    recent_values = history[-window:]
    predicted = sum(recent_values) / len(recent_values)

    return max(0.0, predicted)  # 負の値は0にクリップ


def predict_next_bandwidth_ema(history: list[float], alpha: float = 0.3) -> float:
    """
    指数平滑法（Exponential Smoothing）による予測

    より最近の観測値に大きな重みを付ける手法。

    Args:
        history: 時系列データ
        alpha: 平滑化定数（0 < alpha < 1）。大きいほど最近の値に重みが大きい

    Returns:
        予測される次の帯域幅
    """
    if not history:
        return 0.0

    # alphaを有効範囲に制限
    alpha = max(0.0, min(1.0, alpha))

    # 指数平滑法の計算
    # EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
    ema = history[0]
    for value in history[1:]:
        ema = alpha * value + (1 - alpha) * ema

    return max(0.0, ema)  # 負の値は0にクリップ


def calculate_predictive_heuristic(
    graph: nx.Graph,
    u: int,
    v: int,
    prediction_method: str = "ar1",
    gamma: float = 1.0,
) -> float:
    """
    予測的ヒューリスティック値を計算する

    研究コンペンディウム推奨: Phase 3 - 予測的ヒューリスティック
    エッジ(i, j)の予測される将来の帯域幅に基づくヒューリスティック成分を計算

    Args:
        graph: ネットワークグラフ
        u: 始点ノード
        v: 終点ノード
        prediction_method: 予測に使用する手法（"ar1", "ma", "ema"）
        gamma: 予測ヒューリスティックの重み（研究コンペンディウムでは通常1.0）

    Returns:
        予測的ヒューリスティック値（予測される帯域幅）

    Based on: 研究コンペンディウム Chapter 3.3
    状態遷移確率P_ijは、τ_ij、η_distance(ij)、η_pred(ij)の関数
    """
    history = graph[u][v].get("bandwidth_history", [])
    if len(history) < 2:
        # 履歴が不十分な場合は現在の帯域幅を返す
        return graph[u][v]["weight"]

    # 予測される帯域幅を計算
    predicted = predict_next_bandwidth(history, method=prediction_method)

    # ヒューリスティック値として使用（通常は帯域幅の指数関数）
    return max(0.0, predicted) ** gamma


def predict_next_bandwidth(
    history: list[float], method: str = "ar1", **kwargs
) -> float:
    """
    予測手法を選択して帯域幅を予測する（統一インターフェース）

    Args:
        history: 時系列データ
        method: 予測手法（"ar1", "ma", "ema"）
        **kwargs: 各手法固有のパラメータ
            - method="ma": window (デフォルト: 5)
            - method="ema": alpha (デフォルト: 0.3)

    Returns:
        予測される次の帯域幅

    Examples:
        >>> history = [80, 85, 82, 88, 90]
        >>> predict_next_bandwidth(history, method="ar1")
        >>> predict_next_bandwidth(history, method="ma", window=3)
        >>> predict_next_bandwidth(history, method="ema", alpha=0.5)
    """
    if method == "ar1":
        return predict_next_bandwidth_ar1(history)
    elif method == "ma":
        window = kwargs.get("window", 5)
        return predict_next_bandwidth_ma(history, window)
    elif method == "ema":
        alpha = kwargs.get("alpha", 0.3)
        return predict_next_bandwidth_ema(history, alpha)
    else:
        raise ValueError(
            f"Unknown prediction method: {method}. " "Choose from 'ar1', 'ma', 'ema'"
        )


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


def update_patterns_for_all_edges(
    graph: nx.Graph,
    min_samples: int = 10,
    update_interval: int = 10,
    generation: int = 0,
    use_wavelet: bool = False,
) -> None:
    """
    グラフ内の全エッジ（または選択されたエッジ）に対してパターン学習を実行

    パフォーマンスを考慮して、全エッジではなく観測が十分にあるエッジのみを更新します。
    また、`update_interval` 世代ごとに更新することで計算コストを削減します。

    Args:
        graph: ネットワークグラフ
        min_samples: 学習に必要な最小サンプル数
        update_interval: パターン更新の間隔（世代数）
        generation: 現在の世代番号
        use_wavelet: Trueの場合、ウェーブレット変換による周期性検出を使用
                     研究コンペンディウム推奨（Phase 2）

    Based on: 研究コンペンディウム Chapter 2.3
    ウェーブレット変換による周期性検出を使用することで、時間と周波数の
    両方で局在化した分析が可能になり、バースト検出に適している。
    """
    if generation % update_interval != 0:
        return  # 更新間隔でない場合はスキップ

    updated_count = 0
    for u, v in graph.edges():
        # 観測履歴が存在し、十分なサンプルがある場合のみ学習
        if "bandwidth_history" in graph[u][v]:
            history = graph[u][v]["bandwidth_history"]
            if len(history) >= min_samples:
                learn_bandwidth_pattern(graph, u, v, min_samples, use_wavelet)
                updated_count += 1

    # デバッグ用（必要に応じてコメントアウト）
    # if updated_count > 0:
    #     print(f"世代 {generation}: {updated_count}個のエッジのパターンを更新しました")
