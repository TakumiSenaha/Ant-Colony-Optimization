# BKB 学習手法の改善案

現在の手法（忘却率 + 保存世代範囲）以外の改善案をまとめます。

## 現在の手法の課題

1. **固定忘却率**: 環境の変動に関係なく一律に忘却
2. **固定時間窓**: 記憶範囲が固定的
3. **最大値のみ**: 時間窓内の最大値のみを使用（情報損失）

---

## 提案 1: 重み付き時間窓最大値（Weighted Time-Window Max）

### コンセプト

新しい観測値ほど大きな重みを持つ時間窓内の最大値を計算

### 実装イメージ

```python
def update_node_bkb_weighted_time_window_max(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    generation: int,
    time_window_size: int = 10,
) -> None:
    # 時間窓内の観測値を保持
    window_values = [(value, gen) for value, gen in ...]

    # 重み付き最大値: 新しいほど重み大
    weighted_max = 0
    for value, gen in window_values:
        age = generation - gen
        weight = 1.0 / (1.0 + age)  # 古いほど重みが小さい
        weighted_max = max(weighted_max, value * weight)

    graph.nodes[node]["best_known_bottleneck"] = int(weighted_max)
```

### 利点

- 最新の観測値を重視
- 過去の情報も活用
- 分散的（ノード独立）

### 欠点

- 重みの設計が難しい
- 計算コストがやや高い

---

## 提案 2: 適応的忘却率（Adaptive Evaporation Rate）

### コンセプト

帯域の変動が大きい時は早く忘れ、小さい時は長く記憶

### 実装イメージ

```python
def evaporate_bkb_adaptive(
    graph: nx.Graph,
    node: int,
    base_evaporation_rate: float = 0.999,
) -> None:
    # 最近の観測値の変動係数（CV）を計算
    recent_values = graph.nodes[node].get("recent_observations", [])
    if len(recent_values) < 3:
        cv = 0.0
    else:
        mean_val = sum(recent_values) / len(recent_values)
        std_val = math.sqrt(sum((x - mean_val)**2) / len(recent_values))
        cv = std_val / mean_val if mean_val > 0 else 0.0

    # 変動が大きい → より早く忘れる（揮発率を下げる）
    # 変動が小さい → より長く記憶（揮発率を上げる）
    if cv > 0.3:  # 高変動
        evaporation_rate = base_evaporation_rate * 0.95  # 5%多く揮発
    elif cv > 0.1:  # 中変動
        evaporation_rate = base_evaporation_rate * 0.98  # 2%多く揮発
    else:  # 低変動
        evaporation_rate = base_evaporation_rate * 1.02  # 2%少なく揮発

    graph.nodes[node]["best_known_bottleneck"] *= evaporation_rate
```

### 利点

- 環境に適応
- 静的環境では長く記憶、動的環境では早く更新
- 分散的（ノード独立）

### 欠点

- 変動の計算に観測値履歴が必要
- パラメータ調整が複雑

---

## 提案 3: 移動最大値の指数平滑（Exponential Smoothing of Moving Max）

### コンセプト

時間窓内の最大値を過去の最大値と指数平滑して更新

### 実装イメージ

```python
def update_node_bkb_exp_smooth_max(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    generation: int,
    time_window_size: int = 10,
    alpha: float = 0.3,  # 平滑化係数
) -> None:
    # 時間窓内の最大値
    window_values = graph.nodes[node].get("time_window_values", [])
    window_max = max(window_values) if window_values else 0

    # 過去の平滑化された最大値
    old_smooth_max = graph.nodes[node].get("smooth_max_bkb", 0)

    # 指数平滑: 新しい最大値ほど重視
    new_smooth_max = (1 - alpha) * old_smooth_max + alpha * window_max

    graph.nodes[node]["smooth_max_bkb"] = new_smooth_max
    graph.nodes[node]["best_known_bottleneck"] = int(new_smooth_max)
```

### 利点

- ノイズに強い（平滑化効果）
- 計算がシンプル
- 過去の最大値も考慮

### 欠点

- 平滑化により反応が遅れる可能性
- α の調整が必要

---

## 提案 4: パーセンタイルベース（Percentile-Based）

### コンセプト

最大値ではなく、時間窓内の上位 N%（例: 95 パーセンタイル）を使用

### 実装イメージ

```python
def update_node_bkb_percentile(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    generation: int,
    time_window_size: int = 10,
    percentile: float = 0.95,  # 95パーセンタイル
) -> None:
    window_values = graph.nodes[node].get("time_window_values", [])

    if not window_values:
        bkb = bottleneck
    else:
        sorted_values = sorted(window_values, reverse=True)
        index = int(len(sorted_values) * (1 - percentile))
        bkb = sorted_values[index]

    graph.nodes[node]["best_known_bottleneck"] = int(bkb)
```

### 利点

- 外れ値（ノイズ）に強い
- より保守的な推定
- 実装が簡単

### 欠点

- 最大値より低い値を採用するため、最適化が控えめになる可能性

---

## 提案 5: 変動検知ベース適応更新（Change Detection Adaptive Update）

### コンセプト

最近の変動が大きい時だけ、より積極的に更新

### 実装イメージ

```python
def update_node_bkb_change_detection(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    generation: int,
    time_window_size: int = 10,
    change_threshold: float = 0.2,  # 20%以上の変化で検知
) -> None:
    current_bkb = graph.nodes[node].get("best_known_bottleneck", 0)

    # 最近の観測値から変動を検知
    recent_values = graph.nodes[node].get("recent_observations", [])

    if len(recent_values) >= 3:
        # 最新値と平均値の差分を計算
        recent_mean = sum(recent_values[-3:]) / 3
        change_ratio = abs(bottleneck - recent_mean) / recent_mean if recent_mean > 0 else 0

        if change_ratio > change_threshold:
            # 大きな変動検知 → より積極的に更新
            graph.nodes[node]["best_known_bottleneck"] = max(
                current_bkb,
                int(bottleneck * 0.9)  # 少し控えめに
            )
        else:
            # 小さな変動 → 従来通り最大値
            graph.nodes[node]["best_known_bottleneck"] = max(
                current_bkb,
                int(bottleneck)
            )
    else:
        # 初期段階は従来通り
        graph.nodes[node]["best_known_bottleneck"] = max(
            current_bkb,
            int(bottleneck)
        )
```

### 利点

- 変動に敏感に反応
- ノイズには強い
- 分散的（ノード独立）

### 欠点

- 閾値の調整が必要
- 検知に観測値履歴が必要

---

## 提案 6: 階層的時間窓（Hierarchical Time Window）

### コンセプト

短期・中期・長期の複数時間窓を組み合わせ、適応的に選択

### 実装イメージ

```python
def update_node_bkb_hierarchical_window(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    generation: int,
    short_window: int = 5,
    medium_window: int = 20,
    long_window: int = 100,
) -> None:
    # 各時間窓の最大値を計算
    short_max = max(window_values[-short_window:]) if len(window_values) >= short_window else 0
    medium_max = max(window_values[-medium_window:]) if len(window_values) >= medium_window else 0
    long_max = max(window_values) if window_values else 0

    # 最近の変動を検知
    if 最近の変動が大きい:
        bkb = max(short_max, medium_max)  # 短期重視
    else:
        bkb = max(medium_max, long_max)  # 長期重視

    graph.nodes[node]["best_known_bottleneck"] = int(bkb)
```

### 利点

- 複数の時間スケールを活用
- 環境に適応
- 既存の multi-scale 手法の拡張

### 欠点

- 実装が複雑
- パラメータが多い

---

## 提案 7: 観測頻度ベース適応忘却（Observation Frequency Adaptive Forgetting）

### コンセプト

観測頻度が高いノードは早く忘れ、低いノードは長く記憶

### 実装イメージ

```python
def evaporate_bkb_frequency_based(
    graph: nx.Graph,
    node: int,
    base_evaporation_rate: float = 0.999,
) -> None:
    # 最近の観測頻度を計算
    observation_count = graph.nodes[node].get("recent_observation_count", 0)
    observation_window = 10  # 直近10世代

    frequency = observation_count / observation_window

    # 頻度が高い → 早く忘れる（新しい情報が来るから）
    # 頻度が低い → 長く記憶（情報が少ないから）
    if frequency > 0.5:  # 高頻度
        evaporation_rate = base_evaporation_rate * 0.95
    elif frequency < 0.1:  # 低頻度
        evaporation_rate = base_evaporation_rate * 1.02
    else:
        evaporation_rate = base_evaporation_rate

    graph.nodes[node]["best_known_bottleneck"] *= evaporation_rate
```

### 利点

- 情報の多寡に応じて適応
- 分散的（ノード独立）
- 実装が簡単

### 欠点

- 頻度の計算に追加の状態管理が必要

---

## 推奨実装順序

1. **提案 3: 移動最大値の指数平滑** ⭐⭐⭐

   - シンプルで効果的
   - ノイズに強い
   - 実装が容易

2. **提案 2: 適応的忘却率** ⭐⭐

   - 環境適応性が高い
   - 既存の CV 計算を活用可能

3. **提案 1: 重み付き時間窓最大値** ⭐
   - 最新情報を重視
   - 実装がやや複雑

---

## 次のステップ

1. 提案 3 を実装して性能を評価
2. 静的環境と動的環境の両方でテスト
3. 必要に応じて他の提案も試行錯誤

