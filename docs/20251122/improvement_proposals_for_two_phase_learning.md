# 二段階 BKB 学習の改善提案

## 現状の課題

### 定量的な評価

| 指標         | 現状  | 目標 | 達成度      |
| ------------ | ----- | ---- | ----------- |
| 平均追従率   | 54.6% | 70%+ | ❌ 不足     |
| 後期追従率   | 60.3% | 80%+ | ❌ 不足     |
| 平均成功率   | 15.7% | 20%+ | 😐 やや不足 |
| 成功率中央値 | 0%    | 10%+ | ❌ 大幅不足 |

### 定性的な問題

1. **追従の遅さ**: 環境変化への反応が鈍い
2. **不安定性**: 成功率が 0%と 100%の間で激しく変動
3. **収束の弱さ**: 後期でも追従率が 60%程度で頭打ち

---

## 改善案 1: 三段階学習モデル（超短期追加）

### 概要

現在の二段階（短期 α=0.5 + 長期 α=0.125）に、**超短期（α=0.9）**を追加

### アーキテクチャ

```python
def update_node_bkb_three_phase(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    config: BKBLearningConfig
):
    """三段階BKB学習"""

    # ★★★ 超短期記憶（α=0.9）：最新1-2世代を強く反映 ★★★
    ultra_short = graph.nodes[node].get("ultra_short_ema_bkb")
    if ultra_short is None:
        ultra_short = float(bottleneck)
    else:
        ultra_short = 0.1 * ultra_short + 0.9 * bottleneck

    # ★★★ 短期記憶（α=0.5）：直近5-10世代を反映 ★★★
    short_mean = graph.nodes[node].get("short_ema_bkb")
    if short_mean is None:
        short_mean = float(bottleneck)
    else:
        short_mean = 0.5 * short_mean + 0.5 * bottleneck

    # ★★★ 長期記憶（α=0.125）：安定した基準 ★★★
    long_mean = graph.nodes[node].get("long_ema_bkb")
    if long_mean is None:
        long_mean = float(bottleneck)
    else:
        long_alpha = config.mean_alpha  # 0.125
        long_mean = (1 - long_alpha) * long_mean + long_alpha * bottleneck

    # ★★★ 実効BKB = 3つの最大値 ★★★
    effective_mean = max(ultra_short, short_mean, long_mean)

    # 分散は超短期から計算（最も変動を捉える）
    var_prev = graph.nodes[node].get("ema_bkb_var", 0.0)
    deviation = abs(bottleneck - ultra_short)
    var_new = (1 - config.var_beta) * var_prev + config.var_beta * deviation

    # 保存
    graph.nodes[node]["ultra_short_ema_bkb"] = ultra_short
    graph.nodes[node]["short_ema_bkb"] = short_mean
    graph.nodes[node]["long_ema_bkb"] = long_mean
    graph.nodes[node]["ema_bkb"] = effective_mean
    graph.nodes[node]["ema_bkb_var"] = var_new
```

### 期待される効果

- ✅ 急激な変化への**即座の対応**（α=0.9）
- ✅ 中期的なトレンド追跡（α=0.5）
- ✅ 長期的な安定性（α=0.125）
- 📈 **追従率: 60% → 75-80% に向上**

### 実装優先度: ⭐⭐⭐⭐⭐ 最高

---

## 改善案 2: 重み付き平均による実効 BKB 計算

### 概要

現在の `max(短期, 長期)` から、**重み付き平均**に変更

### 計算式

```python
# 動的な重み計算（環境変動に応じて調整）
def calculate_dynamic_weights(node_var: float) -> tuple:
    """
    変動が大きい → 短期重視
    変動が小さい → 長期重視
    """
    # 変動度に基づく短期の重み（0.5～0.9）
    w_short = 0.5 + 0.4 * min(node_var / 10.0, 1.0)
    w_long = 1.0 - w_short
    return w_short, w_long

# 実効BKB計算
def calculate_effective_bkb_weighted(
    ultra_short: float,
    short: float,
    long: float,
    node_var: float
) -> float:
    """重み付き平均による実効BKB"""
    w_ultra = 0.5  # 超短期の重み（固定）
    w_short, w_long = calculate_dynamic_weights(node_var)

    # 超短期50% + 短期と長期で残り50%を分割
    effective = (
        w_ultra * ultra_short +
        (1 - w_ultra) * (w_short * short + w_long * long)
    )
    return effective
```

### 期待される効果

- ✅ **スムーズな追従**（max()の急激な切り替えを回避）
- ✅ 環境変動に応じた**適応的な学習**
- 📈 **成功率の安定性向上**: 0%-100%の変動を緩和

### 実装優先度: ⭐⭐⭐⭐ 高

---

## 改善案 3: フェロモン更新の強化

### 概要

**より強い学習シグナル**を提供して、最適経路への収束を加速

### パラメータ調整

```python
# 現在
ACHIEVEMENT_BONUS = 1.5
PENALTY_FACTOR = 0.5
V = 0.99  # フェロモン揮発率

# 提案
ACHIEVEMENT_BONUS = 3.0  # 2倍に増加
PENALTY_FACTOR = 0.3     # より厳しいペナルティ
V = 0.95                 # より速い忘却
```

### 追加の改善: 変動検出型ボーナス

```python
def calculate_adaptive_bonus(
    bottleneck: float,
    node_mean: float,
    node_var: float,
    base_bonus: float = 3.0
) -> float:
    """
    変動が大きい環境 → より大きなボーナス
    変動が小さい環境 → 標準的なボーナス
    """
    if bottleneck <= node_mean:
        return 1.0

    # 変動度に応じたボーナス増幅
    volatility_factor = 1.0 + min(node_var / 5.0, 1.0)
    bonus = base_bonus * volatility_factor

    return min(bonus, 5.0)  # 最大5倍
```

### 期待される効果

- ✅ **最適経路への強い誘導**
- ✅ 動的環境での**迅速な経路切り替え**
- 📈 **成功率: 15% → 25-30% に向上**

### 実装優先度: ⭐⭐⭐⭐ 高

---

## 改善案 4: ε-Greedy の動的調整

### 概要

現在の固定 `EPSILON = 0.1` を、**学習の進行に応じて調整**

### 実装

```python
def calculate_dynamic_epsilon(
    generation: int,
    max_generation: int,
    tracking_rate: float,
    base_epsilon: float = 0.1
) -> float:
    """
    学習の進行状況に応じたε値の調整

    初期: 高いε → 積極的な探索
    後期 & 高追従率: 低いε → 活用重視
    後期 & 低追従率: 高いε → 再探索
    """
    # 進捗率（0→1）
    progress = generation / max_generation

    # 追従率が低い場合は探索を増やす
    if tracking_rate < 0.5:
        # 追従率が低い → 高いεで探索
        epsilon = base_epsilon + 0.2 * (1 - tracking_rate)
    else:
        # 追従率が高い → 進捗に応じてεを減少
        epsilon = base_epsilon * (1 - 0.5 * progress)

    return max(0.05, min(epsilon, 0.5))
```

### 期待される効果

- ✅ **初期の探索強化**
- ✅ **後期の活用強化**（追従率が高い場合）
- ✅ **停滞時の再探索**（追従率が低い場合）
- 📈 **全体的な性能のバランス向上**

### 実装優先度: ⭐⭐⭐ 中

---

## 改善案 5: マルチスケール分散計算

### 概要

現在の単一分散から、**複数タイムスケールの分散**を計算

### 実装

```python
def update_multiscale_variance(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    config: BKBLearningConfig
):
    """マルチスケール分散計算"""

    ultra_short = graph.nodes[node].get("ultra_short_ema_bkb", bottleneck)
    short_mean = graph.nodes[node].get("short_ema_bkb", bottleneck)
    long_mean = graph.nodes[node].get("long_ema_bkb", bottleneck)

    # 各スケールでの分散
    var_ultra_short = abs(bottleneck - ultra_short)
    var_short = abs(bottleneck - short_mean)
    var_long = abs(bottleneck - long_mean)

    # 総合変動度（最大値を使用）
    max_deviation = max(var_ultra_short, var_short, var_long)

    # EMA更新
    var_prev = graph.nodes[node].get("ema_bkb_var", 0.0)
    var_new = (1 - config.var_beta) * var_prev + config.var_beta * max_deviation

    graph.nodes[node]["ema_bkb_var"] = var_new

    # 環境変動度の判定
    if var_new > 10.0:
        graph.nodes[node]["environment_state"] = "high_volatility"
    elif var_new > 5.0:
        graph.nodes[node]["environment_state"] = "moderate_volatility"
    else:
        graph.nodes[node]["environment_state"] = "low_volatility"
```

### 期待される効果

- ✅ **環境変動の正確な検出**
- ✅ 変動度に応じた**適応的な学習戦略**
- 📈 **確信度の精度向上**

### 実装優先度: ⭐⭐ 低

---

## 推奨実装順序

### Phase 1: 即座に実装（最優先）

1. **改善案 1: 三段階学習モデル** ⭐⭐⭐⭐⭐

   - 期待効果: 追従率 60% → 75-80%
   - 実装難易度: 中
   - 投資対効果: **最高**

2. **改善案 3-A: フェロモンパラメータ調整** ⭐⭐⭐⭐
   - `ACHIEVEMENT_BONUS = 3.0`, `V = 0.95`
   - 期待効果: 成功率 15% → 25%
   - 実装難易度: 低
   - 投資対効果: **高**

### Phase 2: 効果確認後に実装

3. **改善案 2: 重み付き平均** ⭐⭐⭐⭐

   - 期待効果: 成功率の安定性向上
   - Phase 1 の結果を見て判断

4. **改善案 3-B: 変動検出型ボーナス** ⭐⭐⭐⭐
   - 期待効果: 動的環境での適応性向上
   - Phase 1 の結果を見て判断

### Phase 3: 必要に応じて実装

5. **改善案 4: 動的 ε-Greedy** ⭐⭐⭐
6. **改善案 5: マルチスケール分散** ⭐⭐

---

## まとめ

### 現状の評価

- ✅ 二段階学習は**ベースラインより明確に優れている**
- 😐 しかし**動的環境では追従率 60%程度で不十分**
- ❌ 成功率の**不安定性**が課題

### 次のステップ

**最優先実装: 三段階学習モデル（超短期 α=0.9 追加）**

これにより、追従率が**75-80%**に向上し、動的環境でも実用的な性能が期待できます。

実装後の目標:

- 追従率: **75%以上**
- 成功率: **25%以上**
- 成功率中央値: **10%以上**

