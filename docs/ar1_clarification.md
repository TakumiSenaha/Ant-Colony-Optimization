# AR(1)モデルの用途の整理

## AR(1)モデルの 2 つの異なる用途

### 1. 帯域変動の**生成**モデル（シミュレーション用）

**場所**: `src/bandwidth_fluctuation_config.py` → `update_available_bandwidth_ar1()`

**目的**: シミュレーション環境で帯域変動を**生成**する

```python
# AR(1)モデルで帯域変動を生成
new_utilization = (
    (1 - AR_COEFFICIENT) * MEAN_UTILIZATION  # 平均への回帰
    + AR_COEFFICIENT * current_utilization   # 過去への依存
    + noise                                  # ランダムノイズ
)
graph[u][v]["weight"] = available_bandwidth  # 実際の帯域を更新
```

**これは**: シミュレーション環境の特性を決めるモデル。**変更不要**。

---

### 2. 帯域変動の**予測**モデル（学習・予測用）

**場所**: `src/bandwidth_monitoring.py` → `estimate_ar1_coefficient()` （係数推定のみ、予測機能は未実装）

**目的**: 観測された帯域変動から、**次世代の帯域を予測**する

```python
# 現在: AR(1)係数の推定のみ実装されている
ar_coefficient = estimate_ar1_coefficient(history)

# 未実装: 実際の予測
predicted_bandwidth = predict_next_bandwidth_ar1(history, ar_coefficient)
```

**これは**: 予測モデル。研究コンペンディウムでは「AR(1)は不十分」という主張。

---

### 3. ノードの学習（BKB）← **全く別の概念**

**場所**: `src/bkb_learning.py`

**目的**: ノードが知る「最良のボトルネック値」を学習する

- 帯域変動の予測とは**無関係**
- ACO アルゴリズムのフェロモン更新に使用

---

## 研究コンペンディウムの主張の解釈

### 主張

> AR(1)モデルは LRD（長距離依存性）を捉えられず、予測精度が低い。LSTM/GRU への移行が必須。

### これはどの用途について？

**→ 予測モデル（用途 2）についての主張**

- **生成モデル（用途 1）**: 影響なし。シミュレーション環境の特性は AR(1)で問題ない。
- **予測モデル（用途 2）**: AR(1)より LSTM/GRU の方が予測精度が高いと主張。
- **BKB 学習（用途 3）**: 無関係。

---

## 実装方針の整理

### 1. 生成モデル（AR(1)）→ **変更不要**

- `update_available_bandwidth_ar1()` はそのまま使用
- シミュレーション環境の特性として適切

### 2. 予測モデル → **複数の手法を実装**

**シンプルな予測手法**（共通モジュールに実装）:

1. **AR(1)予測**（既存の係数推定を拡張）

   ```python
   def predict_next_bandwidth_ar1(history: list[float]) -> float:
       """
       AR(1)モデルによる1ステップ先予測

       Based on: 時系列予測の基本的な手法
       """
       # 係数を推定
       ar_coeff = estimate_ar1_coefficient(history)
       # 予測: y_{t+1} = mean + ar_coeff * (y_t - mean) + mean
       mean = sum(history) / len(history)
       last_value = history[-1]
       predicted = mean + ar_coeff * (last_value - mean)
       return predicted
   ```

2. **移動平均（MA）**（非常にシンプル）

   ```python
   def predict_next_bandwidth_ma(history: list[float], window: int = 5) -> float:
       """
       移動平均による予測（最もシンプル）
       """
       recent_values = history[-window:]
       return sum(recent_values) / len(recent_values)
   ```

3. **指数平滑法（Exponential Smoothing）**（中程度のシンプルさ）

   ```python
   def predict_next_bandwidth_ema(history: list[float], alpha: float = 0.3) -> float:
       """
       指数平滑法による予測
       """
       # シンプルな指数平滑法
       ema = history[0]
       for value in history[1:]:
           ema = alpha * value + (1 - alpha) * ema
       return ema
   ```

4. **LSTM/GRU**（将来実装、研究コンペンディウム推奨）

### 3. BKB 学習 → **変更不要**

- `src/bkb_learning.py` はそのまま使用
- 帯域変動の予測とは別システム

---

## 結論

1. **生成モデル（AR(1)）**: 変更不要
2. **予測モデル**: 複数の手法を実装（AR(1), MA, EMA, LSTM/GRU）
3. **BKB 学習**: 無関係

研究コンペンディウムの「AR(1)は不十分」という主張は、**予測モデル**についてであり、**生成モデル**や**BKB 学習**とは無関係。
