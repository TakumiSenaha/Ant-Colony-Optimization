# 研究コンペンディウム推奨手法の実装状況

## ✅ 実装完了

### Phase 1: 監視アーキテクチャの修正 ✅

- [x] `observe_all_edges_bandwidth()` 関数の実装
  - 毎世代、全エッジの帯域を監視（アリ依存を排除）
  - `src/bandwidth_monitoring.py` に実装済み
  - `src/aco_main_bkb_available_bandwidth_rfc.py` のメインループで統合済み

### Phase 2: 予測モデルの移行 ✅

- [x] **シンプルな予測手法の実装**（3 種類）

  - `predict_next_bandwidth_ar1()` - AR(1)予測
  - `predict_next_bandwidth_ma()` - 移動平均（最もシンプル）
  - `predict_next_bandwidth_ema()` - 指数平滑法
  - `predict_next_bandwidth()` - 統一インターフェース

- [x] **ウェーブレット変換による周期性検出**
  - `detect_periodicity_wavelet()` 関数の実装
  - `learn_bandwidth_pattern()` に `use_wavelet` パラメータを追加
  - `update_patterns_for_all_edges()` に `use_wavelet` パラメータを追加

### Phase 3: ACO への統合 ✅

- [x] **予測変動性に基づく適応型蒸発率**

  - `calculate_adaptive_evaporation_rate()` を拡張
  - `use_prediction_variability` パラメータを追加
  - ルール 1（高変動）: 予測変動が高い → 蒸発率 ρ を増加（探索促進）
  - ルール 2（低変動）: 予測変動が低い → 蒸発率 ρ を減少（活用促進）

- [x] **予測的ヒューリスティックの統合**
  - `calculate_predictive_heuristic()` 関数の実装
  - `ant_next_node_const_epsilon()` に予測的ヒューリスティックを統合
  - 状態遷移確率: `P_ij ∝ τ_ij^α * η_distance(ij)^β * η_pred(ij)^γ`

---

## 📋 実装詳細

### 実装済み関数一覧

#### `src/bandwidth_monitoring.py`

1. **`observe_all_edges_bandwidth()`** ✅

   - Phase 1: 全エッジの継続的監視

2. **`predict_next_bandwidth_ar1()`** ✅

   - AR(1)モデルによる 1 ステップ先予測

3. **`predict_next_bandwidth_ma()`** ✅

   - 移動平均による予測（最もシンプル）

4. **`predict_next_bandwidth_ema()`** ✅

   - 指数平滑法による予測

5. **`predict_next_bandwidth()`** ✅

   - 統一インターフェース（"ar1", "ma", "ema"を選択可能）

6. **`detect_periodicity_wavelet()`** ✅

   - Phase 2: ウェーブレット変換による周期性検出

7. **`calculate_predictive_heuristic()`** ✅

   - Phase 3: 予測的ヒューリスティック値の計算

8. **`calculate_adaptive_evaporation_rate()`** ✅ (拡張済み)
   - Phase 3: 予測変動性に基づく適応型蒸発率計算

#### `src/aco_main_bkb_available_bandwidth_rfc.py`

1. **メインループの統合** ✅

   - `observe_all_edges_bandwidth()` を毎世代呼び出し

2. **予測的ヒューリスティックの統合** ✅
   - `ant_next_node_const_epsilon()` で使用

---

## 🔧 設定パラメータ

### `src/aco_main_bkb_available_bandwidth_rfc.py`

```python
# 適応的揮発モデル設定
USE_ADAPTIVE_EVAPORATION = True
ADAPTIVE_PATTERN_UPDATE_INTERVAL = 10
ADAPTIVE_MIN_SAMPLES = 10

# 予測的ヒューリスティック設定（研究コンペンディウム推奨: Phase 3）
USE_PREDICTIVE_HEURISTIC = True
PREDICTIVE_HEURISTIC_METHOD = "ar1"  # "ar1", "ma", "ema"
GAMMA = 1.0  # 予測ヒューリスティックの重み
```

### `src/bandwidth_monitoring.py`

```python
# 関数呼び出し時のパラメータ
use_wavelet: bool = False  # ウェーブレット変換を使用するか
use_prediction_variability: bool = True  # 予測変動性を考慮するか
prediction_method: str = "ar1"  # 予測手法（"ar1", "ma", "ema"）
```

---

## 📝 使用方法

### 1. ウェーブレット変換による周期性検出を有効化

```python
update_patterns_for_all_edges(
    graph,
    min_samples=ADAPTIVE_MIN_SAMPLES,
    update_interval=ADAPTIVE_PATTERN_UPDATE_INTERVAL,
    generation=generation,
    use_wavelet=True,  # ウェーブレット変換を使用
)
```

### 2. 予測手法の変更

```python
# AR(1)予測を使用
predicted = predict_next_bandwidth(history, method="ar1")

# 移動平均を使用
predicted = predict_next_bandwidth(history, method="ma", window=5)

# 指数平滑法を使用
predicted = predict_next_bandwidth(history, method="ema", alpha=0.3)
```

### 3. 予測的ヒューリスティックの有効/無効

```python
# src/aco_main_bkb_available_bandwidth_rfc.py で設定
USE_PREDICTIVE_HEURISTIC = True  # 有効化
PREDICTIVE_HEURISTIC_METHOD = "ar1"  # 予測手法を選択
```

---

## ⚠️ 注意事項

### 1. AR(1)モデルの用途の違い

- **生成モデル** (`update_available_bandwidth_ar1`): 変更不要
  - シミュレーション環境で帯域変動を生成する
- **予測モデル** (`predict_next_bandwidth_ar1`): 実装済み
  - 観測された帯域変動から次を予測する

研究コンペンディウムの「AR(1)は不十分」という主張は、**予測モデル**についてであり、**生成モデル**とは無関係。

### 2. BKB 学習との関係

- BKB 学習（ノードの最良ボトルネック値）とは**別システム**
- 帯域変動の予測とは**無関係**

---

## 🔄 次のステップ（将来実装）

### Phase 4: 最適化と評価

- [ ] LSTM/GRU モデルの実装（研究コンペンディウム推奨）
- [ ] エッジクラスタリングの実装
- [ ] 予測精度の評価フレームワーク（MAE, RMSE, MAPE）
- [ ] TinyML 技術の適用（量子化、枝刈り）

---

## 📚 参考

- `docs/research_compendium_summary.md`: 研究コンペンディウムの要約
- `docs/implementation_preparation.md`: 実装準備ドキュメント
- `docs/ar1_clarification.md`: AR(1)モデルの用途の整理
