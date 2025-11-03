# 帯域変動学習の実装準備

## ⚠️ 大前提：シンプルな実装方針

**独自の工夫を極力避け、研究で提案された手法をそのまま適用する**

- 研究論文の手法を**忠実に再現**することを優先
- 「○○ 論文の ○○ 手法を適用しました」と明確に主張できるようにする
- 独自の改良や組み合わせは避ける（比較が困難になるため）
- 必要最小限の実装のみを行う

## 📚 研究コンペンディウムの主要な推奨事項

詳細は `docs/research_compendium_summary.md` を参照。

### 最重要な変更点

1. **AR(1)モデルは不十分** → **LSTM/GRU への移行が必須**

   - AR(1)は LRD（長距離依存性）を捉えられない
   - LSTM/GRU への移行は「段階的改善」ではなく「必要条件」

2. **FFT ではなくウェーブレット変換** を周期性検出に使用

   - 時間的局在性を考慮できるため、バースト検出に適している

3. **予測変動性に基づく適応型蒸発率** の実装
   - 高変動予測 → 蒸発率$\rho$を増加（探索促進）
   - 低変動予測 → 蒸発率$\rho$を減少（活用促進）

---

## 現在の実装状況

### 観測フロー（現在）

```
世代開始
  ↓
AR(1)帯域変動更新 (update_available_bandwidth_ar1)
  ↓
アリ探索 (ant_next_node_const_epsilon)
  ↓
アリがゴール到達 → update_pheromone
  ↓
経路上のエッジのみ observe_edge_bandwidth  ← 問題点：アリが通らないエッジは観測されない
  ↓
10世代ごとに update_patterns_for_all_edges
  ↓
履歴があるエッジのみ learn_bandwidth_pattern
```

### 問題点

1. **観測の欠落**: アリが通らないエッジの帯域変動が記録されない
2. **タイミングの不一致**: 帯域変動は毎世代だが、観測はアリ経路依存

---

## 修正が必要な箇所

### 1. 帯域観測のタイミング変更

**現在**:

- `update_pheromone` 内でアリ経路上のエッジのみ観測

**修正後**:

- 毎世代、`update_available_bandwidth_ar1` の**直後**に全エッジを観測

### 2. 新しい関数の追加

`src/bandwidth_monitoring.py` に以下を追加：

```python
def observe_all_edges_bandwidth(
    graph: nx.Graph,
    max_history_size: int = 100,
) -> None:
    """
    全エッジの現在の帯域幅を観測し、履歴に記録する

    アリに依存せず、毎世代すべてのエッジの帯域を記録します。
    帯域変動更新（update_available_bandwidth_ar1）の直後に呼び出す。

    Args:
        graph: ネットワークグラフ
        max_history_size: 保持する履歴の最大サイズ（リングバッファ）
    """
    for u, v in graph.edges():
        current_bandwidth = graph[u][v]["weight"]
        observe_edge_bandwidth(graph, u, v, current_bandwidth, max_history_size)
```

### 3. メインループの修正

`src/aco_main_bkb_available_bandwidth_rfc.py` のメインループ：

**修正前**:

```python
# === AR(1)モデルによる帯域変動 ===
bandwidth_changed = update_available_bandwidth_ar1(
    graph, edge_states, generation
)

# === アリの探索 ===
# ...

# === 帯域変動パターンの学習（10世代ごと）===
if USE_ADAPTIVE_EVAPORATION:
    update_patterns_for_all_edges(...)
```

**修正後**:

```python
# === AR(1)モデルによる帯域変動 ===
bandwidth_changed = update_available_bandwidth_ar1(
    graph, edge_states, generation
)

# === ★★★ 全エッジの帯域を観測（毎世代）★★★ ===
if USE_ADAPTIVE_EVAPORATION:
    observe_all_edges_bandwidth(graph)

# === アリの探索 ===
# ...

# === 帯域変動パターンの学習（10世代ごと）===
if USE_ADAPTIVE_EVAPORATION:
    update_patterns_for_all_edges(...)
```

### 4. update_pheromone からの削除

`src/pheromone_update.py` の `update_pheromone` 関数：

**現在**:

```python
# アリがエッジを通過したときに観測
if observe_bandwidth_func is not None:
    observe_bandwidth_func(graph, u, v, edge_bandwidth)
```

**修正案**:

- `observe_bandwidth_func` パラメータは残す（将来的に別の用途に使う可能性）
- ただし、`aco_main_bkb_available_bandwidth_rfc.py` では `observe_func=None` に設定

---

## 新手法導入時の拡張ポイント

### 1. 予測機能の追加

現在は統計量（平均、分散、CV 等）を計算しているが、**実際の帯域値を予測**する機能がない。

**拡張ポイント**:

- `predict_next_bandwidth()` 関数を追加
- AR(1)モデル、LSTM、Prophet など、研究で推奨された手法を実装

### 2. 予測精度の評価

**拡張ポイント**:

- `evaluate_prediction_accuracy()` 関数を追加
- MAE, RMSE, MAPE などの指標を計算
- 予測結果をログに記録

### 3. 周期性検出の改善

**拡張ポイント**:

- FFT ベースの周期検出
- Wavelet 変換
- 複数の周期を同時に検出

### 4. 揮発率調整ロジックの拡張

**拡張ポイント**:

- 予測値を考慮した揮発率計算
- 予測区間（信頼区間）を考慮
- 複数の予測モデルの統合

---

## ファイル構成

```
src/
├── bandwidth_monitoring.py       # 帯域監視・学習モジュール
│   ├── observe_edge_bandwidth()              # 単一エッジの観測（既存）
│   ├── observe_all_edges_bandwidth()         # 全エッジの観測（追加予定・Phase 1）
│   │
│   ├── learn_bandwidth_pattern()             # パターン学習（既存・簡易版）
│   │   # → LSTM/GRUベースの学習に置き換え（Phase 2）
│   │
│   ├── predict_next_bandwidth_lstm()         # LSTM予測（Phase 2・最優先）
│   │   # Based on: 研究コンペンディウム
│   │   # AR(1)は非推奨、LSTM/GRUに移行
│   │
│   ├── detect_periodicity_wavelet()          # ウェーブレット変換による周期検出（Phase 2）
│   │   # FFTではなくウェーブレット変換を使用
│   │
│   ├── cluster_edges_by_traffic_pattern()    # エッジクラスタリング（Phase 2・オプション）
│   │   # トラフィックパターンが類似したエッジをクラスタリング
│   │
│   ├── evaluate_prediction_accuracy()        # 予測精度評価（Phase 4）
│   │   # MAE, RMSE, MAPE
│   │
│   └── calculate_adaptive_evaporation_rate() # 適応的揮発率計算（Phase 3）
│       # 予測変動性に基づく$\rho$調整
│       # ルール1: 高変動 → $\rho$増加
│       # ルール2: 低変動 → $\rho$減少
│
└── aco_main_bkb_available_bandwidth_rfc.py  # メインループ（修正予定）
    ├── 予測的ヒューリスティックの統合（Phase 3）
    └── 使用する手法をパラメータで選択可能にする
```

### 実装時の命名規則

- 研究手法名を関数名に含める

  - `predict_next_bandwidth_arima()`
  - `predict_next_bandwidth_lstm()`
  - `detect_periodicity_fft()`

- 各関数の先頭に論文情報をコメントで記載

  ```python
  def predict_next_bandwidth_lstm(history: list[float]) -> float:
      """
      LSTMモデルによる帯域予測

      Based on: 研究コンペンディウム「予測的適応型ルーティングのための研究コンペンディウム」
      Chapter 2.2.2: 非線形深層学習モデル (LSTM/GRU) の優位性

      Rationale: AR(1)モデルはLRD（長距離依存性）を捉えられず、
                 予測精度が低い。LSTMへの移行は必要条件。

      Algorithm: [論文のアルゴリズム番号やセクション]
      Parameters: [論文の推奨値]
      """
  ```

---

## 次のステップ（研究結果待ち）

### Phase 1: 基礎修正（即座に実行可能）

1. ✅ 観測タイミングの修正（`observe_all_edges_bandwidth` の実装）
   - アリ依存を排除
   - 毎世代全エッジを監視

### Phase 2: 研究手法の適用（研究結果に基づく）

2. ⏳ **研究で提案された予測手法の実装**

   - 論文のアルゴリズムをそのまま実装
   - パラメータは論文の推奨値を使用
   - 関数名に手法名を含める（例: `predict_arima()`）

3. ⏳ **予測精度の評価フレームワーク**

   - 論文で使用されている評価指標を実装
   - 評価方法も論文に従う

4. ⏳ **揮発率調整ロジックの実装**
   - 研究で提案されたルールをそのまま実装
   - 論文の数式・アルゴリズムに従う

### 実装時のチェックリスト

- [ ] 論文のアルゴリズムを正確に再現しているか
- [ ] パラメータは論文の推奨値を使用しているか
- [ ] 独自の調整・改良を加えていないか
- [ ] 関数名に手法名が含まれているか（例: `predict_lstm()`, `detect_wavelet()`）
- [ ] コメントに研究コンペンディウムの章番号などが記載されているか
- [ ] 「研究コンペンディウムの ○○ 手法を適用」と明確に説明できるか
- [ ] AR(1)モデルを使用していないか（非推奨）
- [ ] LSTM/GRU が実装されているか（Phase 2 で必須）

---

## 実装時の注意点

### 計算コスト

- 全エッジの観測: O(E) 毎世代（E ≈ 594 エッジ）
- パターン学習: 10 世代ごと O(E × 学習計算量)
- 新手法導入時は計算コストも考慮

### メモリ使用量

- 履歴データ: エッジ数 × 履歴サイズ × float サイズ
- 現在: 594 × 100 × 8 bytes ≈ 475 KB（軽量）
- 新手法導入時も履歴サイズの制御が必要

### 分散システムの制約

- 各ノードは自分の接続エッジの帯域のみを知ることができる
- グローバルな情報に依存しない設計を維持
