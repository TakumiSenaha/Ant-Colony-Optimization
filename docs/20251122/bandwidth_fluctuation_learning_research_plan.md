# 帯域変動学習の研究調査計画

## ⚠️ 大前提：シンプルな実装方針

**独自の工夫を極力避け、研究で提案された手法をそのまま適用する**

- 研究論文の手法を**忠実に再現**することを優先
- 「○○ 論文の ○○ 手法を適用しました」と明確に主張できるようにする
- 独自の改良や組み合わせは避ける（比較が困難になるため）
- 必要最小限の実装のみを行う

---

## 🚨 現在の実装の問題点

### 問題 1: 観測タイミングの不一致

**現在の実装**:

```python
# アリがエッジを通過したときだけ観測
if observe_bandwidth_func is not None:
    observe_bandwidth_func(graph, u, v, edge_bandwidth)
```

**問題点**:

- アリが通らないエッジは観測されない
- 帯域変動は毎世代 `update_available_bandwidth_ar1` で全エッジに適用される
- 観測と変動のタイミングが合っていない

**正しいアプローチ**:

```python
# 毎世代、全エッジの帯域幅をチェックして記録
for u, v in graph.edges():
    current_bandwidth = graph[u][v]["weight"]
    observe_edge_bandwidth(graph, u, v, current_bandwidth)
```

### 問題 2: BKB と帯域変動学習の混同

|                    | BKB 学習                                                  | 帯域変動学習                         |
| ------------------ | --------------------------------------------------------- | ------------------------------------ |
| **目的**           | ノードが知る「最良のボトルネック値」                      | エッジの帯域幅変動パターンを学習     |
| **学習タイミング** | アリが経路を通ったとき                                    | 毎世代、全エッジを監視               |
| **必要な情報**     | アリが実際に通過した経路のボトルネック                    | エッジの `weight` 属性の時系列データ |
| **アリが必要？**   | ✅ 必要（経路を実際に通らないとボトルネックが分からない） | ❌ 不要（エッジの属性を読むだけ）    |

**結論**: 帯域変動学習は**アリに依存しない独立したシステム**であるべき。

---

## 📚 調査したい研究分野

### 1. ネットワーク帯域変動のモデリング

#### 1.1 時系列予測モデル

- **ARIMA / AR(1)モデル**: 現在使用中だが、予測精度の評価方法
- **LSTM / GRU**: 深層学習ベースの時系列予測
- **Prophet**: Facebook の時系列予測ライブラリ
- **状態空間モデル**: Kalman Filter など

#### 1.2 帯域幅変動の特性に関する研究

- **変動パターンの分類**: 周期的 / トレンド / ランダム
- **変動の自己相関**: AR(1)係数の適切な範囲
- **変動係数（CV）の閾値**: どの程度の変動を「高変動」とみなすか
- **帯域幅の分布特性**: 正規分布 / ロングテール分布など

#### 1.3 ネットワークトラフィック予測

- **Traffic Prediction**: ISP のトラフィック予測手法
- **Available Bandwidth Estimation**: ネットワーク測定プロトコル
- **Network Tomography**: エンドツーエンド測定から帯域を推定

### 2. 適応的ルーティング・経路選択への応用

#### 2.1 フェロモン（経路評価値）の動的調整

- **ACO in Dynamic Environments**: 動的環境でのアリコロニー最適化
- **Adaptive Evaporation Rate**: 揮発率の動的調整手法
- **Path Quality Prediction**: 経路品質の予測に基づく選択

#### 2.2 帯域予測に基づく経路選択

- **Proactive Routing**: 予測に基づく先制的な経路選択
- **Predictive Load Balancing**: 負荷予測に基づく負荷分散
- **QoS-aware Routing**: 帯域予測を考慮した QoS ルーティング

### 3. 分散システムにおける帯域監視

#### 3.1 分散帯域監視プロトコル

- **Network Monitoring Protocols**: SNMP, NetFlow, sFlow など
- **Passive vs Active Measurement**: パッシブ測定 vs アクティブ測定
- **Distributed Monitoring**: 分散環境での監視データの集約

#### 3.2 ノードローカルな学習

- **Local Learning**: 各ノードがローカルに学習する手法
- **Edge-based Monitoring**: エッジ（リンク）ベースの監視
- **In-network Learning**: ネットワーク内での学習

---

## 🔍 具体的な調査項目

### A. 帯域変動のモデリング手法

1. **AR(1)モデルの適用性**

   - AR(1)係数の推定方法（最小二乗法以外）
   - パラメータの適切な範囲
   - 予測精度の評価指標（MAE, RMSE, MAPE など）

2. **周期性検出の精度向上**

   - FFT ベースの周期検出
   - Autocorrelation Function (ACF) / Partial ACF (PACF)
   - Wavelet 変換による時系列解析

3. **複数の変動パターンへの対応**
   - 周期的変動 + トレンド
   - 突発的な変動（バースト）の検出
   - 段階的な変化（レジームチェンジ）の検出

### B. 帯域予測の評価方法

1. **予測精度の指標**

   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)
   - 予測区間（Confidence Interval）

2. **予測の有用性**
   - 予測に基づく経路選択の性能向上
   - 予測誤差がルーティングに与える影響
   - 予測期間（何世代先まで予測できるか）

### C. 適応的揮発率の設計

1. **揮発率調整のルール**

   - 変動係数に基づく調整の閾値
   - 周期的変動を考慮した調整タイミング
   - トレンドを考慮した調整の方向性

2. **BKB ペナルティとの統合**
   - 帯域予測による揮発と BKB ペナルティの相互作用
   - 両方の要因を組み合わせた最適な揮発率の計算方法

### D. 実装の最適化

1. **計算コストの削減**

   - 全エッジの監視コスト（現在: O(E) 毎世代）
   - パターン学習の計算量削減
   - 更新間隔の最適化

2. **メモリ使用量**
   - 履歴データの保持期間
   - リングバッファサイズの最適化
   - 圧縮・要約手法

---

## 📖 関連研究のキーワード

### 時系列予測

- Time Series Forecasting
- ARIMA / AR(1) Model
- Exponential Smoothing
- LSTM for Time Series
- Kalman Filter

### ネットワークトラフィック

- Network Traffic Prediction
- Available Bandwidth Estimation
- Network Tomography
- Traffic Pattern Analysis
- Bandwidth Forecasting

### 動的環境での最適化

- Dynamic Optimization
- Adaptive Algorithms
- Online Learning
- Reinforcement Learning for Routing
- Predictive Routing

### ACO の拡張

- Ant Colony Optimization in Dynamic Environments
- Adaptive ACO
- Time-varying ACO
- Dynamic Routing with ACO

---

## 🎯 調査の優先順位

### Phase 1: 基礎調査（最優先）

1. ✅ **帯域変動の観測方法の修正**: アリ依存を排除し、毎世代全エッジを監視
2. **AR(1)モデルの予測実装**: 現在は係数推定のみ。実際の帯域値を予測する機能を追加
3. **予測精度の評価**: MAE, RMSE などの指標で予測精度を測定

### Phase 2: 手法の拡張

1. **周期性検出の改善**: FFT や Wavelet による精度向上
2. **複数パターンの検出**: 周期的 + トレンドの同時検出
3. **予測区間の計算**: 信頼区間を考慮した揮発率調整

### Phase 3: 統合と最適化

1. **BKB と帯域予測の統合**: 両方の要因を考慮した揮発率
2. **計算コストの削減**: 不要なエッジの監視をスキップ
3. **実環境データでの評価**: 実ネットワークのトラフィックデータでの検証

---

## 📝 次のステップ

1. **実装の修正**: `observe_edge_bandwidth` をアリ経路から独立させ、毎世代全エッジを監視
2. **関連論文の調査**: 上記キーワードで文献検索
3. **予測機能の実装**: AR(1)モデルで実際の帯域値を予測
4. **評価フレームワーク**: 予測精度を測定する仕組みを構築
