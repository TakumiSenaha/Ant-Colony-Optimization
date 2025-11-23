## ネットワーク帯域幅変動メカニズム

## Network Bandwidth Fluctuation Mechanism

本シミュレーションでは、**AR(1)モデル（1 次自己回帰モデル）**を用いて、ネットワークエッジの帯域幅を動的に変動させています。
In this simulation, the bandwidth of network edges is dynamically fluctuated using an **AR(1) Model (First-Order Autoregressive Model)**.

### 変動の仕組み

### Fluctuation Mechanism

#### 1. AR(1)モデルによる利用率の更新

#### 1. Utilization Update via AR(1) Model

各エッジの利用率は以下の式で更新されます：
The utilization rate of each edge is updated by the following formula:

```
利用率(t+1) = (1 - φ) × 平均利用率 + φ × 利用率(t) + ノイズ
           = 0.02 + 0.95 × 利用率(t) + ノイズ
```

```
Utilization(t+1) = (1 - φ) × Mean Utilization + φ × Utilization(t) + Noise
           = 0.02 + 0.95 × Utilization(t) + Noise
```

**パラメータ**:
**Parameters**:

- **平均利用率**: 0.4（40%）
- **Mean Utilization**: 0.4 (40%)
- **自己相関係数（φ）**: 0.95（直前の値に 95%依存）
- **Autocorrelation Coefficient (φ)**: 0.95 (95% dependent on the previous value)
  - **φ（ファイ）とは**: 直前の値への依存度を表すパラメータ
  - **What is φ (phi)**: A parameter representing the degree of dependence on the previous value
- **ノイズ**: 正規分布 N(0, 0.000975) から生成
- **Noise**: Generated from a Normal distribution N(0, 0.000975)

#### 2. 利用率から可用帯域への変換

#### 2. Conversion from Utilization to Available Bandwidth

```
可用帯域 = キャパシティ × (1 - 利用率)
```

```
Available Bandwidth = Capacity × (1 - Utilization)
```

**例**:
**Example**:

- キャパシティ: 100Mbps
- Capacity: 100Mbps
- 利用率: 0.4（40%）
- Utilization: 0.4 (40%)
- 可用帯域: 100 × (1 - 0.4) = 60Mbps
- Available Bandwidth: 100 × (1 - 0.4) = 60Mbps

#### 3. ノイズとは？

#### 3. What is Noise?

**ノイズ**は、予測不可能なランダムな変動成分です。
**Noise** is an unpredictable, random fluctuation component.

- **生成方法**: 正規分布（ガウス分布）から生成
- **Generation Method**: Generated from a Normal (Gaussian) distribution.
- **平均**: 0
- **Mean**: 0
- **標準偏差**: √0.000975 ≈ 0.0312（約 3.12%）
- **Standard Deviation**: √0.000975 ≈ 0.0312 (approx. 3.12%)

**役割**:
**Role**:

- 完全な予測を防ぐ
- Prevents perfect predictability.
- 探索の必要性を保証
- Ensures the necessity of exploration.
- 現実的なネットワークトラフィックの予期しない変動を再現
- Replicates unexpected fluctuations found in realistic network traffic.

### 変動の特徴

### Characteristics of Fluctuation

#### 1. 高い自己相関（φ = 0.95）

#### 1. High Autocorrelation (φ = 0.95)

- 現在の値は直前の値に 95%依存
- The current value is 95% dependent on the previous value.
- 急激な変化は起こりにくい
- Abrupt changes are unlikely.
- 現実のネットワークトラフィックに近い挙動
- This behavior is similar to realistic network traffic patterns.

#### 2. 平均回帰性

#### 2. Mean Reversion

- 長期的には平均利用率 40%に収束
- In the long term, the value converges to the mean utilization of 40%.
- 極端な状態（高帯域・低帯域）が長期間持続しない
- Extreme states (very high or very low bandwidth) do not persist for long periods.

#### 3. 確率的変動

#### 3. Stochastic Fluctuation

- ノイズ項により完全な予測は不可能
- Perfect prediction is impossible due to the noise term.
- 探索の必要性を保証
- This guarantees the necessity of exploration.

#### 4. 変動範囲の制限

#### 4. Fluctuation Range Limit

- 利用率は **0.05 ～ 0.95** の範囲にクリップ
- The utilization rate is **clipped** to the range [0.05, 0.95].
- 極端な値を防ぐ
- This prevents extreme, unrealistic values.

### 変動のタイミング

### Fluctuation Timing

- **更新間隔**: `BANDWIDTH_UPDATE_INTERVAL` 世代ごと
- **Update Interval**: Every `BANDWIDTH_UPDATE_INTERVAL` generations.
- **現在の設定**: 10000 世代ごと（実質的にほぼ固定）
- **Current Setting**: Every 10,000 generations (i.e., virtually static).

### 変動対象エッジ

### Target Edges for Fluctuation

- **全エッジではなく、選択されたエッジのみが変動**
- **Fluctuation is applied only to selected edges, not all edges.**
- **デフォルト設定**: ハブノード（次数の高いノード）に接続するエッジの約 10%
- **Default Setting**: Approx. 10% of edges connected to hub nodes (nodes with high degree).
- **選択方法**: `EDGE_SELECTION_METHOD` で指定可能
- **Selection Method**: Can be specified via `EDGE_SELECTION_METHOD`.
  - `"hub"`: ハブノード選択（推奨）
  - `"hub"`: Hub node selection (Recommended)
  - `"random"`: 完全ランダム
  - `"random"`: Completely random
  - `"betweenness"`: 媒介中心性選択
  - `"betweenness"`: Betweenness centrality selection

### 実装ファイル

### Implementation Files

- **設定ファイル**: `src/bandwidth_fluctuation_config.py`
- **Configuration file**: `src/bandwidth_fluctuation_config.py`
- **メイン実装**: `src/aco_main_bkb_available_bandwidth.py`
- **Main implementation**: `src/aco_main_bkb_available_bandwidth.py`

### 簡潔な説明（質問されたとき用）

### Concise Explanation (For Q&A)

**Q: このネットワークはどのように変動するか？**
**Q: How does this network fluctuate?**

**A**: AR(1)モデル（1 次自己回帰モデル）により、各エッジの利用率を更新し、それを可用帯域に変換して変動させます。
**A**: It uses an AR(1) model to update the "utilization rate" of each edge, which is then converted into "available bandwidth."

1. **利用率の更新**: `利用率(t+1) = 0.02 + 0.95 × 利用率(t) + ノイズ`
1. **Utilization Update**: `Utilization(t+1) = 0.02 + 0.95 × Utilization(t) + Noise`

   - 直前の値に 95%依存（高い自己相関）
   - Highly autocorrelated (95% dependent on the previous value).
   - ランダムノイズで予測不能な変動も含む
   - Includes random noise for unpredictable changes.
   - 長期的には平均利用率 40%に収束
   - Reverts to a 40% mean utilization over the long term.

1. **可用帯域の計算**: `可用帯域 = キャパシティ × (1 - 利用率)`
1. **Available Bandwidth Calc**: `Available Bandwidth = Capacity × (1 - Utilization)`

   - 利用率が上がる → 可用帯域が減る
   - Higher utilization → Lower available bandwidth.
   - 利用率が下がる → 可用帯域が増える
   - Lower utilization → Higher available bandwidth.

1. **変動タイミング**: 10000 世代ごと（現在の設定）
1. **Fluctuation Timing**: Every 10,000 generations (in the current setting).

1. **変動対象**: 全エッジの約 10%（ハブノードに接続するエッジ）
1. **Fluctuation Target**: Approx. 10% of all edges (connected to hub nodes).
