# ネットワーク帯域幅変動メカニズム

# Network Bandwidth Fluctuation Mechanism

本シミュレーションでは、**AR(1)モデル（1 次自己回帰モデル）**を用いて、ネットワークエッジの帯域幅を動的に変動させています。
In this simulation, the bandwidth of network edges is dynamically fluctuated using an **AR(1) Model (First-Order Autoregressive Model)**.

AR(1)モデルは時系列データの変動をモデル化する統計的手法で、ネットワークトラフィックの現実的な変動パターンを再現するために選択されました。
The AR(1) model is a statistical method for modeling time series data fluctuations, selected to replicate realistic network traffic fluctuation patterns.

---

## 変動の仕組み

## Fluctuation Mechanism

### 1. AR(1)モデルによる利用率の更新

### 1. Utilization Update via AR(1) Model

#### 数学的定式化

#### Mathematical Formulation

各エッジの利用率は以下の式で更新されます：
The utilization rate of each edge is updated by the following formula:

```
利用率(t+1) = (1 - φ) × 平均利用率 + φ × 利用率(t) + ε(t)
           = 0.02 + 0.95 × 利用率(t) + ε(t)
```

```
Utilization(t+1) = (1 - φ) × Mean Utilization + φ × Utilization(t) + ε(t)
           = 0.02 + 0.95 × Utilization(t) + ε(t)
```

ここで、`ε(t)` は時刻 `t` におけるランダムノイズ項です。
Here, `ε(t)` is the random noise term at time `t`.

#### パラメータの詳細

#### Parameter Details

**平均利用率（Mean Utilization）**: 0.4（40%）
**Mean Utilization**: 0.4 (40%)

- **選択根拠**: ISP（インターネットサービスプロバイダ）の一般的な運用マージンに基づく
- **Rationale**: Based on typical operational margins of ISPs (Internet Service Providers)
- 実際のネットワークでは、常に 100%の帯域を使用することはなく、余裕を持たせて運用される
- In real networks, bandwidth is not used at 100% capacity; operational margins are maintained
- 40%の平均利用率は、60%の可用帯域を確保することを意味する
- A 40% mean utilization means 60% available bandwidth is maintained

**自己相関係数（φ, phi）**: 0.95
**Autocorrelation Coefficient (φ, phi)**: 0.95

- **選択根拠**: ネットワークトラフィックの高い自己相関特性に基づく
- **Rationale**: Based on the high autocorrelation characteristics of network traffic
- 直前の値に 95%依存するため、急激な変化は起こりにくい
- Since it depends 95% on the previous value, abrupt changes are unlikely
- 現実のネットワークトラフィックは時間的に連続的で、急激な変動は稀である
- Real network traffic is temporally continuous, with rare abrupt fluctuations
- 半減期（影響が半分になるまでの時間）: 約 14 世代
- Half-life (time for influence to halve): Approximately 14 generations

**ノイズ分散（Noise Variance）**: 0.000975
**Noise Variance**: 0.000975

- **標準偏差**: √0.000975 ≈ 0.0312（約 3.12%）
- **Standard Deviation**: √0.000975 ≈ 0.0312 (approx. 3.12%)
- **生成方法**: 正規分布（ガウス分布）N(0, 0.000975) から生成
- **Generation Method**: Generated from a Normal (Gaussian) distribution N(0, 0.000975)
- **選択根拠**: 平均利用率と自己相関係数から逆算された値
- **Rationale**: Calculated inversely from mean utilization and autocorrelation coefficient
- 95%信頼区間: 約 ±6.3%の変動範囲
- 95% confidence interval: Approximately ±6.3% fluctuation range

#### AR(1)モデルの特性

#### AR(1) Model Characteristics

**定常性（Stationarity）**:

- AR(1)モデルは定常過程であり、長期的には平均値に収束する
- The AR(1) model is a stationary process that converges to the mean in the long term
- 自己相関係数 |φ| < 1 のため、システムは安定している
- Since |φ| < 1, the system is stable

**平均回帰性（Mean Reversion）**:

- 長期的には平均利用率 40%に収束する
- Converges to a 40% mean utilization in the long term
- 収束速度: 約 20 世代で 63%収束（1 - 1/e ≈ 0.632）
- Convergence speed: Approximately 63% convergence in 20 generations (1 - 1/e ≈ 0.632)
- 極端な状態（高帯域・低帯域）が長期間持続しない
- Extreme states (very high or very low bandwidth) do not persist for long periods

**自己相関構造（Autocorrelation Structure）**:

- 1 次ラグの自己相関: 0.95
- First-order lag autocorrelation: 0.95
- k 次ラグの自己相関: 0.95^k
- k-th order lag autocorrelation: 0.95^k
- 例: 10 世代前の値との相関は約 0.60（0.95^10）
- Example: Correlation with value 10 generations ago is approximately 0.60 (0.95^10)

---

### 2. 利用率から可用帯域への変換

### 2. Conversion from Utilization to Available Bandwidth

#### 変換式

#### Conversion Formula

```
可用帯域(t) = キャパシティ × (1 - 利用率(t))
```

```
Available Bandwidth(t) = Capacity × (1 - Utilization(t))
```

#### 実装の詳細

#### Implementation Details

**10Mbps 刻みの丸め処理**:
**Rounding to 10Mbps increments**:

計算された可用帯域は、実装上の都合により 10Mbps 刻みに丸められます：
The calculated available bandwidth is rounded to 10Mbps increments for implementation purposes:

```python
available_bandwidth = ((int(available_bandwidth) + 5) // 10) * 10
```

この処理により、帯域幅の値が離散化され、より現実的なネットワーク設定に近づきます。
This processing discretizes bandwidth values, making them closer to realistic network settings.

**双方向エッジの扱い**:
**Bidirectional Edge Handling**:

- 各エッジ (u, v) に対して、双方向 (u, v) と (v, u) の両方が独立して変動する
- For each edge (u, v), both directions (u, v) and (v, u) fluctuate independently
- 各方向に独立した利用率状態が保持される
- Independent utilization states are maintained for each direction
- 初期化時には、各方向の利用率は 0.3 ～ 0.5 の範囲でランダムに設定される
- During initialization, utilization for each direction is randomly set in the range 0.3-0.5

#### 計算例

#### Calculation Example

**例 1: 標準的なケース**
**Example 1: Standard Case**

- キャパシティ: 100Mbps
- Capacity: 100Mbps
- 利用率: 0.4（40%）
- Utilization: 0.4 (40%)
- 可用帯域: 100 × (1 - 0.4) = 60Mbps
- Available Bandwidth: 100 × (1 - 0.4) = 60Mbps

**例 2: 高負荷ケース**
**Example 2: High Load Case**

- キャパシティ: 100Mbps
- Capacity: 100Mbps
- 利用率: 0.7（70%）
- Utilization: 0.7 (70%)
- 可用帯域: 100 × (1 - 0.7) = 30Mbps
- Available Bandwidth: 100 × (1 - 0.7) = 30Mbps

**例 3: 低負荷ケース**
**Example 3: Low Load Case**

- キャパシティ: 100Mbps
- Capacity: 100Mbps
- 利用率: 0.2（20%）
- Utilization: 0.2 (20%)
- 可用帯域: 100 × (1 - 0.2) = 80Mbps
- Available Bandwidth: 100 × (1 - 0.2) = 80Mbps

---

### 3. ノイズの役割と特性

### 3. Role and Characteristics of Noise

#### ノイズとは？

#### What is Noise?

**ノイズ**は、予測不可能なランダムな変動成分です。
**Noise** is an unpredictable, random fluctuation component.

#### 統計的特性

#### Statistical Properties

- **分布**: 正規分布（ガウス分布）N(0, σ²)
- **Distribution**: Normal (Gaussian) distribution N(0, σ²)
- **平均**: 0（ゼロ平均）
- **Mean**: 0 (zero mean)
- **分散**: 0.000975
- **Variance**: 0.000975
- **標準偏差**: √0.000975 ≈ 0.0312（約 3.12%）
- **Standard Deviation**: √0.000975 ≈ 0.0312 (approx. 3.12%)

#### ノイズの役割

#### Role of Noise

1. **完全な予測を防ぐ**
1. **Prevents perfect predictability**

   - ノイズ項により、将来の帯域幅を完全に予測することは不可能
   - The noise term makes it impossible to perfectly predict future bandwidth
   - アルゴリズムは常に探索と適応を続ける必要がある
   - Algorithms must continuously explore and adapt

1. **探索の必要性を保証**
1. **Ensures the necessity of exploration**

   - 確率的変動により、最適解が時間とともに変化する可能性がある
   - Probabilistic fluctuations mean optimal solutions may change over time
   - 探索戦略の重要性が高まる
   - The importance of exploration strategies increases

1. **現実的なネットワークトラフィックの再現**
1. **Replicates realistic network traffic**
   - 実際のネットワークでは、予期しないトラフィック変動が発生する
   - Real networks experience unexpected traffic fluctuations
   - ノイズ項は、このような予測困難な変動をモデル化する
   - The noise term models such unpredictable fluctuations

#### ノイズの影響範囲

#### Noise Impact Range

- **68%信頼区間**: ±3.12%（1 標準偏差）
- **68% Confidence Interval**: ±3.12% (1 standard deviation)
- **95%信頼区間**: ±6.24%（2 標準偏差）
- **95% Confidence Interval**: ±6.24% (2 standard deviations)
- **99.7%信頼区間**: ±9.36%（3 標準偏差）
- **99.7% Confidence Interval**: ±9.36% (3 standard deviations)

---

## 変動の特徴

## Characteristics of Fluctuation

### 1. 高い自己相関（φ = 0.95）

### 1. High Autocorrelation (φ = 0.95)

#### 自己相関の意味

#### Meaning of Autocorrelation

- 現在の値は直前の値に 95%依存している
- The current value is 95% dependent on the previous value
- 急激な変化は起こりにくい（現実のネットワークトラフィックに近い）
- Abrupt changes are unlikely (similar to realistic network traffic)
- 時間的に連続的な変動パターンを示す
- Shows temporally continuous fluctuation patterns

#### 半減期の計算

#### Half-Life Calculation

半減期（影響が半分になるまでの時間）は以下の式で計算されます：
The half-life (time for influence to halve) is calculated by:

```
半減期 = ln(0.5) / ln(φ) ≈ 13.5世代
```

```
Half-life = ln(0.5) / ln(φ) ≈ 13.5 generations
```

実用的には約 14 世代で、直前の値の影響が半分になります。
Practically, the influence of the previous value halves in approximately 14 generations.

#### 自己相関の減衰

#### Autocorrelation Decay

- 1 世代前: 0.95
- 1 generation ago: 0.95
- 5 世代前: 0.95^5 ≈ 0.77
- 5 generations ago: 0.95^5 ≈ 0.77
- 10 世代前: 0.95^10 ≈ 0.60
- 10 generations ago: 0.95^10 ≈ 0.60
- 20 世代前: 0.95^20 ≈ 0.36
- 20 generations ago: 0.95^20 ≈ 0.36

---

### 2. 平均回帰性

### 2. Mean Reversion

#### 平均回帰のメカニズム

#### Mean Reversion Mechanism

AR(1)モデルは、以下の項により平均値への回帰が実現されます：
The AR(1) model achieves mean reversion through the following term:

```
(1 - φ) × 平均利用率 = 0.05 × 0.4 = 0.02
```

```
(1 - φ) × Mean Utilization = 0.05 × 0.4 = 0.02
```

この項により、利用率が平均値から離れると、自動的に平均値に戻る力が働きます。
This term creates a force that automatically returns utilization to the mean when it deviates.

#### 収束速度

#### Convergence Speed

- **時定数（Time Constant）**: 1 / (1 - φ) = 1 / 0.05 = 20 世代
- **Time Constant**: 1 / (1 - φ) = 1 / 0.05 = 20 generations
- **63%収束**: 約 20 世代で初期偏差の 63%が解消される
- **63% Convergence**: Approximately 63% of initial deviation is resolved in 20 generations
- **95%収束**: 約 60 世代で初期偏差の 95%が解消される
- **95% Convergence**: Approximately 95% of initial deviation is resolved in 60 generations

#### 極端な状態の持続時間

#### Duration of Extreme States

- 極端な状態（高帯域・低帯域）が長期間持続しない
- Extreme states (very high or very low bandwidth) do not persist for long periods
- 平均回帰により、長期的には平均利用率 40%に収束する
- Mean reversion causes long-term convergence to 40% mean utilization

---

### 3. 確率的変動

### 3. Stochastic Fluctuation

#### 確率性の重要性

#### Importance of Stochasticity

- ノイズ項により完全な予測は不可能
- Perfect prediction is impossible due to the noise term
- 探索の必要性を保証する
- This guarantees the necessity of exploration
- アルゴリズムの適応能力が試される
- Tests the adaptive capabilities of algorithms

#### 変動の予測可能性

#### Predictability of Fluctuations

- **短期予測**: 高い自己相関により、1 ～ 2 世代先の予測は比較的容易
- **Short-term Prediction**: High autocorrelation makes 1-2 generation ahead predictions relatively easy
- **中期予測**: 5 ～ 10 世代先の予測は、ノイズの累積により不確実性が増す
- **Medium-term Prediction**: 5-10 generation ahead predictions become more uncertain due to noise accumulation
- **長期予測**: 20 世代以上先の予測は、平均値への収束のみが確実
- **Long-term Prediction**: For 20+ generations ahead, only convergence to the mean is certain

---

### 4. 変動範囲の制限

### 4. Fluctuation Range Limit

#### クリッピング処理

#### Clipping Process

利用率は **0.05 ～ 0.95** の範囲にクリップされます：
The utilization rate is **clipped** to the range [0.05, 0.95]:

```python
new_utilization = max(0.05, min(0.95, new_utilization))
```

#### クリッピングの理由

#### Reasons for Clipping

1. **現実的な範囲の維持**
1. **Maintaining realistic ranges**

   - 0%や 100%の利用率は現実的ではない
   - 0% or 100% utilization is unrealistic
   - 常に最小 5%の可用帯域、最大 95%の利用率を保証
   - Always guarantees minimum 5% available bandwidth, maximum 95% utilization

1. **数値的安定性**
1. **Numerical stability**

   - 極端な値による数値計算の不安定性を防ぐ
   - Prevents numerical instability from extreme values
   - アルゴリズムの収束性を保証
   - Guarantees algorithm convergence

1. **シミュレーションの妥当性**
1. **Simulation validity**
   - 現実的なネットワーク条件を反映
   - Reflects realistic network conditions
   - テストケースの妥当性を維持
   - Maintains test case validity

#### クリッピングの影響

#### Impact of Clipping

- 利用率が 0.05 未満になった場合: 0.05 にクリップされ、可用帯域は最大 95%になる
- When utilization falls below 0.05: Clipped to 0.05, available bandwidth becomes maximum 95%
- 利用率が 0.95 を超えた場合: 0.95 にクリップされ、可用帯域は最小 5%になる
- When utilization exceeds 0.95: Clipped to 0.95, available bandwidth becomes minimum 5%

---

## 変動のタイミング

## Fluctuation Timing

### 更新間隔

### Update Interval

- **更新間隔**: `BANDWIDTH_UPDATE_INTERVAL` 世代ごと
- **Update Interval**: Every `BANDWIDTH_UPDATE_INTERVAL` generations
- **現在の設定**: 10000 世代ごと（実質的にほぼ固定）
- **Current Setting**: Every 10,000 generations (i.e., virtually static)

### 更新条件

### Update Condition

帯域幅の更新は、以下の条件を満たす世代でのみ実行されます：
Bandwidth updates are executed only in generations that meet the following condition:

```python
if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
    return False  # 更新なし
```

### 更新間隔の選択

### Update Interval Selection

**現在の設定（10000 世代）**:
**Current Setting (10,000 generations)**:

- 実質的に帯域幅は固定されている
- Bandwidth is effectively fixed
- 静的環境でのアルゴリズム性能を評価するために使用
- Used to evaluate algorithm performance in static environments
- 変動の影響を最小限に抑える
- Minimizes the impact of fluctuations

**動的環境の設定例**:
**Dynamic Environment Setting Examples**:

- `BANDWIDTH_UPDATE_INTERVAL = 1`: 毎世代更新（最も動的）
- `BANDWIDTH_UPDATE_INTERVAL = 1`: Update every generation (most dynamic)
- `BANDWIDTH_UPDATE_INTERVAL = 10`: 10 世代ごと更新
- `BANDWIDTH_UPDATE_INTERVAL = 10`: Update every 10 generations
- `BANDWIDTH_UPDATE_INTERVAL = 100`: 100 世代ごと更新
- `BANDWIDTH_UPDATE_INTERVAL = 100`: Update every 100 generations

---

## 変動対象エッジ

## Target Edges for Fluctuation

### 選択の原則

### Selection Principle

**全エッジではなく、選択されたエッジのみが変動**
**Fluctuation is applied only to selected edges, not all edges**

この設計により、以下の利点があります：
This design provides the following benefits:

1. **計算効率**: 全エッジを更新する必要がない
1. **Computational Efficiency**: No need to update all edges
1. **現実性**: 実際のネットワークでは、すべてのリンクが同時に変動するわけではない
1. **Realism**: In real networks, not all links fluctuate simultaneously
1. **テストの焦点**: 重要なエッジ（ハブノードなど）に焦点を当てる
1. **Test Focus**: Focus on important edges (e.g., hub nodes)

### デフォルト設定

### Default Setting

- **デフォルト設定**: ハブノード（次数の高いノード）に接続するエッジの約 10%
- **Default Setting**: Approx. 10% of edges connected to hub nodes (nodes with high degree)
- **設定パラメータ**: `FLUCTUATION_PERCENTAGE = 0.1`
- **Configuration Parameter**: `FLUCTUATION_PERCENTAGE = 0.1`

### エッジ選択方法

### Edge Selection Methods

選択方法は `EDGE_SELECTION_METHOD` で指定可能です：
The selection method can be specified via `EDGE_SELECTION_METHOD`:

#### 1. ハブノード選択（`"hub"`）- 推奨

#### 1. Hub Node Selection (`"hub"`) - Recommended

- **説明**: 次数の高いノード（ハブノード）に接続するエッジを選択
- **Description**: Selects edges connected to nodes with high degree (hub nodes)
- **選択プロセス**:
- **Selection Process**:

  1. 全ノードの次数（隣接ノード数）を計算
  1. Calculate degree (number of adjacent nodes) for all nodes
  1. 次数の高い順にソート
  1. Sort by degree in descending order
  1. 上位 10%のノードをハブノードとして選択
  1. Select top 10% of nodes as hub nodes
  1. ハブノードに接続するすべてのエッジを変動対象とする
  1. All edges connected to hub nodes become fluctuation targets

- **利点**: ネットワークの重要な部分（ハブ）に焦点を当てる
- **Advantage**: Focuses on important parts of the network (hubs)
- **現実性**: 実際のネットワークでは、ハブノード周辺のトラフィック変動が大きい
- **Realism**: In real networks, traffic fluctuations around hub nodes are significant

#### 2. 完全ランダム選択（`"random"`）

#### 2. Completely Random Selection (`"random"`)

- **説明**: 全エッジからランダムに 10%を選択
- **Description**: Randomly selects 10% of all edges
- **選択タイミング**: `RANDOM_SELECTION_TIMING` で指定可能
- **Selection Timing**: Can be specified via `RANDOM_SELECTION_TIMING`
  - `"fixed"`: シミュレーション開始時に固定（再現可能）
  - `"fixed"`: Fixed at simulation start (reproducible)
  - `"dynamic"`: 毎回ランダムに選択
  - `"dynamic"`: Randomly selected each time

#### 3. 媒介中心性選択（`"betweenness"`）

#### 3. Betweenness Centrality Selection (`"betweenness"`)

- **説明**: エッジ媒介中心性（Edge Betweenness Centrality）が高いエッジを選択
- **Description**: Selects edges with high Edge Betweenness Centrality
- **計算方法**: NetworkX の `edge_betweenness_centrality()` 関数を使用
- **Calculation Method**: Uses NetworkX's `edge_betweenness_centrality()` function
- **利点**: 多くの最短経路が通過する重要なエッジに焦点を当てる
- **Advantage**: Focuses on important edges that many shortest paths traverse

#### 4. 柔軟なハブノード選択（`"hub_partial"`, `"hub_random"`）

#### 4. Flexible Hub Node Selection (`"hub_partial"`, `"hub_random"`)

- **説明**: ハブノードの隣接エッジを部分的に選択
- **Description**: Partially selects adjacent edges of hub nodes
- **パラメータ**: `HUB_NEIGHBOR_EDGE_RATIO` で隣接エッジの選択割合を指定
- **Parameter**: `HUB_NEIGHBOR_EDGE_RATIO` specifies the selection ratio of adjacent edges
- **選択方法**: `HUB_NEIGHBOR_SELECTION_METHOD` で指定
- **Selection Method**: Specified via `HUB_NEIGHBOR_SELECTION_METHOD`
  - `"degree"`: 次数順で選択
  - `"degree"`: Select by degree order
  - `"random"`: ランダムに選択
  - `"random"`: Random selection

### 選択エッジの初期化

### Initialization of Selected Edges

選択されたエッジの初期利用率は、以下のように設定されます：
The initial utilization of selected edges is set as follows:

```python
util_uv = random.uniform(0.3, 0.5)  # エッジ (u, v) の方向
util_vu = random.uniform(0.3, 0.5)  # エッジ (v, u) の方向
```

- 各方向の利用率は独立して 0.3 ～ 0.5 の範囲でランダムに設定される
- Utilization for each direction is independently set randomly in the range 0.3-0.5
- 初期状態から平均利用率（0.4）に近い値から開始する
- Starts from values close to the mean utilization (0.4) from the initial state

---

## 実装の詳細

## Implementation Details

### 実装ファイル

### Implementation Files

#### 設定ファイル

#### Configuration File

- **ファイル**: `src/bandwidth_fluctuation_config.py`
- **File**: `src/bandwidth_fluctuation_config.py`
- **役割**: 帯域変動の全パラメータを一元管理
- **Role**: Centralized management of all bandwidth fluctuation parameters
- **主要パラメータ**:
- **Main Parameters**:
  - `BANDWIDTH_UPDATE_INTERVAL`: 更新間隔（デフォルト: 10000）
  - `BANDWIDTH_UPDATE_INTERVAL`: Update interval (default: 10000)
  - `EDGE_SELECTION_METHOD`: エッジ選択方法（デフォルト: "hub"）
  - `EDGE_SELECTION_METHOD`: Edge selection method (default: "hub")
  - `FLUCTUATION_PERCENTAGE`: 変動対象エッジの割合（デフォルト: 0.1）
  - `FLUCTUATION_PERCENTAGE`: Ratio of fluctuating edges (default: 0.1)
  - `MEAN_UTILIZATION`: 平均利用率（デフォルト: 0.4）
  - `MEAN_UTILIZATION`: Mean utilization (default: 0.4)
  - `AR_COEFFICIENT`: 自己相関係数（デフォルト: 0.95）
  - `AR_COEFFICIENT`: Autocorrelation coefficient (default: 0.95)
  - `NOISE_VARIANCE`: ノイズ分散（デフォルト: 0.000975）
  - `NOISE_VARIANCE`: Noise variance (default: 0.000975)

#### メイン実装関数

#### Main Implementation Functions

**初期化関数**:
**Initialization Function**:

```python
def initialize_ar1_states(
    graph: nx.Graph,
    fluctuating_edges: List[Tuple[int, int]] | None = None
) -> Dict[Tuple[int, int], float]:
```

- 選択されたエッジの AR(1)状態を初期化
- Initializes AR(1) states for selected edges
- 各エッジの初期利用率を 0.3 ～ 0.5 の範囲でランダムに設定
- Randomly sets initial utilization for each edge in the range 0.3-0.5
- 初期可用帯域を計算してグラフに設定
- Calculates and sets initial available bandwidth in the graph

**更新関数**:
**Update Function**:

```python
def update_available_bandwidth_ar1(
    graph: nx.Graph,
    edge_states: Dict[Tuple[int, int], Dict],
    generation: int
) -> bool:
```

- AR(1)モデルに基づいて帯域幅を更新
- Updates bandwidth based on AR(1) model
- `BANDWIDTH_UPDATE_INTERVAL` 世代ごとにのみ実行
- Executes only every `BANDWIDTH_UPDATE_INTERVAL` generations
- 更新された場合は `True` を返す
- Returns `True` if updated

#### モジュール実装（新規）

#### Module Implementation (New)

- **ファイル**: `aco_moo_routing/src/aco_routing/modules/bandwidth_fluctuation.py`
- **File**: `aco_moo_routing/src/aco_routing/modules/bandwidth_fluctuation.py`
- **クラス**: `AR1Model`
- **Class**: `AR1Model`
- **役割**: オブジェクト指向的な実装で、複数の変動モデルに対応
- **Role**: Object-oriented implementation supporting multiple fluctuation models

### 実装の流れ

### Implementation Flow

1. **初期化フェーズ**
1. **Initialization Phase**

   - エッジ選択関数を呼び出して変動対象エッジを決定
   - Call edge selection function to determine fluctuating edges
   - `initialize_ar1_states()` で各エッジの初期状態を設定
   - Set initial state for each edge with `initialize_ar1_states()`
   - グラフの `weight` 属性に初期可用帯域を設定
   - Set initial available bandwidth in graph's `weight` attribute

1. **更新フェーズ（各世代）**
1. **Update Phase (Each Generation)**

   - `update_available_bandwidth_ar1()` を呼び出す
   - Call `update_available_bandwidth_ar1()`
   - 更新間隔をチェック（`generation % BANDWIDTH_UPDATE_INTERVAL == 0`）
   - Check update interval (`generation % BANDWIDTH_UPDATE_INTERVAL == 0`)
   - 条件を満たす場合、AR(1)モデルで利用率を更新
   - If condition is met, update utilization with AR(1) model
   - 可用帯域を再計算してグラフを更新
   - Recalculate available bandwidth and update graph

1. **状態管理**
1. **State Management**

   - 各エッジの利用率状態は `edge_states` 辞書で管理
   - Utilization state for each edge is managed in `edge_states` dictionary
   - キー: `(u, v)` タプル（エッジの方向）
   - Key: `(u, v)` tuple (edge direction)
   - 値: `{"utilization": float}` 辞書
   - Value: `{"utilization": float}` dictionary

### グラフ属性の更新

### Graph Attribute Updates

更新時に以下のグラフ属性が更新されます：
The following graph attributes are updated during updates:

- `graph[u][v]["weight"]`: 可用帯域（10Mbps 刻み）
- `graph[u][v]["weight"]`: Available bandwidth (rounded to 10Mbps)
- `graph[u][v]["local_min_bandwidth"]`: 現在の可用帯域（最小値として）
- `graph[u][v]["local_min_bandwidth"]`: Current available bandwidth (as minimum)
- `graph[u][v]["local_max_bandwidth"]`: 現在の可用帯域（最大値として）
- `graph[u][v]["local_max_bandwidth"]`: Current available bandwidth (as maximum)
- `graph[u][v]["original_weight"]`: 元のキャパシティ（変更されない）
- `graph[u][v]["original_weight"]`: Original capacity (unchanged)

---

## 統計的特性

## Statistical Properties

### 定常分布

### Stationary Distribution

長期的には、利用率は以下の正規分布に従います：
In the long term, utilization follows the following normal distribution:

```
利用率 ~ N(μ, σ²/(1 - φ²))
      ~ N(0.4, 0.000975/(1 - 0.95²))
      ~ N(0.4, 0.01026)
```

```
Utilization ~ N(μ, σ²/(1 - φ²))
           ~ N(0.4, 0.000975/(1 - 0.95²))
           ~ N(0.4, 0.01026)
```

- **平均**: 0.4（40%）
- **Mean**: 0.4 (40%)
- **標準偏差**: √0.01026 ≈ 0.101（約 10.1%）
- **Standard Deviation**: √0.01026 ≈ 0.101 (approx. 10.1%)

### 変動の大きさ

### Magnitude of Fluctuation

- **短期変動（1 世代）**: ノイズの標準偏差 ≈ ±3.12%
- **Short-term Fluctuation (1 generation)**: Noise standard deviation ≈ ±3.12%
- **長期変動（定常分布）**: 標準偏差 ≈ ±10.1%
- **Long-term Fluctuation (stationary distribution)**: Standard deviation ≈ ±10.1%
- **95%信頼区間**: 約 0.2 ～ 0.6（20%～ 60%の利用率）
- **95% Confidence Interval**: Approximately 0.2-0.6 (20%-60% utilization)

### 自己共分散関数

### Autocovariance Function

k 次ラグの自己共分散は以下の式で与えられます：
The autocovariance at lag k is given by:

```
γ(k) = σ² × φ^k / (1 - φ²)
     = 0.000975 × 0.95^k / (1 - 0.95²)
```

```
γ(k) = σ² × φ^k / (1 - φ²)
     = 0.000975 × 0.95^k / (1 - 0.95²)
```

- **k=0（分散）**: ≈ 0.01026
- **k=0 (variance)**: ≈ 0.01026
- **k=1（1 次ラグ）**: ≈ 0.00975
- **k=1 (first-order lag)**: ≈ 0.00975
- **k=10（10 次ラグ）**: ≈ 0.00588
- **k=10 (10th-order lag)**: ≈ 0.00588

---

## 簡潔な説明（質問されたとき用）

## Concise Explanation (For Q&A)

### Q: このネットワークはどのように変動するか？

### Q: How does this network fluctuate?

**A**: AR(1)モデル（1 次自己回帰モデル）により、各エッジの利用率を更新し、それを可用帯域に変換して変動させます。
**A**: It uses an AR(1) model to update the "utilization rate" of each edge, which is then converted into "available bandwidth."

#### 1. 利用率の更新

#### 1. Utilization Update

```
利用率(t+1) = 0.02 + 0.95 × 利用率(t) + ノイズ
```

```
Utilization(t+1) = 0.02 + 0.95 × Utilization(t) + Noise
```

- 直前の値に 95%依存（高い自己相関）
- Highly autocorrelated (95% dependent on the previous value)
  - ランダムノイズで予測不能な変動も含む
- Includes random noise for unpredictable changes
- 長期的には平均利用率 40%に収束
- Reverts to a 40% mean utilization over the long term

#### 2. 可用帯域の計算

#### 2. Available Bandwidth Calculation

```
可用帯域 = キャパシティ × (1 - 利用率)
```

```
Available Bandwidth = Capacity × (1 - Utilization)
```

- 利用率が上がる → 可用帯域が減る
- Higher utilization → Lower available bandwidth
  - 利用率が下がる → 可用帯域が増える
- Lower utilization → Higher available bandwidth

#### 3. 変動タイミング

#### 3. Fluctuation Timing

- **現在の設定**: 10000 世代ごと（実質的にほぼ固定）
- **Current Setting**: Every 10,000 generations (i.e., virtually static)
- **動的環境の例**: 1 世代ごと、10 世代ごとなど
- **Dynamic Environment Examples**: Every generation, every 10 generations, etc.

#### 4. 変動対象

#### 4. Fluctuation Target

- **デフォルト**: 全エッジの約 10%（ハブノードに接続するエッジ）
- **Default**: Approx. 10% of all edges (connected to hub nodes)
- **選択方法**: ハブノード選択、ランダム選択、媒介中心性選択など
- **Selection Methods**: Hub node selection, random selection, betweenness centrality selection, etc.

---

## 参考文献・関連ドキュメント

## References and Related Documents

- **実装ファイル**: `src/bandwidth_fluctuation_config.py`
- **Implementation File**: `src/bandwidth_fluctuation_config.py`
- **モジュール実装**: `aco_moo_routing/src/aco_routing/modules/bandwidth_fluctuation.py`
- **Module Implementation**: `aco_moo_routing/src/aco_routing/modules/bandwidth_fluctuation.py`
- **関連ドキュメント**:
- **Related Documents**:
  - `docs/20251122/bandwidth_fluctuation_and_learning_methods.md`
  - `docs/20251122/fluctuation_models_catalog.md`
  - `docs/20251122/ar1_clarification.md`
