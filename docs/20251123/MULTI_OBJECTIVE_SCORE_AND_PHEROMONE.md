# 多目的最適化におけるスコア計算とフェロモン更新の仕組み

## 📊 目次

1. [BKB/BLD/BKH の扱い](#bkbbldbkhの扱い)
2. [スコア計算](#スコア計算)
3. [フェロモン付加](#フェロモン付加)
4. [ボーナス判定](#ボーナス判定)
5. [ペナルティ判定](#ペナルティ判定)

---

## BKB/BLD/BKH の扱い

### ノードの学習値

各ノードは、自分を通ってゴールしたアリの観測値をリングバッファ（サイズ 10）で学習します。

- **BKB (Best Known Bottleneck)**: バッファ内の最大帯域値

  - `BKB = max(bandwidth_buffer)`
  - 帯域は 10Mbps 刻みなので、`int()`で変換してから保存

- **BLD (Best Known Lowest Delay)**: バッファ内の最小遅延値

  - `BLD = min(delay_buffer)`
  - 初期値: `float("inf")`

- **BKH (Best Known Hops)**: バッファ内の最小ホップ数
  - `BKH = min(hops_buffer)`
  - 初期値: `float("inf")`

### 多目的最適化での扱い

多目的最適化では、**BKB/BLD/BKH を組み合わせてスコアを計算**します。

- 帯域のみ最適化: `BKB`のみ使用
- 帯域+ホップ数: `BKB`と`BKH`を使用
- 帯域+遅延: `BKB`と`BLD`を使用
- 帯域+遅延+ホップ数: `BKB`、`BLD`、`BKH`をすべて使用

---

## スコア計算

### 評価関数 f(B, D, H)

スコアは、目的関数の組み合わせに応じて以下の式で計算されます。

#### 1. 帯域のみ最適化

**注意**: 帯域のみ最適化の場合、**スコア**と**フェロモン付加量**は別の概念です。

**スコア（比較用）**:

```
score = bandwidth
```

**定性的な意味**: スコアは純粋にボトルネック帯域そのもの。BKB も純粋にボトルネック帯域。

**フェロモン付加量**:

```
Δτ = bandwidth × 10  (ボーナスなし)
Δτ = bandwidth × 10 × bonus_factor  (ボーナスあり)
```

**定性的な意味**: フェロモン付加量は既存実装との互換性のため、スコアに 10 をかける。スコアとは別の値。

#### 2. 帯域 + ホップ数

```
score = bandwidth / hops
```

**定性的な意味**: 帯域が大きく、ホップ数が少ないほど良い。効率性（帯域/ホップ）を評価。

**例**:

- `(bandwidth=100, hops=4)` → `score = 100/4 = 25.0`
- `(bandwidth=80, hops=3)` → `score = 80/3 = 26.67` ✅ より良い

#### 3. 帯域 + 遅延

```
score = bandwidth / delay
```

**定性的な意味**: 帯域が大きく、遅延が小さいほど良い。スループット効率（帯域/遅延）を評価。

**例**:

- `(bandwidth=100, delay=20)` → `score = 100/20 = 5.0`
- `(bandwidth=80, delay=15)` → `score = 80/15 = 5.33` ✅ より良い

#### 4. 帯域 + 遅延 + ホップ数

```
score = bandwidth / (delay × hops)
```

**定性的な意味**: 帯域が大きく、遅延とホップ数が小さいほど良い。総合的な効率性を評価。

**例**:

- `(bandwidth=100, delay=20, hops=4)` → `score = 100/(20×4) = 1.25`
- `(bandwidth=80, delay=15, hops=3)` → `score = 80/(15×3) = 1.78` ✅ より良い

### スコア計算の実装

```python
# evaluator.py
def evaluate(self, bandwidth: float, delay: float, hops: int) -> float:
    if objectives == ["bandwidth"]:
        return bandwidth * 10.0
    elif set(objectives) == {"bandwidth", "hops"}:
        return bandwidth / hops
    elif set(objectives) == {"bandwidth", "delay"}:
        return bandwidth / delay
    elif set(objectives) == {"bandwidth", "delay", "hops"}:
        return bandwidth / (delay * hops)
```

---

## フェロモン付加

### 基本式

アリがゴールに到達した際、経路上の各エッジにフェロモンを付加します。

**すべての最適化で統一**:

```
Δτ = score × 10 × bonus_factor  (ボーナスありの場合)
Δτ = score × 10                 (ボーナスなしの場合)
```

**具体例**:

- 帯域のみ: `Δτ = bandwidth × 10` (スコア = bandwidth)
- 帯域+ホップ数: `Δτ = (bandwidth / hops) × 10` (スコア = bandwidth / hops)
- 帯域+遅延: `Δτ = (bandwidth / delay) × 10` (スコア = bandwidth / delay)
- 帯域+遅延+ホップ数: `Δτ = (bandwidth / (delay × hops)) × 10` (スコア = bandwidth / (delay × hops))

**注意**: `evaluate()` は純粋なスコアを返し、フェロモン付加時に `× 10` をかけることで統一しています。

### フェロモン更新

```
τ_new = τ_old + Δτ
```

ただし、最小値・最大値の制約があります：

```
τ_min ≤ τ_new ≤ τ_max
```

### 双方向更新

エッジ `(u, v)` と `(v, u)` の両方に同じ量のフェロモンを付加します。

### 実装

```python
# pheromone.py
score = self.evaluator.evaluate(bandwidth, delay, hops)

if self.evaluator.check_bonus_condition(ant_solution, node_memory):
    delta_pheromone = score * self.bonus_factor  # 通常は1.5倍
else:
    delta_pheromone = score

graph.update_pheromone(u, v, delta_pheromone, bidirectional=True)
```

---

## ボーナス判定

### 判定条件

ボーナスは、**アリの解のスコアがノードの記憶値のスコアより良い場合**に付与されます。

#### 1. 帯域のみ最適化

```
ボーナス条件: bandwidth_ant ≥ BKB
```

**定性的な意味**: アリが見つけた帯域が、ノードが知っている最高値（BKB）以上であればボーナス。**スコアは使わず、純粋に帯域値で比較**します。

**注意**: BKB は純粋にボトルネック帯域そのもので、スコア（`bandwidth × 10`）ではありません。

#### 2. 多目的最適化

```
ボーナス条件: score_ant > score_memory
```

**定性的な意味**: アリの解のスコアが、ノードの記憶値（BKB/BLD/BKH）から計算したスコアより良い場合にボーナス。

### スコア比較の詳細

```python
# アリの解のスコア
score_ant = evaluate(bandwidth_ant, delay_ant, hops_ant)

# ノードの記憶値のスコア（仮想的な解として）
score_memory = evaluate(BKB, BLD, BKH)

# ボーナス判定
if score_ant > score_memory:
    bonus = True
```

### 特殊ケース（inf の扱い）

- `BLD == inf` または `BKH == inf` の場合:
  - まだ観測されていない目的関数がある
  - 観測されている目的関数のみでスコアを計算
  - 例: `BLD == inf` の場合、`score_memory = evaluate_bandwidth_hops(BKB, BKH)`

### 実装

```python
# evaluator.py
def check_bonus_condition(self, ant_solution, node_memory):
    b_ant, d_ant, h_ant = ant_solution
    k_j, l_j, m_j = node_memory  # BKB, BLD, BKH

    if self.target_objectives == ["bandwidth"]:
        return b_ant >= k_j

    # 多目的最適化: スコア比較
    score_ant = self.evaluate(b_ant, d_ant, h_ant)
    score_memory = self.evaluate(k_j, l_j, int(m_j))

    return score_ant > score_memory
```

---

## ペナルティ判定

### 揮発率の調整

フェロモン揮発時に、エッジの属性がノードの記憶値より悪い場合、揮発率を上げます（ペナルティ）。

### 基本揮発

```
τ_new = τ_old × (1 - evaporation_rate)
```

通常の揮発率: `evaporation_rate = 0.02`（2%揮発）

### ペナルティ付き揮発

#### 判定条件

```
ペナルティ条件: edge_bandwidth < BKB
```

**定性的な意味**: エッジの帯域が BKB より低い場合、そのエッジを使った経路が BKB を超えることは数学的に不可能。そのため、揮発を促進してフェロモンを減らす。

#### 揮発率の計算

```
if edge_bandwidth < BKB:
    evaporation = 1.0 - (1.0 - base_evaporation) × penalty_factor
else:
    evaporation = base_evaporation
```

**例**:

- `base_evaporation = 0.02`（基本揮発率）
- `penalty_factor = 0.5`（ペナルティ係数）
- ペナルティあり: `evaporation = 1.0 - 0.98 × 0.5 = 0.51`（51%揮発）
- ペナルティなし: `evaporation = 0.02`（2%揮発）

#### 残存率の計算

```
retention_rate = 1.0 - evaporation
τ_new = floor(τ_old × retention_rate)
```

**例**:

- ペナルティあり: `retention_rate = 0.49` → フェロモンが 51%減る
- ペナルティなし: `retention_rate = 0.98` → フェロモンが 2%減る

### 実装

```python
# pheromone.py
def _evaporate_with_bkb_penalty(self, graph):
    for u, v in graph.graph.edges():
        edge_bandwidth = graph.graph.edges[u, v]["bandwidth"]
        bkb_u = graph[u].bkb

        if edge_bandwidth < bkb_u:
            # ペナルティあり
            evaporation = 1.0 - (1.0 - base_evaporation) * penalty_factor
        else:
            # ペナルティなし
            evaporation = base_evaporation

        retention_rate = 1.0 - evaporation
        new_pheromone = floor(current * retention_rate)
        graph.graph.edges[u, v]["pheromone"] = new_pheromone
```

---

## 📝 まとめ

### スコア計算

| 目的関数           | スコア式      | 意味                          | 備考                                            |
| ------------------ | ------------- | ----------------------------- | ----------------------------------------------- |
| 帯域のみ           | `B`           | 帯域が大きいほど良い          | スコアは純粋に帯域値。フェロモン付加は `B × 10` |
| 帯域+ホップ数      | `B / H`       | 効率性（帯域/ホップ）         |                                                 |
| 帯域+遅延          | `B / D`       | スループット効率（帯域/遅延） |                                                 |
| 帯域+遅延+ホップ数 | `B / (D × H)` | 総合効率性                    |                                                 |

### フェロモン付加

```
Δτ = score × 10 × bonus_factor  (ボーナスあり)
Δτ = score × 10                 (ボーナスなし)
```

**注意**: すべての最適化で、スコアに `× 10` をかけてフェロモン付加量を統一しています。

### ボーナス判定

- **帯域のみ**: `bandwidth_ant ≥ BKB`（スコアは使わず、純粋に帯域値で比較）
- **多目的**: `score_ant > score_memory`（スコア比較）

### ペナルティ判定

- **条件**: `edge_bandwidth < BKB`
- **効果**: 揮発率を上げる（`evaporation = 0.51` vs `0.02`）

---

## 🔍 参考実装ファイル

- `src/aco_routing/modules/evaluator.py`: スコア計算とボーナス判定
- `src/aco_routing/modules/pheromone.py`: フェロモン付加と揮発
- `src/aco_routing/core/node.py`: BKB/BLD/BKH の学習
