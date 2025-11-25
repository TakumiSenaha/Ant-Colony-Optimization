# パラメータ比較: 既存実装 vs 新実装

## 📊 パラメータ比較表

| パラメータ                                      | 既存実装 (`aco_main_bkb_available_bandwidth.py`) | 新実装 (`config.yaml`) | 一致？          |
| ----------------------------------------------- | ------------------------------------------------ | ---------------------- | --------------- |
| **ACO 基本パラメータ**                          |
| `ALPHA` / `alpha`                               | 1.0                                              | 1.0                    | ✅              |
| `BETA` / `beta_bandwidth`                       | 1.0                                              | 1.0                    | ✅              |
| `EPSILON` / `epsilon`                           | 0.1                                              | 0.1                    | ✅              |
| `ANT_NUM` / `num_ants`                          | 10                                               | 10                     | ✅              |
| `GENERATION` / `generations`                    | 1000                                             | 1000                   | ✅              |
| `SIMULATIONS` / `simulations`                   | 100                                              | 100                    | ✅              |
| `TTL` / `ttl`                                   | 100                                              | 100                    | ✅              |
| **フェロモンパラメータ**                        |
| `V` (残存率) / `evaporation_rate` (揮発率)      | 0.98 (残存率)                                    | 0.02 (揮発率)          | ✅ 同じ         |
| `MIN_F` / `min_pheromone`                       | 100                                              | 100                    | ✅              |
| `MAX_F` / `max_pheromone`                       | 1000000000                                       | 1000000000             | ✅              |
| **BKB 学習パラメータ**                          |
| `TIME_WINDOW_SIZE` / `bkb_window_size`          | 10                                               | 10                     | ✅              |
| `PENALTY_FACTOR` / `penalty_factor`             | 0.5                                              | 0.5                    | ✅              |
| `BKB_EVAPORATION_RATE` / `bkb_evaporation_rate` | 0.999 (残存率)                                   | 0.001 (揮発率)         | ✅ 同じ         |
| `ACHIEVEMENT_BONUS` / `bonus_factor`            | 1.5                                              | 1.5                    | ✅              |
| `VOLATILIZATION_MODE` / `volatilization_mode`   | 3                                                | 3                      | ✅              |
| **グラフ生成パラメータ**                        |
| `num_nodes`                                     | 100                                              | 100                    | ✅              |
| `num_edges` (BA model)                          | 6                                                | 6                      | ✅ **修正済み** |
| `bandwidth_range`                               | `[10, 150]` (整数)                               | `[10, 150]` (整数)     | ✅ **修正済み** |
| `set_pheromone_min_max_by_degree_and_width`     | あり                                             | あり                   | ✅ **実装済み** |
| **帯域変動パラメータ**                          |
| `FLUCTUATION_MODEL`                             | "ar1"                                            | "ar1"                  | ✅              |
| `EDGE_SELECTION_METHOD`                         | "hub"                                            | "hub"                  | ✅              |
| `FLUCTUATION_PERCENTAGE`                        | 0.1                                              | 0.1                    | ✅              |

## ⚠️ 重要な違い

### 1. **グラフ生成パラメータ**

**既存実装**:

```python
graph = ba_graph(num_nodes=NUM_NODES, num_edges=6, lb=1, ub=15)
# weight = random.randint(1, 15) * 10  # [10, 150] の整数値
```

**新実装**:

```yaml
num_edges: 3 # ⚠️ 既存は6
bandwidth_range: [10, 150] # 浮動小数点
```

**影響**:

- グラフの構造が異なる（エッジ数が少ない = より疎なグラフ）
- 帯域幅の分布が異なる（整数 vs 浮動小数点）

### 2. **BKB 揮発率の扱い**

**既存実装**:

```python
BKB_EVAPORATION_RATE = 0.999  # 残存率
evaporate_bkb_values(graph, BKB_EVAPORATION_RATE)
# 実装: new_value = bkb * 0.999
```

**新実装**:

```yaml
bkb_evaporation_rate: 0.001 # 揮発率
# 実装: self.bkb = self.bkb * (1.0 - 0.001) = self.bkb * 0.999
```

**結論**: ✅ **実質的に同じ**（0.999 = 1 - 0.001）

### 3. **フェロモン最小値・最大値の設定**

**既存実装**:

```python
set_pheromone_min_max_by_degree_and_width(graph)
# ノードの次数に応じてmin_pheromoneを調整
# graph[u][v]["min_pheromone"] = MIN_F * 3 // degree_u
```

**新実装**:

```python
# 固定値を使用（次数による調整なし）
graph.edges[u, v]["min_pheromone"] = min_pheromone
```

**影響**: 新実装では次数による調整がない

## ✅ 修正完了

### 1. `num_edges`を 6 に変更 ✅

```yaml
graph:
  num_edges: 6 # 既存実装と同じ
```

### 2. 帯域幅生成方法を整数に統一 ✅

既存実装: `random.randint(1, 15) * 10` → 10 刻みの整数値 [10, 20, 30, ..., 150]
新実装: 同じ方法を実装

### 3. フェロモン最小値・最大値の設定を追加 ✅

既存実装の`set_pheromone_min_max_by_degree_and_width`を新実装にも実装済み：

- フェロモン最小値: `MIN_F * 3 // degree`（次数に応じて調整）
- フェロモン最大値: `bandwidth^5`（帯域幅に基づく）

## ✅ 結論

**修正後、既存実装と同じシミュレーションになります。**

**修正内容**:

1. ✅ `num_edges: 6`に変更
2. ✅ 帯域幅生成を整数値（10 刻み）に統一
3. ✅ フェロモン最小値・最大値の設定方法を統一

**注意**: `target_objectives: ["bandwidth"]`の場合、ヒューリスティック計算も既存実装と同じになります（`aco_solver.py`で実装済み）。
