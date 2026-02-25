# 🔴 重大な違い（結果に影響する可能性が非常に高い）

## 1. **フェロモン揮発の計算式の違い**

### aco_sim_caching_model_eval.py (apply_volatilization, 137-150 行目)

```python
elif volatilization_mode == 3:
    rate = base_evaporation_rate  # base_evaporation_rate = V = 0.98（残存率）

    if weight_uv < bkb_u:
        rate *= penalty_factor  # rate = 0.98 * 0.5 = 0.49（残存率）

    new_pheromone = math.floor(current_pheromone * rate)
    # ペナルティ時: new_pheromone = math.floor(current * 0.49)
```

**重要**: `base_evaporation_rate`は**残存率**（0.98）として扱われている

---

### aco_solver.py (\_evaporate_with_bkb_penalty, 131-170 行目)

```python
base_evaporation = self.evaporation_rate  # 0.02（揮発率）

if edge_bandwidth < bkb_u:
    evaporation = 1.0 - (1.0 - base_evaporation) * self.penalty_factor
    # evaporation = 1.0 - (1.0 - 0.02) * 0.5 = 1.0 - 0.49 = 0.51（揮発率）
else:
    evaporation = base_evaporation  # 0.02（揮発率）

retention_rate = 1.0 - evaporation  # 1.0 - 0.51 = 0.49（残存率）

new_pheromone = math.floor(current * retention_rate)
# ペナルティ時: new_pheromone = math.floor(current * 0.49)
```

**重要**: `evaporation_rate`は**揮発率**（0.02）として扱われている

---

### 問題点

**計算結果は同じになるはず**ですが、**パラメータの意味が逆**です：

- `aco_sim_caching_model_eval.py`: `V = 0.98` = **残存率**（98%残る = 2%揮発）
- `aco_solver.py`: `evaporation_rate: 0.02` = **揮発率**（2%揮発 = 98%残る）

**ただし、`volatilize_by_width`の呼び出しを確認**:

- `aco_sim_caching_model_eval.py (370行目)`:

  ```python
  volatilize_by_width(
      graph,
      volatilization_mode=VOLATILIZATION_MODE,
      base_evaporation_rate=V,  # V = 0.98（残存率）
      penalty_factor=PENALTY_FACTOR,
      adaptive_rate_func=None,
  )
  ```

- `aco_solver.py`は`volatilize_by_width`を**直接呼び出していない**
- 代わりに`_evaporate_with_bkb_penalty`を使用

**確認が必要**: `aco_solver.py`の`_evaporate_with_bkb_penalty`が正しく実装されているか？

---

## 2. **フェロモン付加量の計算方法**

### aco_sim_caching_model_eval.py (calculate_pheromone_increase_simple, 28 行目)

```python
return float(bottleneck_bandwidth * 10)
```

### aco_solver.py (PheromoneUpdater.update_from_ant, 58-68 行目)

```python
score = self.evaluator.evaluate(bandwidth, delay, hops)
base_pheromone = score * 10.0
```

**帯域のみ最適化の場合**: `score = bandwidth`なので、実質的に同じ

---

## 3. **ボーナス判定の方法**

### aco_sim_caching_model_eval.py (update_pheromone, 309-311 行目)

```python
k_v = node_old_bkb.get(v, 0)  # ノードvの記憶値（更新前の値）
if bottleneck_bn >= k_v:  # B ≥ K_j の場合、ボーナスあり
    pheromone_increase *= achievement_bonus
```

### aco_solver.py (PheromoneUpdater.update_from_ant, 63-77 行目)

```python
k_v, l_v, m_v = node_old_memory[v]
ant_solution = (bandwidth, delay, hops)
node_memory = (k_v, l_v, m_v)

if self.evaluator.check_bonus_condition(
    ant_solution, node_memory, self.delay_tolerance
):
    delta_pheromone = base_pheromone * self.bonus_factor
```

**帯域のみ最適化の場合**: `check_bonus_condition`は`b_ant >= k_j`を返すので、実質的に同じ

---

## 4. **BKB 更新のタイミング**

### aco_sim_caching_model_eval.py (update_pheromone, 261-267 行目)

```python
# BKBの更新を先に行う（フェロモン付加の前に）
node_old_bkb: dict[int, float] = {}
for node in ant.route:
    old_bkb = graph.nodes[node].get("best_known_bottleneck", 0)
    node_old_bkb[node] = old_bkb  # 更新前の値を記録
    bkb_update_func(graph, node, float(bottleneck_bn), generation)
```

### aco_solver.py (PheromoneUpdater.update_from_ant, 40-51 行目)

```python
# Step 1: 各ノードの学習値（BKB/BLD/BKH）を更新し、更新前の値を記録
node_old_memory: Dict[int, Tuple[float, float, float]] = {}
for node in ant.route:
    node_old_memory[node] = (
        graph[node].bkb,
        graph[node].bld,
        graph[node].bkh,
    )
    graph[node].update_all(bandwidth, delay, hops)
```

**同じ順序**: BKB 更新 → フェロモン付加

---

## 5. **フェロモン揮発の対象エッジ**

### aco_sim_caching_model_eval.py (volatilize_by_width, 191-211 行目)

```python
for u, v in graph.edges():
    # u → v の揮発計算
    apply_volatilization(...)
    # v → u の揮発計算
    apply_volatilization(...)
```

### aco_solver.py (\_evaporate_with_bkb_penalty, 124 行目)

```python
for u, v in graph.graph.edges():
    # エッジの属性を取得
    edge_bandwidth = graph.graph.edges[u, v]["bandwidth"]
    # ノードuの学習値（BKB）
    bkb_u = graph[u].bkb
    # ...
```

**問題**: `aco_solver.py`は**双方向の揮発を処理していない**可能性がある！

`graph.edges()`は無向グラフのエッジを返すが、フェロモンは双方向に存在する。
`aco_sim_caching_model_eval.py`は明示的に`(u, v)`と`(v, u)`の両方を処理しているが、
`aco_solver.py`は`(u, v)`のみを処理している可能性がある。

---

## 📝 結論

**最も重要な違い**:

1. **🔴 フェロモン揮発の双方向処理（修正済み）**:

   - `aco_sim_caching_model_eval.py`: `(u, v)`と`(v, u)`の両方を明示的に処理
   - `aco_solver.py`: **修正前は`graph.graph.edges()`をループしているだけで、双方向の処理が不足していた**
   - **修正後**: `(u, v)`と`(v, u)`の両方を明示的に処理するように変更

2. **パラメータの意味**: `base_evaporation_rate`（残存率）vs `evaporation_rate`（揮発率）の違い（計算結果は同じになるはず）

---

## ✅ 修正内容

`aco_solver.py`の`_evaporate_with_bkb_penalty`を修正し、`aco_sim_caching_model_eval.py`と同じ方法で双方向を明示的に処理するように変更しました。
