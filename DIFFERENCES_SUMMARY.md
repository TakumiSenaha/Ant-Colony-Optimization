# aco_sim_caching_model_eval.py と aco_solver.py の違いまとめ

## 🔴 重要な違い（結果に影響する可能性が高い）

### 1. **BKB リセット（スタートノード切り替え時）**

- **aco_sim_caching_model_eval.py (325-326 行目)**:

  ```python
  for node in graph.nodes():
      graph.nodes[node]["best_known_bottleneck"] = 0
  ```

  → スタートノード切り替え時に**明示的に BKB を 0 にリセット**

- **aco_solver.py**:
  → BKB リセット処理が**存在しない**
  → ユーザーの指示により「BKB をリセットしない（バッファと揮発で忘却）」としている

**影響**: スタートノード切り替え後の学習速度に影響する可能性

---

### 2. **BKB 更新方法**

- **aco_sim_caching_model_eval.py (75-80 行目)**:

  ```python
  def _bkb_update_simple_max(graph, node, bottleneck, generation):
      current_bkb = graph.nodes[node].get("best_known_bottleneck", 0)
      graph.nodes[node]["best_known_bottleneck"] = max(current_bkb, bottleneck)
  ```

  → **単純な max 手法**（常に最大値を保持）

- **aco_solver.py (NodeLearning.update_bandwidth)**:
  ```python
  self.bandwidth_buffer.append(bottleneck)
  time_window_max = max(self.bandwidth_buffer) if self.bandwidth_buffer else 0.0
  self.bkb = float(int(time_window_max))
  ```
  → **リングバッファ手法**（`bkb_window_size: 1000`）
  → ただし、`bkb_window_size: 1000` = 世代数なので、実質的に同じ動作になる可能性

**影響**: リングバッファサイズが小さい場合、古い情報が忘れられる

---

### 3. **ログ形式**

- **aco_sim_caching_model_eval.py (142 行目)**:

  ```python
  ant_log.append(1 if min(ant.width) >= current_optimal_bottleneck else 0)
  ```

  → `1` = 最適解、`0` = 非最適解/失敗

- **aco_solver.py (329-333 行目)**:
  ```python
  log_value = (
      0 if solution_bandwidth >= current_optimal_bottleneck else -1
  )
  ```
  → `0` = 最適解、`-1` = 非最適解

**影響**: 解析スクリプトが`idx >= 0`で最適解を判定するため、実質的に問題なし

---

## 🟡 中程度の違い（実装の違いだが動作は同じ可能性）

### 4. **ゴールノードの管理方法**

- **aco_sim_caching_model_eval.py**: `set`を使用

  ```python
  goal_nodes = {initial_provider_node}  # set
  goal_nodes.add(previous_start)
  ```

- **aco_solver.py**: `list`を使用
  ```python
  goal_nodes = [goal_node]  # list
  goal_nodes.append(previous_start)
  ```

**影響**: 動作は同じだが、重複チェックの方法が異なる

---

### 5. **フェロモン最小値の計算**

- **aco_sim_caching_model_eval.py (58 行目)**:

  ```python
  graph[u][v]["min_pheromone"] = MIN_F * 3 // degree_u
  ```

- **aco_solver.py (\_set_pheromone_min_max_by_degree_and_width)**:
  ```python
  graph[u][v]["min_pheromone"] = base_min_pheromone * 3 // degree_u
  ```
  → `base_min_pheromone = MIN_F = 100`なので、実質的に同じ

**影響**: なし（同じ計算）

---

### 6. **フェロモン最大値の計算**

- **aco_sim_caching_model_eval.py (65 行目)**:

  ```python
  graph[u][v]["max_pheromone"] = width_u_to_v**5
  ```

- **aco_solver.py**:
  ```python
  graph[u][v]["max_pheromone"] = bandwidth**5
  ```
  → `width_u_to_v` = `bandwidth`なので、実質的に同じ

**影響**: なし（同じ計算）

---

### 7. **アリの処理順序**

- **aco_sim_caching_model_eval.py (94 行目)**:

  ```python
  for ant in reversed(ant_list):
  ```

  → 逆順で処理（安全な削除のため）

- **aco_solver.py (281 行目)**:
  ```python
  for ant in list(active_ants):
  ```
  → 通常順で処理（コピーを作成して安全に削除）

**影響**: なし（アルゴリズム的には同じ）

---

## 🟢 同じ（確認済み）

### 8. **パラメータ値**

- ✅ `ACHIEVEMENT_BONUS = 2.0` vs `bonus_factor: 2.0` → 同じ
- ✅ `V = 0.98` (残存率) vs `evaporation_rate: 0.02` (揮発率) → 同じ
- ✅ `PENALTY_FACTOR = 0.5` vs `penalty_factor: 0.5` → 同じ
- ✅ `BKB_EVAPORATION_RATE = 0.999` (残存率) vs `bkb_evaporation_rate: 0.001` (揮発率) → 同じ
- ✅ `ALPHA = 1.0` vs `alpha: 1.0` → 同じ
- ✅ `BETA = 1.0` vs `beta_bandwidth: 1.0` → 同じ
- ✅ `EPSILON = 0.1` vs `epsilon: 0.1` → 同じ
- ✅ `TTL = 100` vs `ttl: 100` → 同じ
- ✅ `ANT_NUM = 10` vs `num_ants: 10` → 同じ

### 9. **フェロモン更新タイミング**

- ✅ 両方とも「アリがゴールに到達した時点で即座に更新」（完全分散方式）

### 10. **フェロモン揮発タイミング**

- ✅ 両方とも「各世代の終了時に揮発」

### 11. **BKB 揮発タイミング**

- ✅ 両方とも「各世代の終了時に揮発」

### 12. **最適解の計算方法**

- ✅ 両方とも`max_load_path`と`bottleneck_capacity`を使用

### 13. **最適解の判定方法**

- ✅ 両方とも`>=`で比較

### 14. **スタートノードの選択方法**

- ✅ 両方とも「初期ゴールノードを除外してから 10 個選択」

### 15. **ゴールノードの初期化**

- ✅ 両方とも「最初から初期ゴールノードを含める」

### 16. **最適解の計算タイミング**

- ✅ 両方とも「スタートノード切り替え時に計算」

### 17. **最適解の取得方法**

- ✅ 両方とも「辞書から取得」

---

## 📝 結論

**主な違いは 3 つ**:

1. **BKB リセット**: `aco_sim_caching_model_eval.py`はリセット、`aco_solver.py`はリセットしない
2. **BKB 更新方法**: `aco_sim_caching_model_eval.py`は単純 max、`aco_solver.py`はリングバッファ（ただし`bkb_window_size: 1000`で実質同じ）
3. **ログ形式**: `aco_sim_caching_model_eval.py`は`1/0`、`aco_solver.py`は`0/-1`（解析スクリプトは`idx >= 0`で判定するため問題なし）

**最も重要な違いは「BKB リセット」**です。`aco_sim_caching_model_eval.py`ではスタートノード切り替え時に BKB を 0 にリセットしていますが、`aco_solver.py`ではリセットしていません。これが結果の違いの原因である可能性が高いです。








