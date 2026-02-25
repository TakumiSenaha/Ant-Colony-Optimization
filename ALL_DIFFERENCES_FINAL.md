# aco_sim_caching_model_eval.py と aco_solver.py の全違いまとめ（最終版）

## 🔴 重大な違い（結果に影響する可能性が非常に高い）

### 1. **フェロモン揮発の双方向処理（修正済み）**

- **aco_sim_caching_model_eval.py (volatilize_by_width, 191-211 行目)**:

  ```python
  for u, v in graph.edges():
      # u → v の揮発計算
      apply_volatilization(graph, u, v, ...)
      # v → u の揮発計算
      apply_volatilization(graph, v, u, ...)
  ```

  → **双方向を明示的に処理**

- **aco_solver.py (修正前)**:

  ```python
  for u, v in graph.graph.edges():
      # (u, v)のみ処理
  ```

  → **双方向の処理が不足していた**

- **aco_solver.py (修正後)**:
  ```python
  for u, v in graph.graph.edges():
      # u → v の揮発計算
      self._apply_evaporation_to_edge(graph, u, v)
      # v → u の揮発計算
      self._apply_evaporation_to_edge(graph, v, u)
  ```
  → **双方向を明示的に処理（修正済み）**

**影響**: フェロモン揮発が正しく処理されない可能性があった（修正済み）

---

### 2. **BKB リセット（スタートノード切り替え時）**

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

### 3. **BKB 更新方法**

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

## 🟡 中程度の違い（実装の違いだが動作は同じ可能性）

### 4. **ログ形式**

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

### 5. **ゴールノードの管理方法**

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

## 🟢 同じ（確認済み）

### パラメータ値

- ✅ `ACHIEVEMENT_BONUS = 2.0` vs `bonus_factor: 2.0` → 同じ
- ✅ `V = 0.98` (残存率) vs `evaporation_rate: 0.02` (揮発率) → 同じ
- ✅ `PENALTY_FACTOR = 0.5` vs `penalty_factor: 0.5` → 同じ
- ✅ `BKB_EVAPORATION_RATE = 0.999` (残存率) vs `bkb_evaporation_rate: 0.001` (揮発率) → 同じ
- ✅ `ALPHA = 1.0` vs `alpha: 1.0` → 同じ
- ✅ `BETA = 1.0` vs `beta_bandwidth: 1.0` → 同じ
- ✅ `EPSILON = 0.1` vs `epsilon: 0.1` → 同じ
- ✅ `TTL = 100` vs `ttl: 100` → 同じ
- ✅ `ANT_NUM = 10` vs `num_ants: 10` → 同じ

### フェロモン更新タイミング

- ✅ 両方とも「アリがゴールに到達した時点で即座に更新」（完全分散方式）

### フェロモン揮発タイミング

- ✅ 両方とも「各世代の終了時に揮発」

### BKB 揮発タイミング

- ✅ 両方とも「各世代の終了時に揮発」

### 最適解の計算方法

- ✅ 両方とも`max_load_path`と`bottleneck_capacity`を使用

### 最適解の判定方法

- ✅ 両方とも`>=`で比較

### スタートノードの選択方法

- ✅ 両方とも「初期ゴールノードを除外してから 10 個選択」

### ゴールノードの初期化

- ✅ 両方とも「最初から初期ゴールノードを含める」

### 最適解の計算タイミング

- ✅ 両方とも「スタートノード切り替え時に計算」

### 最適解の取得方法

- ✅ 両方とも「辞書から取得」

---

## 📝 最終結論

**主な違いは 3 つ**:

1. **🔴 フェロモン揮発の双方向処理（修正済み）**: `aco_solver.py`で双方向を明示的に処理するように修正
2. **BKB リセット**: `aco_sim_caching_model_eval.py`はリセット、`aco_solver.py`はリセットしない（ユーザー指示により）
3. **BKB 更新方法**: `aco_sim_caching_model_eval.py`は単純 max、`aco_solver.py`はリングバッファ（ただし`bkb_window_size: 1000`で実質同じ）

**修正済み**: フェロモン揮発の双方向処理を修正しました。これが結果の違いの主因である可能性が高いです。








