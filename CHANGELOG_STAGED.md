# 変更点まとめ（ステージング済み）

**日付**: 2026-01-11
**変更ファイル数**: 7ファイル
**追加行数**: +955行
**削除行数**: -249行

---

## 概要

このコミットでは、以下の主要な変更を行いました：

1. **既存実装との互換性確保** - `aco_main_bkb_available_bandwidth.py`と同じ結果を得られるよう調整
2. **有向グラフへの移行** - 双方向エッジで異なる属性（フェロモン値など）を管理可能に
3. **Global Best更新戦略の追加** - 環境変化への適応性を向上
4. **評価指標の拡充** - 新しい評価指標の追加とサマリー表示の改善

---

## ファイル別の変更内容

### 1. `aco_moo_routing/experiments/run_experiment.py`

**追加行数**: +256行

#### 主な変更

- **manual環境用フェロモン再計算機能**
  ```python
  def recalculate_pheromone_min_max_for_manual_environment(
      graph, optimal_path, aco_solver, aco_method, config
  )
  ```
  - ACOSolver初期化後に最適経路のフェロモンmin/maxを再計算
  - ConventionalACOSolverが`_reinitialize_pheromones()`で全エッジを上書きするため必要

- **評価指標の計算改善**
  - `match_optimal_local()`: 最適解判定関数（遅延制約対応）
  - `quality_score_for_local()`: 品質スコア計算（帯域比）

- **新しい評価指標**
  - **Best Solution Optimal Rate**: 各世代の最良解が最適解だった確率
  - **Interest Optimal Rate**: フェロモン貪欲選択が最適解に到達した率
  - **Best Solution Quality Score**: 最良解の品質スコア（平均/最大/最小）

- **出力フォーマットの改善**
  - セクション区切りの明確化（`---- Metrics ----`等）
  - シミュレーションサマリーの追加

---

### 2. `aco_moo_routing/src/aco_routing/algorithms/aco_solver.py`

**変更行数**: +427行/-大幅修正

#### 主な変更

- **既存実装互換モードの追加**
  ```python
  # 既存実装互換のパラメータ
  self.penalty_factor = config["aco"]["learning"]["penalty_factor"]
  self.bkb_evaporation_rate = 0.999  # 残存率（既存実装と同じ）
  self.pheromone_retention_rate = 1.0 - evaporation_rate  # 0.98
  self.bonus_factor = config["aco"]["learning"]["bonus_factor"]
  ```

- **新規メソッド**
  - `_initialize_node_bkb()`: グラフのノードにBKB属性を初期化
  - `_update_pheromone_compatible()`: 既存実装互換のフェロモン更新
    - 遅延制約拡張版：`Δτ = C × B/D_path`（遅延制約有効時）
  - `_evaporate_pheromone_compatible()`: 既存実装互換のフェロモン揮発
  - `_evaporate_bkb_compatible()`: 既存実装互換のBKB揮発

- **最適解処理の修正**
  - generation 0で渡された最適解を正しく使用するよう修正
  - ループ外で`current_optimal_*`変数を初期化し、値を保持

- **デバッグ出力の追加**
  - 各世代の開始時に`current_optimal_bottleneck`を確認

---

### 3. `aco_moo_routing/src/aco_routing/algorithms/conventional_aco_solver.py`

**変更行数**: +234行/-大幅修正

#### 主な変更

- **Global Best更新戦略の追加**

  | 戦略 | 説明 | 特徴 |
  |------|------|------|
  | `ttl` | TTLモード | 一定世代経過後にGlobal Bestを無効化。階段状の性能回復 |
  | `window` | スライディングウィンドウ（推奨） | 直近N世代の履歴から最良を選択。滑らかな適応 |

  ```python
  self.gb_strategy = config["aco"].get("global_best_update_strategy", "window")
  self.gb_retention = config["aco"].get("global_best_retention", 100)
  self.best_history: deque = deque(maxlen=self.gb_retention)
  ```

- **beta_bandwidthのオーバーライド機能**
  ```python
  beta_override = config["aco"].get("beta_bandwidth_override")
  if beta_override is not None:
      self.beta_bandwidth = beta_override  # basic_aco用
  else:
      self.beta_bandwidth = 2.0  # ACS論文準拠
  ```

- **ドキュメントの充実**
  - Global Best更新戦略の詳細な説明を追加
  - ACS論文との対応関係を明記

---

### 4. `aco_moo_routing/src/aco_routing/core/graph.py`

**変更行数**: +80行/-大幅修正

#### 主な変更

- **無向グラフから有向グラフへの変換**
  ```python
  # 無向グラフを有向グラフに変換（双方向で異なる属性を持たせるため）
  graph = nx.DiGraph()
  graph.add_nodes_from(undirected_graph.nodes())
  for u, v in undirected_graph.edges():
      graph.add_edge(u, v)  # u → v
      graph.add_edge(v, u)  # v → u （異なる属性を持てる）
  ```

- **既存実装との互換性確保**
  - 帯域幅の乱数消費順序を維持
  - 各無向エッジに対して1つの帯域値を生成し、双方向で共有
  - 遅延は帯域幅の乱数消費後に別のシードで生成

- **weight属性の追加**
  ```python
  graph.edges[u, v]["weight"] = float(bandwidth)  # 既存実装互換
  ```

---

### 5. `aco_moo_routing/src/aco_routing/core/node.py`

**変更行数**: +18行/-5行

#### 主な変更

- **BKBの型互換性**
  ```python
  # 既存実装: graph.nodes[node]["best_known_bottleneck"] = int(time_window_max)
  self.bkb = float(int(time_window_max))
  ```

- **evaporate()メソッドの修正**
  - 揮発率から残存率への変換を明示化
  ```python
  # 既存実装: evaporation_rate = 0.999（残存率）
  # 新実装: evaporation_rate = 0.001（揮発率）→ 残存率 = 0.999
  retention_rate = 1.0 - evaporation_rate
  self.bkb = self.bkb * retention_rate
  ```

---

### 6. `aco_moo_routing/src/aco_routing/modules/pheromone.py`

**変更行数**: +184行/-大幅修正

#### 主な変更

- **既存実装との互換性確保**
  - 帯域幅を整数として扱う
  ```python
  bandwidth_int = int(bandwidth)  # 帯域は10Mbps刻みなので整数として扱う
  ```
  - 功績ボーナス判定を整数比較
  ```python
  if bandwidth_int >= int(k_v):
  ```

- **有向グラフ対応**
  ```python
  # G.edges()には(u, v)と(v, u)の両方が含まれるため、
  # 各エッジは1回だけ処理する（双方向処理は不要）
  for u, v in graph.graph.edges():
      self._apply_evaporation_to_edge(graph, u, v)
  ```

- **ペナルティ判定の修正**
  ```python
  # 既存実装: if weight_uv < bkb_u:
  if int(edge_bandwidth) < int(bkb_u):
      retention_rate *= self.penalty_factor  # 0.98 * 0.5 = 0.49
  ```

- **SimplePheromoneUpdater/Evaporatorのドキュメント改善**
  - 使用される手法を明記（提案手法、先行研究手法）
  - ACS論文準拠実装との違いを説明

---

### 7. `aco_moo_routing/src/aco_routing/utils/visualization.py`

**変更行数**: +5行/-5行

#### 主な変更

- **コードフォーマットの調整**
  - 条件式の改行を修正（PEP8準拠）

---

## 技術的な背景

### 有向グラフへの移行理由

無向グラフ（`nx.Graph`）では、エッジ`(u, v)`と`(v, u)`は同一として扱われます。
しかし、ACOアルゴリズムでは方向によってフェロモン値やペナルティが異なる場合があります。

**例**:
- ノードuのBKB = 80Mbps
- ノードvのBKB = 50Mbps
- エッジ(u, v)の帯域 = 60Mbps

この場合：
- エッジ(u → v): 60 < 80 なのでペナルティあり
- エッジ(v → u): 60 > 50 なのでペナルティなし

有向グラフにすることで、このような非対称な処理が正しく動作します。

### Global Best更新戦略

従来のACSは静的環境を想定していますが、帯域変動のある動的環境では
古いGlobal Bestに固執してしまう問題があります。

**TTLモード**: 一定期間後にリセット → 階段状の性能回復
**Windowモード**: 直近N世代の最良を使用 → 滑らかな適応

---

## 互換性への影響

- 既存の設定ファイル（config.yaml）は引き続き使用可能
- 新しい設定項目:
  - `aco.global_best_update_strategy`: "ttl" または "window"
  - `aco.global_best_retention`: 保持期間/ウィンドウサイズ
  - `aco.beta_bandwidth_override`: beta値のオーバーライド（basic_aco用）
