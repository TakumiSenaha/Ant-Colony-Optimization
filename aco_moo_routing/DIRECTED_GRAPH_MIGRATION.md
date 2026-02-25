# 有向グラフ（DiGraph）への移行完了

## 🎯 変更内容

### 1. **無向グラフ → 有向グラフへの変更**

**変更前（無向グラフ）**:
```python
graph = nx.barabasi_albert_graph(num_nodes, num_edges)
# → G.edges[u, v]とG.edges[v, u]は同じエッジを指す
# → 双方向で異なる属性を持てない
```

**変更後（有向グラフ）**:
```python
# まず無向グラフを生成
undirected_graph = nx.barabasi_albert_graph(num_nodes, num_edges)

# 有向グラフに変換
graph = nx.DiGraph()
graph.add_nodes_from(undirected_graph.nodes())
for u, v in undirected_graph.edges():
    graph.add_edge(u, v)  # u → v
    graph.add_edge(v, u)  # v → u （異なる属性を持てる）
```

**効果**:
- ✅ `G.edges[u, v]`と`G.edges[v, u]`が**異なるエッジ**になる
- ✅ 双方向で異なる`min_pheromone`、`max_pheromone`を設定できる
- ✅ フェロモン量も双方向で独立して管理される

---

## 📊 双方向の属性が正しく設定されることを確認

### min_pheromoneの例

```
ノードA（degree=7）→ ノードB（degree=3）
  G.edges[A, B]['min_pheromone'] = 100 * 3 // 7 = 42

ノードB（degree=3）→ ノードA（degree=7）
  G.edges[B, A]['min_pheromone'] = 100 * 3 // 3 = 100
```

**無向グラフの場合**（変更前）:
- 両方とも100（後で設定した値が残る）

**有向グラフの場合**（変更後）:
- A→B: 42
- B→A: 100  ✅ 正しく異なる値を持つ

---

## 🔒 タブーリスト（ループ防止）

既に実装されています：

```python
# ant.py
def has_visited(self, node: int) -> bool:
    return node in self.route

# aco_solver.py
candidates = [n for n in neighbors if not ant.has_visited(n)]
```

**効果**:
- ✅ 訪問済みノードは候補から除外される
- ✅ 直接のループ（戻るパス）は防止される
- ✅ アリが同じノードを2回訪問することはない

---

## 🧪 テスト結果

### 全7テストが成功

```bash
tests/test_manual_environment.py .......                                 [100%]
============================== 7 passed in 0.79s ==============================
```

### テストの内容

1. ✅ **test_manual_environment_bandwidth_setup**
   - manual環境で最適経路を100Mbpsに設定
   - 全エッジの帯域が100Mbpsになることを確認

2. ✅ **test_manual_environment_pheromone_min_max_proposed_method**
   - 提案手法のフェロモンmin/max設定を確認
   - 双方向で異なるmin_pheromone値を確認
   - manual環境処理後のmax_pheromone=100^5を確認

3. ✅ **test_manual_environment_pheromone_min_max_conventional_method**
   - ConventionalACOSolverが正規化スケール（min=0.01, max=10.0）を使用
   - _reinitialize_pheromones()で全エッジが正しく初期化される

4. ✅ **test_proposed_method_pheromone_accumulation**
   - 提案手法で100Mbpsパスのフェロモンが正しく蓄積される
   - max_pheromone=100^5で切り捨てられない

5. ✅ **test_conventional_method_pheromone_normalization**
   - 従来手法が正規化スケールを維持
   - manual環境処理後も正規化スケールを維持

6. ✅ **test_proposed_method_finds_optimal_solution_in_manual_environment**
   - ACOが正常に実行される
   - 結果が生成される

7. ✅ **test_min_pheromone_bidirectional_different_values**
   - 双方向でmin_pheromoneが異なることを確認
   - 次数に基づいて正しく計算される

---

## 📝 修正されたファイル

### 1. graph.py
- 無向グラフ→有向グラフへの変換処理を追加
- 帯域・遅延の生成を有向グラフ対応に修正

### 2. conventional_aco_solver.py
- beta_bandwidthの設定ロジックを修正（2.0を明示的に設定）
- _reinitialize_pheromones()メソッドを追加

### 3. run_experiment.py
- manual環境のフェロモンmin/max再計算関数を追加
- basic_aco用にbeta_bandwidth_overrideを使用

### 4. tests/test_manual_environment.py（新規作成）
- manual環境の初期値設定テスト
- フェロモン更新テスト
- 双方向属性のテスト

---

## ✅ 期待される効果

### 1. 双方向で独立したフェロモン管理

- 往路（u→v）と復路（v→u）で異なるフェロモン量
- より細かい学習が可能

### 2. manual環境での正確な動作

- フェロモンmin/maxが正しく計算される
- 提案手法・従来手法の両方で正しく動作

### 3. タブーリストによるループ防止

- 訪問済みノードは選択されない
- 効率的な探索が保証される

---

## 🚀 次のステップ

1. **既存テストを実行**
   ```bash
   cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/aco_moo_routing
   python -m pytest tests/ -v
   ```

2. **manual環境で実験を実行**
   ```bash
   # config.yamlを編集: graph_type: "manual", method: "proposed"
   python experiments/run_experiment.py
   ```

3. **成功率を確認**
   - 提案手法: 60% → 80%に改善されるはず
   - 従来手法: 正しく動作することを確認

---

## 📖 重要な学び

### NetworkXの無向グラフ vs 有向グラフ

**無向グラフ（nx.Graph）**:
- `G.edges[u, v]`と`G.edges[v, u]`は**同じエッジ**
- 双方向で異なる属性を持てない
- 既存実装もこれを使用していた

**有向グラフ（nx.DiGraph）**:
- `G.edges[u, v]`と`G.edges[v, u]`は**異なるエッジ**
- 双方向で独立した属性を持てる
- より精密な制御が可能

### 既存実装の意図 vs 実際の動作

**意図**: 双方向で異なるmin_pheromoneを設定
```python
graph[u][v]["min_pheromone"] = MIN_F * 3 // degree_u
graph[v][u]["min_pheromone"] = MIN_F * 3 // degree_v
```

**実際の動作**（無向グラフ）:
- 両方とも同じエッジを指すため、後で設定した値（degree_vに基づく値）が残る

**新実装**（有向グラフ）:
- 意図通り、双方向で異なる値を持つ ✅

---

## 🎓 フェロモンの双方向付加について

現在の実装では、フェロモンは**双方向に付加**されます：

```python
# aco_solver.py
G.edges[u, v]["pheromone"] += pheromone_increase
G.edges[v, u]["pheromone"] += pheromone_increase
```

**理由**:
- ICNのコンテンツ要求ノード変動環境を模擬
- アリが往路（u→v）を通った時、復路（v→u）も学習される
- 双方向で同じ経験を共有

**有向グラフの利点**:
- 双方向に付加しても、各方向のフェロモン量は独立して管理される
- 揮発時に各方向で異なるBKBペナルティが適用される
- より精密な学習が可能


