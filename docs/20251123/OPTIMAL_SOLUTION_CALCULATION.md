# 最適解の算出方法とパレート最適解の一意性

## 📊 目次

1. [最適解の算出方法](#最適解の算出方法)
2. [パレート最適解の一意性](#パレート最適解の一意性)

---

## 最適解の算出方法

### 1. 帯域のみ最適化

**アルゴリズム**: Modified Dijkstra 法

**概要**: 経路上の最小エッジ重み（ボトルネック帯域）を最大化する経路を見つけます。

**アルゴリズムの流れ**:

1. **初期化**

   - 各ノードの最良ボトルネック値を `-∞` で初期化
   - 開始ノードのボトルネック値を `∞` に設定
   - 優先度キューに開始ノードを追加

2. **探索**

   - 優先度キューから最大ボトルネック値を持つノードを選択
   - 隣接ノードに対して、新しいボトルネック値を計算
     - `new_bottleneck = min(current_bottleneck, edge_bandwidth)`
   - より良いボトルネック値が見つかったら更新

3. **終了条件**
   - 目的地ノードに到達したら終了
   - 経路を再構築して返す

**実装**:

```python
# single_objective_solver.py
def max_load_path(graph, source, target):
    # 各ノードの最良ボトルネック値を記録
    bottleneck = {node: float("-inf") for node in graph}
    bottleneck[source] = float("inf")

    # 優先度キュー（最大ヒープをシミュレート）
    pq = []
    heappush(pq, (-bottleneck[source], source))

    while pq:
        curr_bottle_neg, u = heappop(pq)
        curr_bottle = -curr_bottle_neg

        if u == target:
            break

        # 隣接ノードを処理
        for v in graph.neighbors(u):
            w = graph[u][v].get("bandwidth", 1)
            new_bottle = min(curr_bottle, w)

            if new_bottle > bottleneck[v]:
                bottleneck[v] = new_bottle
                heappush(pq, (-new_bottle, v))

    # 経路を再構築
    return reconstruct_path(pred, source, target)
```

**厳密性**: ✅ 厳密解（必ず最適解を見つける）

---

### 2. パレート最適化（多目的最適化）

**アルゴリズム**: 多目的ラベリング法（Multi-Objective Labeling）

**概要**: 各ノードに複数のラベル（非支配解）を保持し、支配関係でフィルタリングしながら探索します。

**アルゴリズムの流れ**:

1. **初期化**

   - 各ノードのラベルリストを空で初期化
   - 開始ノードに初期ラベル `(∞, 0, 0, [source])` を追加
   - 優先度キューに初期ラベルを追加

2. **探索**

   - 優先度キューからラベルを取得（帯域が大きい順）
   - 既に訪問済みのラベルに支配されている場合はスキップ
   - 隣接ノードに対して新しいラベルを生成
     - `new_bandwidth = min(current_bandwidth, edge_bandwidth)`
     - `new_delay = current_delay + edge_delay`
     - `new_hops = current_hops + 1`

3. **支配関係のフィルタリング**

   - 新しいラベルが既存ラベルに支配される → 破棄
   - 新しいラベルが既存ラベルを支配する → 既存ラベルを削除
   - 支配されないラベルのみを保持

4. **終了条件**
   - 優先度キューが空になるまで探索
   - 目的地ノードのラベルリストがパレート最適解

**実装**:

```python
# pareto_solver.py
def find_pareto_frontier(self, source, destination):
    # 各ノードのラベルリスト
    labels = {node: [] for node in self.graph.nodes()}

    # 優先度キュー
    pq = []
    initial_label = Label(bandwidth=float("inf"), delay=0.0, hops=0, path=[source])
    heappush(pq, initial_label)

    while pq:
        current_label = heappop(pq)
        current_node = current_label.path[-1]

        # 支配されている場合はスキップ
        if current_label.is_dominated_by_any(visited_labels[current_node]):
            continue

        # 目的地に到達した場合は保持（探索は続行）
        if current_node == destination:
            continue

        # 隣接ノードへ探索
        for neighbor in self.graph.neighbors(current_node):
            # 新しいラベルを生成
            new_label = Label(
                bandwidth=min(current_label.bandwidth, edge_bandwidth),
                delay=current_label.delay + edge_delay,
                hops=current_label.hops + 1,
                path=current_label.path + [neighbor]
            )

            # 支配関係でフィルタリング
            if new_label.is_dominated_by_any(labels[neighbor]):
                continue  # 支配されている場合は破棄

            # 既存ラベルを削除（新しいラベルに支配される場合）
            labels[neighbor] = [
                label for label in labels[neighbor]
                if not new_label.dominates(label)
            ]

            labels[neighbor].append(new_label)
            heappush(pq, new_label)

    # 目的地のラベルがパレート最適解
    return labels[destination]
```

**厳密性**: ✅ 厳密解（すべてのパレート最適解を見つける）

---

## パレート最適解の一意性

### 質問: パレート最適解はどのやり方でも必ず同じ解が複数出るのか？

**答え**: ✅ **はい、厳密解法を使っている場合、必ず同じ解集合が得られます。**

### 理由

1. **数学的な定義**

   - パレート最適解は「他のどの解にも支配されない解の集合」として数学的に定義される
   - グラフと目的関数が同じなら、パレート最適解は一意に決まる

2. **厳密解法の性質**

   - 多目的ラベリング法は厳密解法（exact algorithm）
   - すべての可能な経路を探索し、支配関係でフィルタリング
   - 結果として、すべてのパレート最適解を見つける

3. **アルゴリズムの違い**
   - 異なるアルゴリズム（例: NSGA-II、MOEA/D）を使っても、**厳密解法なら同じ解集合**が得られる
   - ただし、**近似解法**（ヒューリスティック）を使うと、異なる解が得られる可能性がある

### 注意点

- **解の順序**: 解の順序は異なる可能性がある（実装依存）
- **表現方法**: 経路の表現方法（ノードの順序など）は異なる可能性がある
- **解の値**: `(bandwidth, delay, hops)` の値は同じ

### 例

同じグラフと目的関数に対して：

**アルゴリズム A（多目的ラベリング法）**:

```
Solution 1: (100, 34.55, 4)
Solution 2: (110, 26.07, 5)
Solution 3: (80, 25.02, 5)
...
```

**アルゴリズム B（別の厳密解法）**:

```
Solution 1: (110, 26.07, 5)  ← 順序が違う
Solution 2: (100, 34.55, 4)
Solution 3: (80, 25.02, 5)
...
```

→ **解の集合は同じ**（順序だけ違う）

### 実装での確認

```python
# 同じグラフと目的関数で2回実行
pareto_solver1 = ParetoSolver(graph)
pareto_solver2 = ParetoSolver(graph)

solutions1 = pareto_solver1.find_pareto_frontier(source, destination)
solutions2 = pareto_solver2.find_pareto_frontier(source, destination)

# 解の集合を比較（順序を無視）
set1 = set((s[0], s[1], s[2]) for s in solutions1)
set2 = set((s[0], s[1], s[2]) for s in solutions2)

assert set1 == set2  # ✅ 必ず同じ
```

---

## 📝 まとめ

### 最適解の算出方法

| 最適化タイプ   | アルゴリズム         | 厳密性    |
| -------------- | -------------------- | --------- |
| 帯域のみ       | Modified Dijkstra 法 | ✅ 厳密解 |
| パレート最適化 | 多目的ラベリング法   | ✅ 厳密解 |

### パレート最適解の一意性

- ✅ **厳密解法を使っている場合、必ず同じ解集合が得られる**
- ✅ **グラフと目的関数が同じなら、パレート最適解は一意に決まる**
- ⚠️ **解の順序や表現方法は異なる可能性がある**
- ⚠️ **近似解法を使うと、異なる解が得られる可能性がある**

---

## 🔍 参考実装ファイル

- `src/aco_routing/algorithms/single_objective_solver.py`: Modified Dijkstra 法
- `src/aco_routing/algorithms/pareto_solver.py`: 多目的ラベリング法
