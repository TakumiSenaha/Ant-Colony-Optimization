# 実装で使用している厳密解法

## 使用しているアルゴリズム

### アルゴリズム名

**Modified Dijkstra 法を拡張した制約付きボトルネック最大化アルゴリズム**

### 実装の詳細

#### 1. 基本構造

- **データ構造**: 優先度キュー（ヒープ）を使用
- **探索方式**: ダイクストラ法の拡張版
- **目的**: 遅延制約を満たす経路の中で、ボトルネック帯域幅を最大化

#### 2. アルゴリズムの動作

```python
# 優先度キュー: (-bottleneck, delay, node)
# 各ノードに対して記録する情報:
# - bottleneck[v]: ノードvに到達可能な最大ボトルネック帯域
# - min_delay[v]: ノードvに到達可能な最小遅延

# 更新条件:
# 1. new_bottle > bottleneck[v] (より大きなボトルネック帯域)
# 2. new_bottle == bottleneck[v] and new_delay < min_delay[v] (同じ帯域でより小さい遅延)
# 3. new_delay <= max_delay (遅延制約を満たす)
```

#### 3. 実装の特徴

1. **優先度キュー**: `(-bottleneck, delay, node)` の形式で、ボトルネック帯域が大きい順に処理
2. **制約チェック**: 各エッジ追加時に、累積遅延が `max_delay` を超えないことを確認
3. **辞書式順序**: 同じボトルネック帯域を持つ経路の中で、最小遅延のものを選択
4. **経路再構築**: 前駆ノード（`pred`）を記録し、目的地から開始ノードへ逆順に経路を再構築

#### 4. コードの主要部分

```python
# 1. 初期化
bottleneck = {node: float("-inf") for node in graph}
min_delay = {node: float("inf") for node in graph}
bottleneck[source] = float("inf")
min_delay[source] = 0.0

# 2. 優先度キューに開始ノードを追加
heappush(pq, (-bottleneck[source], min_delay[source], source))

# 3. メインループ
while pq:
    curr_bottle_neg, curr_delay, u = heappop(pq)
    curr_bottle = -curr_bottle_neg

    # 4. 隣接ノードを処理
    for v in graph.neighbors(u):
        new_bottle = min(curr_bottle, bw)
        new_delay = curr_delay + delay

        # 5. 遅延制約チェック
        if new_delay > max_delay:
            continue

        # 6. 更新条件
        if new_bottle > bottleneck[v] or (
            new_bottle == bottleneck[v] and new_delay < min_delay[v]
        ):
            bottleneck[v] = new_bottle
            min_delay[v] = new_delay
            pred[v] = u
            heappush(pq, (-new_bottle, new_delay, v))
```

---

## このアルゴリズムの分類

### 1. 問題の分類

- **問題名**: Constrained Maximum Capacity Path Problem（制約付き最大容量経路問題）
- **制約**: 遅延制約（Delay Constraint）
- **目的**: ボトルネック帯域幅の最大化（Bottleneck Bandwidth Maximization）

### 2. アルゴリズムの分類

- **基本アルゴリズム**: Modified Dijkstra 法
- **拡張**: 制約付き最適化（Constrained Optimization）
- **最適化手法**: 優先度キュー（Priority Queue）ベースの貪欲法

### 3. 関連する研究分野

- **Constrained Shortest Path Problem (CSPP)**: 制約付き最短経路問題
- **QoS Routing**: Quality of Service ルーティング
- **Maximum Capacity Path Problem**: 最大容量経路問題

---

## 引用すべき参考文献

### 推奨される引用

このアルゴリズムは、以下の研究分野に関連しています：

1. **Wang & Crowcroft (1996)**

   - "Quality-of-service routing for supporting multimedia applications"
   - _IEEE Journal on Selected Areas in Communications_, 14(7), 1228-1234
   - **理由**: 帯域幅と遅延の両方を考慮した QoS ルーティングの初期研究

2. **Gabow (1985)**

   - "Scaling algorithms for network problems"
   - _Journal of Computer and System Sciences_, 31(2), 148-168
   - **理由**: 最大容量経路問題のアルゴリズム（ボトルネック最大化）

3. **Handler & Zang (1980)**
   - "A dual algorithm for the constrained shortest path problem"
   - _Networks_, 10(4), 293-310
   - **理由**: 制約付き最短経路問題のアルゴリズム

### 論文での記述例

> "The optimal solution is computed using a modified Dijkstra algorithm that extends the standard shortest path algorithm to maximize the bottleneck bandwidth while satisfying the delay constraint [Wang1996]. The algorithm maintains, for each node, the maximum bottleneck bandwidth and minimum delay achievable from the source, and explores paths in order of decreasing bottleneck bandwidth using a priority queue. When multiple paths achieve the same maximum bottleneck bandwidth, the one with the minimum delay is selected (lexicographical ordering)."

---

## まとめ

### 使用しているアルゴリズム

- **名称**: Modified Dijkstra 法を拡張した制約付きボトルネック最大化アルゴリズム
- **データ構造**: 優先度キュー（ヒープ）
- **探索方式**: 貪欲法（Greedy Algorithm）
- **制約処理**: 遅延制約を満たす経路のみを探索対象とする

### 引用すべき参考文献

- **Wang & Crowcroft (1996)**: QoS ルーティング（帯域幅と遅延を考慮）
- **Gabow (1985)**: 最大容量経路問題のアルゴリズム
- **Handler & Zang (1980)**: 制約付き最短経路問題のアルゴリズム








