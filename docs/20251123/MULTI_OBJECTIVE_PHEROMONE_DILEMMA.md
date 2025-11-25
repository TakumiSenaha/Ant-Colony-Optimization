# 多目的最適化におけるフェロモン更新・揮発のジレンマ

## 問題の背景

多目的最適化（帯域、遅延、ホップ数の最適化）において、フェロモンの付加（功績ボーナス）と揮発（ペナルティ）の判定方法にジレンマが存在します。

## 現在の実装

### 1. 功績ボーナスの判定

```python
# aco_moo_routing/src/aco_routing/modules/evaluator.py
def check_bonus_condition(
    self,
    ant_solution: Tuple[float, float, int],
    node_memory: Tuple[float, float, float],
    delay_tolerance: float = 5.0,
) -> bool:
    b_ant, d_ant, h_ant = ant_solution
    k_j, l_j, m_j = node_memory  # BKB, BLD, BKH

    objectives = self.target_objectives

    # 帯域のみ最適化
    if objectives == ["bandwidth"]:
        return b_ant >= k_j

    # 帯域 + 遅延
    elif set(objectives) == {"bandwidth", "delay"}:
        return (b_ant >= k_j) and (
            d_ant <= l_j + delay_tolerance if l_j != float("inf") else True
        )

    # 帯域 + 遅延 + ホップ数
    elif set(objectives) == {"bandwidth", "delay", "hops"}:
        bandwidth_ok = b_ant >= k_j
        delay_ok = d_ant <= l_j + delay_tolerance if l_j != float("inf") else True
        hops_ok = h_ant <= m_j if m_j != float("inf") else True
        return bandwidth_ok and delay_ok and hops_ok
```

**現在のロジック**: すべての目的関数で「良い」場合にのみボーナス

### 2. 揮発のペナルティ判定

```python
# aco_moo_routing/src/aco_routing/modules/pheromone.py
def _evaporate_with_bkb_penalty(self, graph: RoutingGraph) -> None:
    for u, v in graph.graph.edges():
        edge_bandwidth = graph.graph.edges[u, v]["bandwidth"]
        edge_delay = graph.graph.edges[u, v]["delay"]
        edge_hops = 1

        bkb_u = graph[u].bkb
        bld_u = graph[u].bld
        bkh_u = graph[u].bkh

        # 帯域 + 遅延 + ホップ数の場合
        if set(objectives) == {"bandwidth", "delay", "hops"}:
            bandwidth_bad = edge_bandwidth < bkb_u
            delay_bad = edge_delay > bld_u + delay_tolerance
            hops_bad = edge_hops > bkh_u
            if bandwidth_bad or delay_bad or hops_bad:  # OR条件
                should_penalize = True
```

**現在のロジック**: いずれかの目的関数で「悪い」場合にペナルティ

## ジレンマ

### 問題 1: 功績ボーナスの厳しすぎる条件

**例**: 帯域 + 遅延の最適化

- アリの解: `(bandwidth=100, delay=20)`
- ノードの記憶: `(BKB=90, BLD=15)`
- 判定: `bandwidth >= 90` ✅ かつ `delay <= 15 + 5 = 20` ✅ → ボーナスあり

しかし、以下の場合:

- アリの解: `(bandwidth=120, delay=18)` ← 帯域は大幅改善、遅延はわずかに悪化
- ノードの記憶: `(BKB=100, BLD=15)`
- 判定: `bandwidth >= 100` ✅ だが `delay <= 15 + 5 = 20` ❌ → ボーナスなし

**問題**: 帯域が 20%改善したのに、遅延が 3ms 悪化しただけでボーナスがもらえない

### 問題 2: 揮発の緩すぎる条件

**例**: 帯域 + 遅延の最適化

- エッジ: `(bandwidth=80, delay=10)`
- ノードの記憶: `(BKB=100, BLD=5)`
- 判定: `bandwidth < 100` ✅ または `delay > 5 + 5 = 10` ✅ → ペナルティあり

しかし、以下の場合:

- エッジ: `(bandwidth=150, delay=12)` ← 帯域は優秀、遅延はわずかに悪い
- ノードの記憶: `(BKB=100, BLD=8)`
- 判定: `bandwidth < 100` ❌ だが `delay > 8 + 5 = 13` ❌ → ペナルティなし

**問題**: 帯域は優秀だが、遅延がわずかに悪いだけでペナルティがかからない

### 問題 3: トレードオフの扱い

多目的最適化では、目的関数間にトレードオフが存在します：

- 帯域を優先すると遅延が増える可能性
- 遅延を優先すると帯域が減る可能性
- ホップ数を減らすと帯域や遅延が悪化する可能性

現在の実装では、このトレードオフを適切に評価できていません。

## 検討すべき改善案

### 案 1: 重み付きスコアベースの判定

```python
def check_bonus_condition_weighted(
    self,
    ant_solution: Tuple[float, float, int],
    node_memory: Tuple[float, float, float],
) -> bool:
    """重み付きスコアで判定"""
    b_ant, d_ant, h_ant = ant_solution
    k_j, l_j, m_j = node_memory

    # 各目的関数の改善度を計算
    bandwidth_improvement = (b_ant - k_j) / k_j if k_j > 0 else 0
    delay_improvement = (l_j - d_ant) / l_j if l_j > 0 and l_j != float("inf") else 0
    hops_improvement = (m_j - h_ant) / m_j if m_j > 0 and m_j != float("inf") else 0

    # 重み付きスコア
    weights = {"bandwidth": 1.0, "delay": 1.0, "hops": 1.0}
    total_score = (
        weights["bandwidth"] * bandwidth_improvement
        + weights["delay"] * delay_improvement
        + weights["hops"] * hops_improvement
    )

    return total_score > 0  # 全体として改善していればボーナス
```

**メリット**: トレードオフを考慮できる
**デメリット**: 重みの設定が難しい

### 案 2: パレート支配ベースの判定

```python
def check_bonus_condition_pareto(
    self,
    ant_solution: Tuple[float, float, int],
    node_memory: Tuple[float, float, float],
) -> bool:
    """パレート支配で判定"""
    b_ant, d_ant, h_ant = ant_solution
    k_j, l_j, m_j = node_memory

    # アリの解がノードの記憶をパレート支配するか
    bandwidth_better = b_ant >= k_j
    delay_better = d_ant <= l_j if l_j != float("inf") else True
    hops_better = h_ant <= m_j if m_j != float("inf") else True

    # 少なくとも1つで優れ、他のすべてで劣らない
    at_least_one_better = bandwidth_better or delay_better or hops_better
    all_not_worse = (
        (b_ant >= k_j or not bandwidth_better)
        and (d_ant <= l_j or not delay_better)
        and (h_ant <= m_j or not hops_better)
    )

    return at_least_one_better and all_not_worse
```

**メリット**: パレート最適化の概念に沿っている
**デメリット**: 実装が複雑

### 案 3: 閾値ベースの緩和判定

```python
def check_bonus_condition_relaxed(
    self,
    ant_solution: Tuple[float, float, int],
    node_memory: Tuple[float, float, float],
    improvement_threshold: float = 0.1,  # 10%以上の改善
    degradation_tolerance: float = 0.05,  # 5%以下の悪化は許容
) -> bool:
    """改善が大きく、悪化が小さい場合にボーナス"""
    b_ant, d_ant, h_ant = ant_solution
    k_j, l_j, m_j = node_memory

    # 改善度と悪化度を計算
    bandwidth_improvement = (b_ant - k_j) / k_j if k_j > 0 else 0
    delay_degradation = (d_ant - l_j) / l_j if l_j > 0 and l_j != float("inf") else 0
    hops_degradation = (h_ant - m_j) / m_j if m_j > 0 and m_j != float("inf") else 0

    # いずれかが閾値以上改善し、他の悪化が許容範囲内
    has_significant_improvement = (
        bandwidth_improvement >= improvement_threshold
        or delay_degradation <= -improvement_threshold
        or hops_degradation <= -improvement_threshold
    )

    degradation_acceptable = (
        delay_degradation <= degradation_tolerance
        and hops_degradation <= degradation_tolerance
    )

    return has_significant_improvement and degradation_acceptable
```

**メリット**: 実装が比較的簡単
**デメリット**: 閾値の設定が難しい

### 案 4: 評価関数スコアベースの判定

```python
def check_bonus_condition_score_based(
    self,
    ant_solution: Tuple[float, float, int],
    node_memory: Tuple[float, float, float],
) -> bool:
    """評価関数のスコアで判定"""
    # アリの解のスコア
    ant_score = self.evaluator.evaluate(*ant_solution)

    # ノードの記憶値のスコア（仮想的な解として）
    memory_score = self.evaluator.evaluate(k_j, l_j, m_j)

    # スコアが改善していればボーナス
    return ant_score > memory_score
```

**メリット**: 評価関数と一貫性がある
**デメリット**: 評価関数の設計に依存

## 揮発のペナルティについても同様の検討が必要

現在の揮発判定も、同様のジレンマがあります：

- OR 条件だと、1 つでも悪ければペナルティ
- しかし、他の目的関数で優秀な場合、ペナルティをかけるべきか？

## 質問事項

1. **功績ボーナス**: すべての目的関数で良い場合のみボーナス vs 全体として改善していればボーナス
2. **揮発ペナルティ**: いずれかが悪ければペナルティ vs 全体として悪ければペナルティ
3. **トレードオフ**: どのように評価すべきか？
4. **実装の複雑さ**: どの程度の複雑さまで許容できるか？

## 参考: 既存実装との互換性

既存実装（帯域のみ最適化）との互換性も考慮する必要があります：

- 帯域のみ最適化: `bandwidth >= BKB` で判定（シンプル）
- 多目的最適化: より複雑な判定が必要

## 提案

1. **段階的な改善**: まず案 3（閾値ベース）を試す
2. **パラメータの調整**: 閾値を設定ファイルで調整可能にする
3. **実験による検証**: 異なる判定方法で実験し、結果を比較
