# ACO 性能改善の提案

## 🎯 現在の問題点

1. **BKB 揮発率が逆だった**: 0.001 → 0.999 に修正（完了）
2. **ハードコードされた定数**: config.yaml から取得するように修正（完了）

## 📊 追加の改善施策

### 1. 適応的な探索戦略

#### 現状

```python
epsilon = 0.1  # 固定
```

#### 提案: 世代に応じた ε-Greedy の調整

```python
# 初期は探索（高いε）、後期は活用（低いε）
epsilon_start = 0.3
epsilon_end = 0.05
epsilon = epsilon_start + (epsilon_end - epsilon_start) * (generation / generations)
```

**理由**: 初期は新しい経路を積極的に探索し、後期は見つけた良い経路を活用する方が効率的

---

### 2. 動的なボーナス係数

#### 現状

```python
bonus_factor = 2.0  # 固定
```

#### 提案: BKB 更新の重要度に応じた動的ボーナス

```python
# BKBの更新幅が大きいほど高いボーナス
improvement_ratio = bottleneck / max(old_bkb, 1.0)
if improvement_ratio >= 1.5:  # 50%以上の改善
    bonus_factor = 3.0
elif improvement_ratio >= 1.2:  # 20%以上の改善
    bonus_factor = 2.0
else:
    bonus_factor = 1.5
```

**理由**: 大幅な改善を発見した場合、より強く報酬を与えることで学習を加速

---

### 3. リングバッファサイズの最適化

#### 現状

```python
bkb_window_size = 100  # 固定
```

#### 提案: 環境の変動速度に応じた適応的なバッファサイズ

```python
# 帯域変動が速い環境: 小さいバッファ（直近の情報を重視）
# 帯域変動が遅い環境: 大きいバッファ（長期的な傾向を重視）
if update_interval <= 5:  # 高頻度変動
    bkb_window_size = 50
elif update_interval <= 20:  # 中頻度変動
    bkb_window_size = 100
else:  # 低頻度変動
    bkb_window_size = 200
```

**理由**: 環境の変動速度に合わせて記憶期間を調整することで、より適切な学習が可能

---

### 4. フェロモン付加量の正規化

#### 現状

```python
base_pheromone = float(bottleneck_int) * 10.0
```

#### 提案: ネットワーク規模に応じた正規化

```python
# グラフの平均帯域で正規化
avg_bandwidth = np.mean([d['bandwidth'] for u, v, d in graph.edges(data=True)])
normalized_bottleneck = bottleneck / max(avg_bandwidth, 1.0)
base_pheromone = normalized_bottleneck * 1000.0  # スケーリング
```

**理由**: ネットワーク規模や帯域範囲が異なる場合でも、安定した学習が可能

---

### 5. 多様性の維持

#### 現状

- 全てのアリが同じパラメータで探索

#### 提案: ヘテロジニアスアリの導入

```python
# アリごとに異なる探索戦略
for i in range(num_ants):
    if i < num_ants // 3:
        # 帯域重視のアリ
        beta_bandwidth, beta_delay = 2.0, 0.5
    elif i < 2 * num_ants // 3:
        # バランス型のアリ
        beta_bandwidth, beta_delay = 1.0, 1.0
    else:
        # 遅延重視のアリ
        beta_bandwidth, beta_delay = 0.5, 2.0
```

**理由**: 異なる戦略のアリを混在させることで、パレートフロンティア全体を効率的に探索

---

### 6. フェロモン揮発率の適応的調整

#### 現状

```python
evaporation_rate = 0.02  # 固定（残存率0.98）
```

#### 提案: 収束度に応じた揮発率の調整

```python
# 収束が進んだら揮発を抑制（良い経路を保持）
# 収束が停滞したら揮発を促進（探索を促進）
recent_improvement = calculate_recent_improvement(results)
if recent_improvement < 0.01:  # 停滞
    evaporation_rate = 0.05  # 揮発を促進
else:  # 改善中
    evaporation_rate = 0.02  # 標準
```

**理由**: 探索と活用のバランスを動的に調整することで、局所最適解からの脱出を促進

---

### 7. 初期フェロモンの最適化

#### 現状

```python
initial_pheromone = "auto"  # 最近傍ヒューリスティック解から計算
```

#### 提案: 複数の初期解から計算

```python
# 最大帯域経路、最小遅延経路、最小ホップ経路の3つから計算
bandwidth_path = max_load_path(graph, start, goal)
delay_path = dijkstra_delay(graph, start, goal)
hop_path = dijkstra_hops(graph, start, goal)

# 3つの経路の平均的なフェロモン値を初期値とする
initial_pheromone = (
    calculate_path_quality(bandwidth_path) +
    calculate_path_quality(delay_path) +
    calculate_path_quality(hop_path)
) / 3.0
```

**理由**: 複数の視点から良い経路を考慮することで、より良い初期状態を設定

---

## 🧪 実験的な施策

### 8. BKB 揮発の非線形化

```python
# 現在: 線形的な揮発 (bkb *= 0.999)
# 提案: 非線形的な揮発（古い情報ほど速く忘却）

# BKBの年齢を追跡
if bkb_age > 100:  # 100世代以上前の情報
    evaporation_rate = 0.95  # 速く忘却
elif bkb_age > 50:
    evaporation_rate = 0.98
else:
    evaporation_rate = 0.999  # ゆっくり忘却
```

### 9. エリート保存戦略

```python
# 各世代の最良解をエリートとして保存
elite_solutions = []
for generation in range(generations):
    # 通常のACO探索
    ...
    # 最良解を保存
    if generation_best_solution not in elite_solutions:
        elite_solutions.append(generation_best_solution)

    # エリート解の経路にボーナスフェロモン
    for solution in elite_solutions[-10:]:  # 最近のエリート10個
        update_pheromone_for_elite(solution)
```

---

## 📋 優先順位

1. **高優先度**（即座に実装可能で効果が大きい）

   - ✅ BKB 揮発率の修正（完了）
   - ✅ ハードコード定数の削除（完了）
   - 🔲 適応的な ε-Greedy（施策 1）
   - 🔲 リングバッファサイズの最適化（施策 3）

2. **中優先度**（実装コストは中程度、効果も中程度）

   - 🔲 動的なボーナス係数（施策 2）
   - 🔲 フェロモン揮発率の適応的調整（施策 6）
   - 🔲 ヘテロジニアスアリ（施策 5）

3. **低優先度**（実験的、効果が未知）
   - 🔲 BKB 揮発の非線形化（施策 8）
   - 🔲 エリート保存戦略（施策 9）

---

## 🎓 理論的根拠

1. **Exploration-Exploitation Trade-off**: ACO の根本的な課題は、探索と活用のバランス。適応的な ε や揮発率はこの問題に対処
2. **Dynamic Optimization**: 動的環境では、古い情報を速く忘却し、新しい情報に適応することが重要
3. **Diversity Maintenance**: パレート最適化では、多様性を維持することで、フロンティア全体をカバー
4. **No Free Lunch Theorem**: 単一の手法が全ての問題で最良ということはない。環境に応じてパラメータを調整することが重要

---

## 📊 評価方法

各施策の効果を評価するため、以下の指標を測定：

1. **最適解到達率**: アリが最適解に到達した割合
2. **収束速度**: 最適解に到達するまでの世代数
3. **安定性**: 収束後の成功率の標準偏差
4. **ロバスト性**: 帯域変動に対する性能の変化

```python
# 評価コード例
def evaluate_improvement(baseline_results, improved_results):
    metrics = {
        'success_rate': np.mean(improved_results['optimal_count']) / np.mean(baseline_results['optimal_count']),
        'convergence_speed': np.mean(baseline_results['convergence_gen']) / np.mean(improved_results['convergence_gen']),
        'stability': np.std(baseline_results['success_rate']) / np.std(improved_results['success_rate']),
    }
    return metrics
```

