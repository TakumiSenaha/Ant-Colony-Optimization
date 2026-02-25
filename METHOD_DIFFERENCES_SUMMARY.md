# 提案手法と従来手法の違い（まとめ）

## 手法の違いは 2 点のみ

1. **フェロモンの付加の仕方**
2. **フェロモンの付加タイミング・学習タイミング**（制約条件を超えたらフェロモンを付加しない）

---

## 1. フェロモンの付加の仕方

### 1.1 提案手法（Proposed Method）

#### 帯域のみ最適化（遅延制約なし）

$$
\Delta\tau_{ij} = \begin{cases}
10 \cdot B \cdot B_a, & \text{if } B \geq K_j \quad \text{(功績ボーナス)} \\
10 \cdot B, & \text{if } B < K_j \quad \text{(通常)}
\end{cases}
$$

#### 遅延制約が有効な場合

$$
\Delta\tau_{ij} = \begin{cases}
0, & \text{if } D_{\text{path}} > D_{\text{limit}} \quad \text{(制約違反経路には付加しない)} \\
10 \cdot \frac{B}{D_{\text{path}}} \cdot B_a, & \text{if } B \geq K_j \quad \text{(功績ボーナス)} \\
10 \cdot \frac{B}{D_{\text{path}}}, & \text{if } B < K_j \quad \text{(通常)}
\end{cases}
$$

**実装**:

```python
# 遅延制約が有効な場合
if delay_constraint_enabled:
    if delay > 0:
        base_pheromone = 10.0 * (bandwidth / delay)  # 10 * (B / D_path)
    else:
        base_pheromone = bandwidth * 10.0
else:
    base_pheromone = bandwidth * 10.0  # 10 * B

# 功績ボーナス
if bandwidth >= k_v:
    delta_pheromone = base_pheromone * self.bonus_factor  # B_a = 2.0
else:
    delta_pheromone = base_pheromone
```

**特徴**:

- 基本フェロモン量: `10 * B`（帯域のみ）または `10 * (B / D_path)`（遅延制約あり）
- 功績ボーナス: `B >= K_j` の場合、`B_a = 2.0` 倍
- **ノード学習（BKB）を使用**: 各ノードが過去の最良解を記憶し、それと比較してボーナスを判定

### 1.2 従来手法（Conventional Method）

#### 帯域のみ最適化（遅延制約なし）

$$
\Delta\tau_{ij} = Q \cdot B
$$

#### 遅延制約が有効な場合

$$
\Delta\tau_{ij} = \begin{cases}
0, & \text{if } D_{\text{path}} > D_{\text{limit}} \quad \text{(制約違反経路には付加しない)} \\
Q \cdot \frac{B}{D_{\text{path}}}, & \text{otherwise}
\end{cases}
$$

**実装**:

```python
# 評価関数を使用してスコアを計算
# 帯域のみ最適化: score = bandwidth
# 遅延制約が有効な場合: score = bandwidth / delay
score = self.evaluator.evaluate(bandwidth, delay, hops)

# フェロモン付加量 = Q * score
delta_pheromone = self.q_factor * score  # Q = 1.0
```

**特徴**:

- フェロモン付加量: `Q * B`（帯域のみ）または `Q * (B / D_path)`（遅延制約あり）
- 定数係数: `Q = 1.0`（`q_factor`）
- **ノード学習を使用しない**: 単純にボトルネック帯域（または帯域/遅延スコア）に比例
- **功績ボーナスなし**: 常に同じ計算式

---

## 2. フェロモンの付加タイミング・学習タイミング

### 2.1 共通点

**両手法とも、制約条件を超えたらフェロモンを付加しない**:

```python
# ゴール到達時に制約を確認
if delay_constraint_enabled:
    if solution_delay > self.max_delay:
        # 制約違反の場合は探索失敗として扱う
        # フェロモンを付加しない
        continue
```

### 2.2 提案手法の特徴

#### オンライン更新（完全分散方式）

- **タイミング**: アリがゴールに到達した**即座に**フェロモンを更新
- **ノード学習**: 同時に各ノードの BKB（Best Known Bandwidth）を更新
- **効果**: 良い経路の発見が即座に他のアリの探索に影響を与える

```python
# ゴール到達時の処理
if has_reached_any_goal:
    # 制約チェック
    if solution_delay > self.max_delay:
        continue  # フェロモンを付加しない

    # 即座にフェロモン更新（ノード学習も同時に実行）
    self.pheromone_updater.update_from_ant(ant, self.graph)
```

#### ノード学習の更新

- **タイミング**: フェロモン更新と同時
- **条件**: 制約を満たす経路からの情報のみを学習
- **更新式**: `K_v ← max(K_v, B)`（帯域のみ）

### 2.3 従来手法の特徴

#### オンライン更新（完全分散方式）

- **タイミング**: アリがゴールに到達した**即座に**フェロモンを更新
- **ノード学習**: **行わない**
- **効果**: 良い経路の発見が即座に他のアリの探索に影響を与える

```python
# ゴール到達時の処理
if has_reached_any_goal:
    # 制約チェック
    if solution_delay > self.max_delay:
        continue  # フェロモンを付加しない

    # 即座にフェロモン更新（ノード学習は行わない）
    self.pheromone_updater.update_from_ant(ant, self.graph)
```

---

## 3. まとめ

### 3.1 フェロモンの付加の仕方の違い

| 項目                       | 提案手法                                 | 従来手法                        |
| -------------------------- | ---------------------------------------- | ------------------------------- |
| **基本式（帯域のみ）**     | `10 * B`                                 | `Q * B`（`Q = 1.0`）            |
| **基本式（遅延制約あり）** | `10 * (B / D_path)`                      | `Q * (B / D_path)`（`Q = 1.0`） |
| **功績ボーナス**           | あり（`B >= K_j` の場合 `B_a = 2.0` 倍） | なし                            |
| **ノード学習**             | 使用（BKB と比較してボーナス判定）       | 使用しない                      |

### 3.2 フェロモンの付加タイミング・学習タイミングの違い

| 項目                 | 提案手法                 | 従来手法                 |
| -------------------- | ------------------------ | ------------------------ |
| **更新タイミング**   | オンライン（即座に更新） | オンライン（即座に更新） |
| **制約違反時の処理** | フェロモンを付加しない   | フェロモンを付加しない   |
| **ノード学習**       | あり（BKB 更新）         | なし                     |

### 3.3 重要な点

1. **両手法とも、制約条件を超えたらフェロモンを付加しない**
2. **両手法とも、オンライン更新（即座に更新）を使用**
3. **主な違いは、フェロモン付加量の計算式とノード学習の有無**

---

## 4. 実装の確認

### 4.1 提案手法の実装

```python
# pheromone.py (PheromoneUpdater)
if delay_constraint_enabled:
    if delay > 0:
        base_pheromone = 10.0 * (bandwidth / delay)  # 10 * (B / D_path)
    else:
        base_pheromone = bandwidth * 10.0
else:
    base_pheromone = bandwidth * 10.0  # 10 * B

# 功績ボーナス
if bandwidth >= k_v:
    delta_pheromone = base_pheromone * self.bonus_factor  # B_a = 2.0
else:
    delta_pheromone = base_pheromone
```

### 4.2 従来手法の実装

```python
# pheromone.py (SimplePheromoneUpdater)
score = self.evaluator.evaluate(bandwidth, delay, hops)
# 遅延制約が有効な場合: score = bandwidth / delay

delta_pheromone = self.q_factor * score  # Q * (B / D_path)
```








