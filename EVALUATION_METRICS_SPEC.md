# 評価指標の追加仕様（刷新版）

## 📋 概要

現在の評価は「最適解かどうか」の二値判定（0/-1/-2）のみですが、より詳細な評価指標を追加します。

**設計方針**: 全ての最適化タイプで**共通の CSV 形式**を使用し、シンプルで拡張しやすい構造にします。

---

## 🎯 統一された評価指標

### 基本方針

1. **見つけた解の詳細を必ず記録**: 各アリが見つけた解の`(bandwidth, delay, hops)`を記録
2. **最適解との比較を記録**: 最適化タイプに応じた比較方法で「どれだけ良いか」を数値化
3. **世代ごとの集計を記録**: 1 世代の複数のアリの結果を統計的に集計

---

## 📊 CSV ファイルの設計

### 1. アリごとの詳細ログ（全最適化タイプで共通）

**ファイル**: `ant_solution_log.csv`

**形式**: 世代 × アリ ID の形式

```
generation, ant_id, bandwidth, delay, hops, is_optimal, optimal_index, is_unique_optimal, quality_score
0, 0, 80.0, 45.2, 12, 1, 0, 1, 1.0
0, 1, 75.0, 50.0, 11, 0, -1, 0, 0.9375
0, 2, 80.0, 48.0, 13, 1, 1, 0, 1.0
0, 3, -1, -1, -1, -1, -1, -1, -1
1, 0, 79.0, 46.0, 12, 0, -1, 0, 0.9875
1, 1, 80.0, 45.2, 12, 1, 0, 1, 1.0
...
```

**列の説明**:

- `generation`: 世代番号（0 始まり）
- `ant_id`: その世代内でのアリの ID（0 始まり、通常 0~9）
- `bandwidth`: 見つけたボトルネック帯域値（Mbps）、ゴール未到達の場合は`-1`
- `delay`: 見つけた累積遅延（ms）、ゴール未到達の場合は`-1`
- `hops`: 見つけたホップ数、ゴール未到達の場合は`-1`
- `is_optimal`: 最適解かどうか（1=最適解、0=非最適解、-1=ゴール未到達）
- `optimal_index`: どの最適解に一致したか（最適解リストのインデックス、非最適解の場合は`-1`、ゴール未到達の場合は`-1`）
  - **ボトルネック帯域のみ最適化**: 常に`0`（最適解が 1 つ）または`-1`（非最適解）
  - **遅延制約付き最適化**: 最適解リストのインデックス（0, 1, 2, ...）または`-1`
  - **多目的最適化**: パレートフロンティアのインデックス（0, 1, 2, ...）または`-1`
- `is_unique_optimal`: 一意な最適解（最良の最適解）に一致したか（1=一意な最適解、0=非一意、-1=ゴール未到達）
  - **ボトルネック帯域のみ最適化**: `is_optimal`と同じ値
  - **遅延制約付き最適化**: 最適解リストの中で遅延が最小のものに一致した場合に`1`
  - **多目的最適化**: パレートフロンティアの中で最も良い解に一致した場合に`1`（実装は将来）
- `quality_score`: 最適解に対する品質スコア（後述）、ゴール未到達の場合は`-1`

**注意**: ゴール未到達の場合は`-1`を記録（既存の`ant_log.csv`と統一）

---

### 2. 品質スコア（`quality_score`）の計算方法

最適化タイプによって計算方法が異なりますが、**全て 0.0~1.0 の範囲**で統一します。

#### 2.1 ボトルネック帯域のみ最適化の場合

```
quality_score = found_bandwidth / optimal_bandwidth
```

- 値の範囲: `0.0 ~ 1.0`（`1.0`が最適解）
- 最適解が計算されていない場合: `-1`

#### 2.2 遅延制約付き最適化の場合

最適解リスト（`current_optimal_solutions`）との比較：

```
# 最適解リストのいずれかに一致するかチェック
optimal_index = find_matching_optimal_index(found_solution, current_optimal_solutions)
if optimal_index >= 0:
    is_optimal = 1
    quality_score = 1.0
    # 一意な最適解（遅延最小）に一致したかチェック
    min_delay_in_solutions = min(opt_delay for opt_bw, opt_delay, opt_hops in current_optimal_solutions)
    if abs(found_delay - min_delay_in_solutions) < 1e-6:
        is_unique_optimal = 1
    else:
        is_unique_optimal = 0
else:
    is_optimal = 0
    optimal_index = -1
    is_unique_optimal = 0
    # 最適解リストの中で最も近い解との距離を計算
    quality_score = calculate_distance_score(found_solution, optimal_solutions)
```

- 値の範囲: `0.0 ~ 1.0`（`1.0`が最適解）
- 最適解が計算されていない場合: `-1`
- `optimal_index`: 最適解リストのインデックス（0, 1, 2, ...）または`-1`
- `is_unique_optimal`: 一意な最適解（遅延最小）に一致した場合に`1`

#### 2.3 多目的最適化の場合

パレートフロンティアとの比較：

```
# パレートフロンティアのいずれかに一致するかチェック
optimal_index = find_matching_pareto_index(found_solution, pareto_frontier)
if optimal_index >= 0:
    is_optimal = 1
    quality_score = 1.0
    # 一意な最適解（最も良い解）に一致したかチェック（実装は将来）
    is_unique_optimal = 0  # 将来実装
else:
    is_optimal = 0
    optimal_index = -1
    is_unique_optimal = 0
    # パレートフロンティアとの距離を計算
    quality_score = calculate_pareto_distance_score(found_solution, pareto_frontier)
```

- 値の範囲: `0.0 ~ 1.0`（`1.0`がパレート最適解）
- パレートフロンティアが計算されていない場合: `-1`
- `optimal_index`: パレートフロンティアのインデックス（0, 1, 2, ...）または`-1`
- `is_unique_optimal`: 将来実装（現時点では`0`）

---

### 3. 世代ごとの集計ログ（全最適化タイプで共通）

**ファイル**: `generation_stats.csv`

**形式**: 世代ごとに 1 行

```
generation, num_ants_reached, avg_bandwidth, max_bandwidth, min_bandwidth, std_bandwidth, avg_delay, max_delay, min_delay, std_delay, avg_hops, max_hops, min_hops, std_hops, avg_quality_score, max_quality_score, min_quality_score, std_quality_score, optimal_count, unique_optimal_count
0, 10, 78.5, 80.0, 75.0, 2.5, 47.0, 50.0, 45.2, 1.8, 12.0, 13, 11, 0.8, 0.98125, 1.0, 0.9375, 0.03125, 2, 1
1, 10, 79.2, 80.0, 77.0, 1.2, 46.5, 48.0, 45.2, 1.0, 12.1, 13, 11, 0.7, 0.99, 1.0, 0.9625, 0.015, 3, 2
...
```

**列の説明**:

- `generation`: 世代番号
- `num_ants_reached`: その世代でゴールに到達したアリの数
- `avg_bandwidth`, `max_bandwidth`, `min_bandwidth`, `std_bandwidth`: ボトルネック帯域値の統計
- `avg_delay`, `max_delay`, `min_delay`, `std_delay`: 遅延の統計
- `avg_hops`, `max_hops`, `min_hops`, `std_hops`: ホップ数の統計
- `avg_quality_score`, `max_quality_score`, `min_quality_score`, `std_quality_score`: 品質スコアの統計
- `optimal_count`: 最適解を見つけたアリの数（`is_optimal == 1`の数）
- `unique_optimal_count`: 一意な最適解を見つけたアリの数（`is_unique_optimal == 1`の数）

**集計時の注意**:

- ゴール未到達のアリ（`bandwidth == -1`）は集計から除外
- `quality_score == -1`のものも集計から除外

---

## 🔍 実装時の考慮事項

### 1. 最適解が計算されていない場合

- 初期世代などで最適解がまだ計算されていない場合
- `quality_score = -1`を記録
- 集計時は除外する

### 2. ゴール未到達のアリ

- `bandwidth`, `delay`, `hops`, `is_optimal`, `quality_score`全てに`-1`を記録
- 集計時は除外する（到達したアリのみで集計）

### 3. スタートノード切り替え時の最適解の更新

- スタートノードが切り替わった場合、最適解が更新される
- その時点以降のアリは新しい最適解に対して`quality_score`を計算する

### 4. 既存の`ant_log.csv`との関係

- 既存の`ant_log.csv`（0/-1/-2）は維持（後方互換性のため）
- 新規のログファイルは追加で記録
- 既存の分析スクリプトへの影響を最小限に

---

## 📝 実装方針

### Step 1: アリごとの記録

1. `ACOSolver.run()`内で、各アリがゴールに到達した時点で：

   - `bandwidth = solution[0]`, `delay = solution[1]`, `hops = solution[2]`を記録
   - 最適解との比較を行い、`is_optimal`, `optimal_index`, `is_unique_optimal`, `quality_score`を計算
   - ゴール未到達の場合は全て`-1`を記録

2. 新しいリストに記録：

   - `ant_solution_log: List[Dict]`
     - 各要素は `{"generation": int, "ant_id": int, "bandwidth": float, "delay": float, "hops": int, "is_optimal": int, "optimal_index": int, "is_unique_optimal": int, "quality_score": float}`

3. `run_experiment.py`で CSV に保存
   - ファイル: `ant_solution_log.csv`
   - **全最適化タイプで共通**

### Step 2: 世代ごとの集計

1. `ACOSolver.run()`内で、各世代の終了時に：

   - その世代で到達したアリの結果を集計（`-1`は除外）
   - 各指標（bandwidth, delay, hops, quality_score）の平均、最大、最小、標準偏差を計算
   - 最適解を見つけたアリの数（`optimal_count`）と一意な最適解を見つけたアリの数（`unique_optimal_count`）も記録

2. 新しいリストに記録：

   - `generation_stats: List[Dict]`

3. `run_experiment.py`で CSV に保存
   - ファイル: `generation_stats.csv`
   - **全最適化タイプで共通**

### Step 3: 品質スコアの計算（最適化タイプ別）

#### ボトルネック帯域のみ最適化

```python
if current_optimal_bottleneck is not None:
    if abs(found_bandwidth - current_optimal_bottleneck) < 1e-6:
        is_optimal = 1
        optimal_index = 0  # 最適解が1つなので常に0
        is_unique_optimal = 1  # 最適解が1つなので常に1
        quality_score = 1.0
    else:
        is_optimal = 0
        optimal_index = -1
        is_unique_optimal = 0
        quality_score = found_bandwidth / current_optimal_bottleneck
else:
    is_optimal = -1
    optimal_index = -1
    is_unique_optimal = -1
    quality_score = -1
```

#### 遅延制約付き最適化

```python
if current_optimal_solutions:
    # 最適解リストのいずれかに一致するかチェック
    optimal_index = find_matching_optimal_index(found_solution, current_optimal_solutions)
    if optimal_index >= 0:
        is_optimal = 1
        quality_score = 1.0
        # 一意な最適解（遅延最小）に一致したかチェック
        min_delay_in_solutions = min(opt_delay for opt_bw, opt_delay, opt_hops in current_optimal_solutions)
        if abs(found_delay - min_delay_in_solutions) < 1e-6:
            is_unique_optimal = 1
        else:
            is_unique_optimal = 0
    else:
        is_optimal = 0
        optimal_index = -1
        is_unique_optimal = 0
        # 最適解リストの中で最も近い解との距離を計算
        quality_score = calculate_distance_score(found_solution, current_optimal_solutions)
else:
    is_optimal = -1
    optimal_index = -1
    is_unique_optimal = -1
    quality_score = -1
```

#### 多目的最適化

```python
if pareto_frontier:
    # パレートフロンティアのいずれかに一致するかチェック
    optimal_index = find_matching_pareto_index(found_solution, pareto_frontier)
    if optimal_index >= 0:
        is_optimal = 1
        quality_score = 1.0
        # 一意な最適解（最も良い解）に一致したかチェック（将来実装）
        is_unique_optimal = 0  # 将来実装
    else:
        is_optimal = 0
        optimal_index = -1
        is_unique_optimal = 0
        # パレートフロンティアとの距離を計算
        quality_score = calculate_pareto_distance_score(found_solution, pareto_frontier)
else:
    is_optimal = -1
    optimal_index = -1
    is_unique_optimal = -1
    quality_score = -1
```

---

## ✅ 決定事項

1. **CSV ファイルの統一**: ✅ **全最適化タイプで共通の形式**

   - `ant_solution_log.csv`: アリごとの詳細ログ
   - `generation_stats.csv`: 世代ごとの集計ログ

2. **品質スコア**: ✅ **0.0~1.0 の範囲で統一**

   - 最適化タイプによって計算方法が異なるが、範囲は統一
   - `1.0`が最適解、`0.0`が最悪、`-1`が未計算/未到達

3. **ゴール未到達の扱い**: ✅ **`-1`を採用**（既存の`ant_log.csv`と統一）

4. **既存ファイルとの関係**: ✅ **既存の`ant_log.csv`は維持**（後方互換性）

5. **最適解のインデックス記録**: ✅ **`optimal_index`を追加**

   - パレート最適解や遅延制約の場合、どの最適解に一致したかを記録

6. **一意な最適解の記録**: ✅ **`is_unique_optimal`を追加**

   - 遅延制約の場合、その中で最適な解（一意な最適解）であるかどうかを記録

7. **フォルダ構造**: ✅ **最適化タイプごとにフォルダを分ける**
   - `bandwidth_only/`, `delay_constraint/`, `pareto/`の 3 つのフォルダに分ける

---

## 🎯 期待される効果

1. **統一された形式**: 全最適化タイプで同じ CSV 形式を使うため、分析スクリプトを共通化できる
2. **拡張性**: 新しい最適化タイプを追加しても、CSV 形式は変更不要
3. **詳細な評価**: 見つけた解の詳細（bandwidth, delay, hops）を必ず記録するため、後から様々な分析が可能
4. **柔軟な分析**: 品質スコアや統計値を使って、分析時に最適な指標を選択してグラフ化可能

---

## 📝 実装時の詳細仕様

### CSV ファイルの保存場所

最適化タイプごとにフォルダを分けて保存:

- **ボトルネック帯域のみ最適化**:

  - `aco_moo_routing/results/{method}/{graph_type}/bandwidth_only/ant_solution_log.csv`
  - `aco_moo_routing/results/{method}/{graph_type}/bandwidth_only/generation_stats.csv`
  - `aco_moo_routing/results/{method}/{graph_type}/bandwidth_only/ant_log.csv`（既存）

- **遅延制約付き最適化**:

  - `aco_moo_routing/results/{method}/{graph_type}/delay_constraint/ant_solution_log.csv`
  - `aco_moo_routing/results/{method}/{graph_type}/delay_constraint/generation_stats.csv`
  - `aco_moo_routing/results/{method}/{graph_type}/delay_constraint/ant_log.csv`（既存）

- **多目的最適化**:
  - `aco_moo_routing/results/{method}/{graph_type}/pareto/ant_solution_log.csv`
  - `aco_moo_routing/results/{method}/{graph_type}/pareto/generation_stats.csv`
  - `aco_moo_routing/results/{method}/{graph_type}/pareto/ant_log.csv`（既存）

**フォルダ構造の例**:

```
results/
  ├── proposed/
  │   ├── static/
  │   │   ├── bandwidth_only/
  │   │   │   ├── ant_log.csv
  │   │   │   ├── ant_solution_log.csv
  │   │   │   └── generation_stats.csv
  │   │   ├── delay_constraint/
  │   │   │   ├── ant_log.csv
  │   │   │   ├── ant_solution_log.csv
  │   │   │   └── generation_stats.csv
  │   │   └── pareto/
  │   │       ├── ant_log.csv
  │   │       ├── ant_solution_log.csv
  │   │       └── generation_stats.csv
  │   └── node_switching/
  │       └── ...
  └── conventional/
      └── ...
```

### グラフ化のための準備

`generation_stats.csv`には複数の指標が含まれているため、分析スクリプトで以下のような選択が可能:

```python
# 例: 品質スコアの平均値で比較
df.plot(x='generation', y='avg_quality_score', ...)

# 例: 最適解を見つけたアリの数で比較
df.plot(x='generation', y='optimal_count', ...)

# 例: 一意な最適解を見つけたアリの数で比較（遅延制約の場合）
df.plot(x='generation', y='unique_optimal_count', ...)

# 例: 複数指標を同時に表示
df.plot(x='generation', y=['avg_quality_score', 'max_quality_score', 'min_quality_score'], ...)
```
