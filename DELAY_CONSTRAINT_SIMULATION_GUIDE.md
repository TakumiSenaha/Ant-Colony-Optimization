# 遅延制約付きシミュレーション実行ガイド

## 概要

遅延制約を追加することで、パレート最適化の前段階として、最適解を 1 つに絞り込むことができます。これにより、完全分散環境でも厳密解法で見つけた解にルーティングできている確率を検証できます。

---

## 環境設定

### config.yaml の設定

以下の設定で、静的グラフ環境で遅延制約付きのシミュレーションを実行できます：

```yaml
experiment:
  name: "delay_constraint_test"
  generations: 1000
  num_ants: 10
  simulations: 100
  target_objectives: ["bandwidth"]

  # 遅延制約を有効化
  delay_constraint:
    enabled: true
    max_delay: 50.0 # 最大遅延50ms（適切な値に調整してください）

  # スタートノード切り替えを無効化（静的グラフ）
  start_switching:
    enabled: false

graph:
  num_nodes: 100
  num_edges: 6
  graph_type: "barabasi_albert"
  bandwidth_range: [10, 100]
  delay_range: [1, 10]

  # 帯域変動を無効化（静的グラフ）
  fluctuation:
    enabled: false

aco:
  # 手法を選択（"proposed" または "conventional"）
  method: "proposed"
```

### 遅延制約の値（max_delay）の設定方法

**推奨値の決定方法：**

1. **まず制約なしで最適解を計算**

   - `delay_constraint.enabled: false`で一度実行
   - 最適解の遅延を確認

2. **適切な制約値を設定**

   - 最適解の遅延の 1.2〜1.5 倍程度を設定
   - 例：最適解の遅延が 30ms の場合 → `max_delay: 40.0` または `50.0`
   - 制約が厳しすぎると、解が存在しない可能性がある
   - 制約が緩すぎると、最適解が複数になる可能性がある

3. **グラフの特性を考慮**
   - エッジの遅延範囲: `delay_range: [1, 10]`の場合、100 ノードのグラフでは
   - 最短経路でも 10〜20 ホップ程度 → 累積遅延は 10〜200ms 程度
   - 適切な値: 30〜80ms 程度が目安

---

## シミュレーション実行コマンド

### 1. 提案手法（Proposed Method）で実行

```bash
cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization
python aco_moo_routing/experiments/run_experiment.py
```

**設定確認：**

- `config.yaml`の`aco.method`が`"proposed"`であることを確認
- `delay_constraint.enabled`が`true`であることを確認

### 2. 従来手法（Conventional Method）で実行

```bash
cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization
# config.yamlのaco.methodを"conventional"に変更してから実行
python aco_moo_routing/experiments/run_experiment.py
```

**設定変更：**

```yaml
aco:
  method: "conventional" # 従来手法に変更
```

### 3. 両手法を比較する場合

**手順：**

1. **提案手法で実行**

   ```bash
   # config.yamlで method: "proposed" を設定
   python aco_moo_routing/experiments/run_experiment.py
   ```

2. **従来手法で実行**

   ```bash
   # config.yamlで method: "conventional" に変更
   python aco_moo_routing/experiments/run_experiment.py
   ```

3. **結果の比較**
   ```bash
   python aco_moo_routing/results/analysis/compare_conventional_vs_proposed.py
   ```

---

## 実行結果の保存場所

シミュレーション結果は以下のディレクトリに保存されます：

```
aco_moo_routing/results/
├── proposed/
│   └── static/
│       └── ant_log.csv          # アリごとの成功/失敗ログ
├── conventional/
│   └── static/
│       └── ant_log.csv          # アリごとの成功/失敗ログ
```

**注意：** 同じ手法・同じ環境で実行すると、既存のディレクトリは削除され、新しい結果で上書きされます。

---

## 評価方法

### 1. 最適解選択率の確認

**CSV ログの形式：**

- `0`: 最適解を発見（遅延制約も満たしている）
- `-1`: ゴール未到達、または遅延制約違反
- `-2`: ゴール到達したが最適解ではない（通常は発生しない）

**評価スクリプト：**

```bash
python aco_moo_routing/results/analysis/compare_conventional_vs_proposed.py
```

このスクリプトは：

- 世代ごとの最適解選択率を計算
- グラフを生成（EPS 形式で保存）
- 統計情報（最終世代の選択率、平均選択率）を表示

### 2. 最適解の確認

**実行時の出力から確認：**

```
Optimal Solution (with delay constraint ≤50.0ms):
  Bandwidth=80.0 Mbps, Delay=45.2 ms, Hops=12
```

この情報から：

- 遅延制約（50ms）を満たしているか確認
- 最適解が 1 つに絞られているか確認

### 3. 制約違反の確認

**実行時の警告から確認：**

制約を満たす経路が存在しない場合：

```
⚠️  Warning: Could not calculate optimal solution with delay constraint:
No path found from X to Y satisfying delay constraint (max_delay=50.0ms)
```

この場合、`max_delay`の値を大きくする必要があります。

### 4. 統計情報の確認

**実行終了時に表示される情報：**

```
============================================================
Statistical Summary - Static Graph
============================================================

Conventional Method:
   Final generation success rate: 85.00%
   Average success rate: 78.50%

Proposed Method:
   Final generation success rate: 92.00%
   Average success rate: 85.30%
```

**評価指標：**

- **Final generation success rate**: 最終世代での最適解選択率（収束性能）
- **Average success rate**: 全世代にわたる平均選択率（安定性）

---

## 期待される結果

### 正常な動作

1. **最適解が 1 つに絞られる**

   - 遅延制約を満たす経路の中で、最大ボトルネック帯域を持つ経路が 1 つに定まる

2. **世代が進むにつれて選択率が上昇**

   - 初期世代: 低い選択率（探索段階）
   - 中期世代: 選択率が上昇（学習段階）
   - 後期世代: 高い選択率で安定（収束段階）

3. **提案手法が従来手法を上回る**
   - 提案手法の自律学習メカニズムにより、より高い選択率を達成

### 問題が発生した場合

1. **最適解が計算されない**

   - `max_delay`の値を大きくする
   - グラフの接続性を確認

2. **選択率が低いまま**

   - `max_delay`の値を確認（適切な値か）
   - 世代数を増やす（`generations: 2000`など）
   - アリの数を増やす（`num_ants: 20`など）

3. **制約違反が多い**
   - `max_delay`の値を大きくする
   - グラフの遅延分布を確認（`delay_range`の設定）

---

## 実行例

### 例 1: 提案手法で遅延制約 50ms で実行

```bash
# config.yamlを編集
# delay_constraint.enabled: true
# delay_constraint.max_delay: 50.0
# aco.method: "proposed"
# start_switching.enabled: false
# fluctuation.enabled: false

python aco_moo_routing/experiments/run_experiment.py
```

**出力例：**

```
================================================================================
Experiment: delay_constraint_test
Target Objectives: ['bandwidth']
================================================================================
Environment: static
ACO Method: proposed

Simulation 1/100
================================================================================
Start: 42, Goal: 15
  Optimal Solution (with delay constraint ≤50.0ms):
    Bandwidth=70.0 Mbps, Delay=38.5 ms, Hops=8
Running Proposed ACO (with BKB/BLD/BKH learning)...
...
```

### 例 2: 両手法を比較

```bash
# 1. 提案手法で実行
python aco_moo_routing/experiments/run_experiment.py

# 2. config.yamlを編集（method: "conventional"に変更）

# 3. 従来手法で実行
python aco_moo_routing/experiments/run_experiment.py

# 4. 比較分析
python aco_moo_routing/results/analysis/compare_conventional_vs_proposed.py
```

---

## 注意事項

1. **遅延制約の値は慎重に設定**

   - 制約が厳しすぎると解が存在しない
   - 制約が緩すぎると最適解が複数になる可能性がある

2. **静的グラフ環境でのテスト**

   - 帯域変動とスタートノード切り替えを無効化
   - これにより、遅延制約の効果を純粋に評価できる

3. **最適解の再計算**

   - 遅延制約が有効な場合、最適解は制約付きで計算される
   - 実行時の出力で最適解の遅延を確認できる

4. **ログの解釈**
   - `0`: 最適解を発見（制約も満たしている）
   - `-1`: ゴール未到達または制約違反
   - 制約違反の場合も`-1`として記録される

---

## 次のステップ

遅延制約付きのシミュレーションが正常に動作することを確認したら：

1. **パレート最適化への拡張**

   - 複数の遅延制約を設定
   - 各制約に対応する最適解を探索

2. **動的環境での検証**

   - 帯域変動環境での遅延制約
   - コンテンツ要求ノード変動環境での遅延制約

3. **制約値の最適化**
   - 異なる`max_delay`値での性能比較
   - 制約値と選択率の関係を分析








