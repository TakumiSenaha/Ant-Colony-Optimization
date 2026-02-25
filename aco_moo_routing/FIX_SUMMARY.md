# ACO Solver 修正サマリー

## 🔧 修正内容

### 1. BKB 揮発率の修正（最重要！）

**問題**:  
config.yaml の`bkb_evaporation_rate: 0.001`を残存率として使用していたため、BKB がほぼ消滅していた。

**原因**:

- 既存実装: `BKB_EVAPORATION_RATE = 0.999`（残存率） → `bkb *= 0.999`（99.9%残存）
- 新実装（修正前）: `bkb_evaporation_rate: 0.001`（config.yaml） → `bkb *= 0.001`（0.1%残存）
- **結果**: BKB 学習がほぼ機能せず、フェロモン更新が適切に行われなかった

**修正**:

```python
# aco_solver.py Line 109
self.bkb_evaporation_rate = 0.999  # 既存実装と同じ残存率を直接使用
```

---

### 2. ハードコードされた定数の削除

**問題**:  
`COMPATIBLE_V`と`COMPATIBLE_BONUS_FACTOR`がハードコードされており、config.yaml と重複していた。

**修正**:

```python
# 修正前（aco_solver.py Line 40-43）
COMPATIBLE_V = 0.98
COMPATIBLE_BONUS_FACTOR = 2.0

# 修正後（aco_solver.py Line 108-120）
# フェロモン揮発の残存率（既存実装: V = 0.98）
evaporation_rate = config["aco"]["evaporation_rate"]
self.pheromone_retention_rate = 1.0 - evaporation_rate  # 0.98

# 功績ボーナス係数（既存実装: ACHIEVEMENT_BONUS = 2.0）
self.bonus_factor = config["aco"]["learning"]["bonus_factor"]
```

---

## 📊 期待される効果

### 修正による改善

1. **BKB 学習の正常化**

   - BKB が適切に保持されるようになり、ノードの学習が機能する
   - フェロモン付加時の功績ボーナスが正しく適用される

2. **設定の一元化**

   - config.yaml から全てのパラメータを取得
   - ハードコードされた値を削除し、メンテナンス性が向上

3. **既存実装との完全互換性**
   - 既存実装（aco_main_bkb_available_bandwidth.py）と同じ結果が得られる

---

## 🧪 検証方法

### 1. manual 環境でのテスト

```bash
cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/aco_moo_routing
python experiments/run_experiment.py
```

**確認項目**:

- ✅ 最適解到達率が既存実装と同等かどうか
- ✅ BKB 値が適切に更新されているか（デバッグ出力で確認）
- ✅ 収束速度が改善されているか

### 2. ログの比較

```bash
# 既存実装
cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization
python src/aco_main_bkb_available_bandwidth.py

# 新実装
cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/aco_moo_routing
python experiments/run_experiment.py
```

**比較項目**:

- 最適解到達率（成功率）
- 収束速度（最適解に到達するまでの世代数）
- BKB 値の推移
- フェロモン値の推移

---

## 📝 config.yaml 設定例

### manual 環境（Environment 1）

```yaml
experiment:
  name: "manual_environment_test"
  generations: 1000
  num_ants: 10
  simulations: 100
  target_objectives: ["bandwidth"]

graph:
  num_nodes: 100
  num_edges: 6
  graph_type: "manual" # 最適経路を100Mbpsに設定
  bandwidth_range: [10, 100]
  delay_range: [1, 10]
  fluctuation:
    enabled: false # 変動なし（静的環境）

aco:
  method: "proposed" # 提案手法
  alpha: 1.0
  beta_bandwidth: 1.0
  epsilon: 0.1
  evaporation_rate: 0.02 # 残存率0.98
  learning:
    bkb_window_size: 100
    bonus_factor: 2.0
    penalty_factor: 0.5
```

---

## 🎯 今後の改善案

詳細は`IMPROVEMENT_PROPOSALS.md`を参照。

### 高優先度

1. ✅ BKB 揮発率の修正（完了）
2. ✅ ハードコード定数の削除（完了）
3. 🔲 適応的な ε-Greedy（提案のみ、未実装）
4. 🔲 リングバッファサイズの最適化（提案のみ、未実装）

### 中優先度

5. 🔲 動的なボーナス係数（改善幅に応じた調整）
6. 🔲 フェロモン揮発率の適応的調整（収束度に応じた調整）
7. 🔲 ヘテロジニアスアリ（多様性の維持）

---

## 📖 参考資料

- **既存実装**: `/Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/src/aco_main_bkb_available_bandwidth.py`
- **新実装**: `/Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/aco_moo_routing/src/aco_routing/algorithms/aco_solver.py`
- **BKB 学習**: `/Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/src/bkb_learning.py`
- **フェロモン更新**: `/Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/src/pheromone_update.py`

---

## 🔍 デバッグのヒント

### BKB 値の確認

```python
# aco_solver.py の _evaporate_bkb_compatible メソッドにデバッグ出力を追加
def _evaporate_bkb_compatible(self) -> None:
    G = self.graph.graph
    for node in G.nodes():
        old_bkb = G.nodes[node].get("best_known_bottleneck", 0)
        new_bkb = old_bkb * self.bkb_evaporation_rate
        G.nodes[node]["best_known_bottleneck"] = new_bkb

        # デバッグ出力（最初の数ノードのみ）
        if node < 3:
            print(f"Node {node}: BKB {old_bkb:.2f} → {new_bkb:.2f}")
```

### 最適解到達の確認

```python
# aco_solver.py の run メソッドでデバッグ出力を追加
if generation < 3 and ant.ant_id == 0:
    route_bottleneck = min(ant.bandwidth_log) if ant.bandwidth_log else 0
    if route_bottleneck >= 100.0:
        print(f"[DEBUG] Gen {generation}, Ant {ant.ant_id}: "
              f"100Mbpsパスを発見！ route={ant.route}")
```
