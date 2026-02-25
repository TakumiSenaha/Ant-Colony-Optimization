# Manual環境での問題修正サマリー（最終版）

## 🎯 問題点

manual環境では、最適経路を100Mbpsに設定した後、**フェロモンのmin/maxが再計算されていない**ため、提案手法での成功率が60%に留まっていました（barabasi_albert環境では80%）。

### 原因

1. **フェロモンmax値の不整合**
   - 初期帯域: 例えば50Mbps → `max_pheromone = 50^5 = 312,500,000`
   - manual環境で帯域を100Mbpsに変更
   - ❌ しかし、`max_pheromone`は更新されない（本来は`100^5 = 10,000,000,000`であるべき）
   - 提案手法のフェロモン付加量: `100 * 10 * 2.0 = 2000`
   - → 問題なく蓄積できるが、最大値が低いままだと学習効率が低下

2. **config.yamlのパラメータ混在**
   - config.yamlに提案手法用と従来手法用のパラメータが混在
   - 全ての手法が同じconfig.yamlを使うため、特定の手法に特化した値は設定できない

---

## ✅ 修正内容

### 1. manual環境でのフェロモンmin/max再計算（run_experiment.py）

最適経路を100Mbpsに設定した後、フェロモンのmin/maxを既存実装と同じロジックで再計算：

```python
# 最適経路の各エッジの帯域幅を100Mbpsに設定（双方向）
base_min_pheromone = config["aco"]["min_pheromone"]
for u, v in zip(optimal_path[:-1], optimal_path[1:]):
    # 帯域を100Mbpsに設定
    graph.graph.edges[u, v]["bandwidth"] = 100.0
    graph.graph.edges[v, u]["bandwidth"] = 100.0
    # ... 他の属性も更新 ...
    
    # 【重要】フェロモンのmin/maxを再計算（既存実装と同じロジック）
    # min_pheromone: 次数（degree）に基づいて計算（双方向で異なる）
    degree_u = len(list(graph.graph.neighbors(u)))
    degree_v = len(list(graph.graph.neighbors(v)))
    graph.graph.edges[u, v]["min_pheromone"] = base_min_pheromone * 3 // degree_u
    graph.graph.edges[v, u]["min_pheromone"] = base_min_pheromone * 3 // degree_v
    
    # max_pheromone: 帯域幅の5乗（既存実装と同じ）
    graph.graph.edges[u, v]["max_pheromone"] = int(100.0**5)
    graph.graph.edges[v, u]["max_pheromone"] = int(100.0**5)
```

**重要ポイント**:
- `min_pheromone`は**双方向で異なる値**になる（ノードの次数による）
- 例: ノードA（degree=6）→B（degree=10）: `min = 100 * 3 // 6 = 50`
- 例: ノードB（degree=10）→A（degree=6）: `min = 100 * 3 // 10 = 30`

### 2. config.yamlのパラメータ整理（提案手法・先行研究専用）

**重要**: config.yamlは**提案手法・先行研究用のパラメータのみ**を設定します。従来手法（Conventional ACS）は全てのパラメータをConventionalACOSolver内で明示的に定義します。

#### config.yaml（提案手法・先行研究専用）

```yaml
aco:
  # 提案手法・先行研究用のパラメータのみ
  # 従来手法はConventionalACOSolver内で全て定義
  beta_bandwidth: 1.0
  evaporation_rate: 0.02  # 残存率0.98
  min_pheromone: 100
  max_pheromone: 1000000000
  epsilon: 0.1  # ε-Greedy用
  ttl: 100  # 全手法共通
```

#### ConventionalACOSolver（全パラメータを明示的に定義）

```python
def __init__(self, config: Dict, graph: RoutingGraph):
    # ACS論文準拠の値を明示的に設定（configから読み込まない）
    self.alpha = 1.0
    self.beta_bandwidth = 2.0  # または config.get("beta_bandwidth", 2.0)
    self.beta_delay = 1.0
    self.q0 = 0.9
    self.local_update_xi = 0.1
    self.initial_pheromone = 1.0
    self.bandwidth_normalization = 100.0
    self.evaporation_rate = 0.1  # ρ
    self.min_pheromone = 0.01
    self.max_pheromone = 10.0
    
    # グラフのフェロモンを再初期化（RoutingGraphは100で初期化されているため）
    self._reinitialize_pheromones()

def _reinitialize_pheromones(self) -> None:
    """グラフのフェロモン値をACS論文準拠の値に再初期化"""
    G = self.graph.graph
    for u, v in G.edges():
        G.edges[u, v]["pheromone"] = self.initial_pheromone  # 1.0
        G.edges[v, u]["pheromone"] = self.initial_pheromone
        G.edges[u, v]["min_pheromone"] = self.min_pheromone  # 0.01
        G.edges[v, u]["min_pheromone"] = self.min_pheromone
        G.edges[u, v]["max_pheromone"] = self.max_pheromone  # 10.0
        G.edges[v, u]["max_pheromone"] = self.max_pheromone
```

**理由**:
- config.yamlは全ての手法で共有されるため、特定の手法専用の値は設定できない
- 従来手法は完全に独立して動作するよう、全パラメータを内部で定義
- 提案手法・先行研究はconfig.yamlの値をそのまま使用

---

## 📊 パラメータ比較表

| パラメータ | 提案手法（proposed） | 従来手法（conventional） | 先行研究（previous） | 設定場所 |
|----------|-------------------|------------------------|---------------------|---------|
| alpha | 1.0 | 1.0 | 1.0 | config.yaml / Solver |
| beta_bandwidth | 1.0 | 2.0 | 1.0 | config.yaml / **Solver** |
| evaporation_rate | 0.02 | 0.1（ρ） | 0.02 | config.yaml / **Solver** |
| min_pheromone | 100 | 0.01 | 100 | config.yaml / **Solver** |
| max_pheromone | 10^9 | 10.0 | 10^9 | config.yaml / **Solver** |
| initial_pheromone | min値 | 1.0（τ₀） | min値 | - / **Solver** |
| q0 | - | 0.9 | - | - / **Solver** |
| local_update_xi | - | 0.1（ξ） | - | - / **Solver** |
| bandwidth_norm | - | 100.0 | - | - / **Solver** |
| epsilon | 0.1 | - | 0.1 | config.yaml |
| 探索戦略 | ε-Greedy | Pseudo-Random | ε-Greedy | - |
| フェロモン更新 | 全アリ即座更新 | Global Bestのみ | 全アリ即座更新 | - |
| ノード学習 | BKB/BLD/BKH | なし | w^min/w^max | - |

**設定場所の説明**:
- `config.yaml`: config.yamlで設定（提案手法・先行研究用）
- `Solver`: 各ソルバーの`__init__`で明示的に設定
- **太字**: 従来手法はconfig.yamlの値を使わず、Solver内で明示的に設定

---

## 🔍 min_pheromoneの計算ロジック

**config.yaml**:
```yaml
min_pheromone: 100  # ベース値
```

**実際の計算**（graph.py, run_experiment.py）:
```python
# 双方向で異なる値を計算
degree_u = len(list(graph.neighbors(u)))
degree_v = len(list(graph.neighbors(v)))

# u → v のmin_pheromone
min_pheromone_u_to_v = base_min_pheromone * 3 // degree_u

# v → u のmin_pheromone
min_pheromone_v_to_u = base_min_pheromone * 3 // degree_v
```

**理由**:
- 次数が高いノード（ハブノード）からのエッジは、選択肢が多いため、最小フェロモンを低く設定
- 次数が低いノード（末端ノード）からのエッジは、選択肢が少ないため、最小フェロモンを高く設定
- これにより、探索の多様性を維持

---

## ✅ 期待される効果

### manual環境での改善

1. **フェロモン蓄積の正常化**
   - max_pheromoneが正しく`100^5 = 10,000,000,000`に設定される
   - フェロモン付加が適切に機能し、学習が進む

2. **成功率の向上**
   - 60% → 80%（barabasi_albert環境と同等）
   - 既存実装（aco_main_bkb_available_bandwidth.py）と同じ結果

### 全環境での一貫性

- 提案手法（proposed）: beta=1.0, evap=0.02を使用
- 従来手法（conventional）: beta=2.0, evap=0.1を内部で設定
- 各手法が独立して正しいパラメータを使用

---

## 🧪 検証方法

### manual環境でのテスト

```bash
cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/aco_moo_routing

# config.yamlでmanual環境を設定
# graph_type: "manual"
# method: "proposed"

python experiments/run_experiment.py
```

### 確認ポイント

1. ✅ 最適解到達率が80%前後になるか
2. ✅ フェロモン値が適切に蓄積されるか（debug出力で確認）
3. ✅ BKB値が適切に更新されるか

---

## 📝 修正ファイル一覧

1. **run_experiment.py**
   - manual環境でのフェロモンmin/max再計算を追加

2. **config.yaml**
   - パラメータの説明を更新
   - 提案手法・先行研究のデフォルト値を設定

3. **conventional_aco_solver.py**
   - ACS論文準拠の値を内部で設定
   - beta_bandwidth, evaporation_rate, min/max_pheromoneを上書き

---

## 🎓 学んだこと

1. **config.yamlは全手法で共有される**
   - 手法ごとに異なる値が必要な場合は、各ソルバーで内部設定

2. **manual環境は特殊な処理が必要**
   - 帯域を動的に変更した後は、関連する属性（min/max_pheromone）も再計算

3. **min_pheromoneは双方向で異なる**
   - ノードの次数に基づいて計算される
   - 初期フェロモンとは異なる概念

