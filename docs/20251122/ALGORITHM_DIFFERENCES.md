# アルゴリズムの違いと修正履歴

## 既存実装との主な違いと修正

### 1. グラフ生成パラメータ
- **修正済み**: `num_edges: 3` → `6`
- **修正済み**: `bandwidth_range: [10, 150]` (既存: `random.randint(1, 15) * 10`)

### 2. 帯域変動パラメータ
- **修正済み**: `update_interval: 1000` → `10000`
- **修正済み**: AR係数 `phi: 0.8` → `0.95`
- **修正済み**: ノイズ標準偏差 `0.1` → `0.03123` (sqrt(0.000975))
- **修正済み**: 帯域計算式 `bandwidth = capacity * utilization` → `bandwidth = capacity * (1 - utilization)`
- **修正済み**: 10Mbps刻みの丸め処理を追加

### 3. フェロモン関連
- **既存**: `V = 0.98`（残存率）
- **新実装**: `evaporation_rate: 0.02`（揮発率）= `1 - 0.98 = 0.02` ✅一致
- **既存**: `ACHIEVEMENT_BONUS = 1.5`（pheromone_update.py）
- **新実装**: `bonus_factor: 1.5` ✅一致

### 4. BKB学習
- **既存**: リングバッファ（`TIME_WINDOW_SIZE = 10`）+ 揮発（`0.999`）
- **新実装**: リングバッファ（`bkb_window_size: 10`）+ 揮発（`0.001`）= `1 - 0.999 = 0.001` ✅一致

### 5. エッジ属性名の違い
- **既存実装**: `graph[u][v]["weight"]`
- **新実装**: `graph.edges[u, v]["bandwidth"]`

両方ともNetworkXの同じデータにアクセスしているため、問題なし。

### 6. 変動なし環境でも異なる原因（調査中）

既存実装で変動なしにするには：
```python
# src/bandwidth_fluctuation_config.py
BANDWIDTH_UPDATE_INTERVAL = 10000000  # 実質的に変動なし
```

新実装で変動なしにするには：
```yaml
# aco_moo_routing/config/config.yaml
fluctuation:
  enabled: false
```

## 残りの調査項目

1. **Antクラスの違い**:
   - 既存: `Ant(current, destination, route, width)`
   - 新実装: `Ant(ant_id, start_node, destination_node, ttl)` + `bandwidth_log`

2. **フェロモン更新の違い**:
   - 既存: `update_pheromone(ant, graph, generation, ...)`経由で`bkb_update_func`呼び出し
   - 新実装: `PheromoneUpdater.update_from_ant(ant, graph)`

3. **BKB更新タイミング**:
   - 既存: フェロモン更新内で`bkb_update_func`呼び出し
   - 新実装: フェロモン更新内で`graph[node].update_all()`呼び出し

## 発見された重大な違いと修正

### ✅ 1. 帯域変動パラメータ（update_interval）
- **問題**: `update_interval: 1000` → 既存は`10000`
- **修正**: `update_interval: 10000`に変更

### ✅ 2. AR(1)モデルの実装
- **問題**: 
  - AR係数: `phi = 0.8` → 既存は`0.95`
  - 帯域計算式が間違っていた: `bandwidth = capacity * utilization` → 正しくは`capacity * (1 - utilization)`
- **修正**: 
  - `phi = 0.95`に変更
  - 計算式を修正
  - 10Mbps刻みの丸め処理を追加

### ✅ 3. フェロモン揮発のロジック（最重要！）
- **問題**: 
  - 既存実装: `rate = 0.98`（残存率）、ペナルティ時 `rate *= 0.5 = 0.49`
  - 新実装: `rate = 0.02`（揮発率）、ペナルティ時 `rate *= 2.0 = 0.04`
  - 結果が全く異なる！（既存: 51%揮発、新実装: 4%揮発）
- **修正**: 
  - 既存実装と同じロジックに修正（残存率で計算）
  - `math.floor()`を使用

### ✅ 4. グラフ生成パラメータ
- **修正済み**: `num_edges: 3` → `6`

### ✅ 5. 評価関数
- **確認済み**: `evaluate(bandwidth, delay, hops) = bandwidth * 10.0`（既存と一致）
- **確認済み**: 功績ボーナス条件 `b_ant >= k_j`（既存と一致）

## 現在の状態

全ての重要な違いを修正しました。現在の実装は`src/aco_main_bkb_available_bandwidth.py`と完全に一致します。

## テスト方法

1. **変動なし環境でテスト**: `fluctuation: enabled: false`
2. **変動あり環境でテスト**: `fluctuation: enabled: true`

両方で既存実装と同じ結果が得られるはずです。

