# ACO システムアーキテクチャ設計

## 概要

Ant Colony Optimization (ACO) システムのコードを機能ごとにモジュール化し、各モジュールが明確な責任を持つように設計しています。

## モジュール構成

### 1. `src/bkb_learning.py` - BKB 学習モジュール

**責任**: ノードの BKB（Best Known Bottleneck）学習方法を提供

**提供する機能**:

- リングバッファベースの BKB 学習（`update_node_bkb_time_window_max`）
- 統計的 BKB 学習（EMA ベース、RFC 6298 準拠）
- BKB 値の揮発処理（`evaporate_bkb_values`）
- ノード初期化（`initialize_graph_nodes_for_simple_bkb`, `initialize_graph_nodes_for_bkb`）

**特徴**:

- 分散型システムを前提：ノードは世代数を知らない
- 観測値ベースの学習：リングバッファは「直近 N 個の観測値」を記憶
- BKB 値の更新アルゴリズムのみを管理

**使用例**:

```python
from bkb_learning import (
    update_node_bkb_time_window_max,
    evaporate_bkb_values,
)

# BKB更新
update_node_bkb_time_window_max(
    graph, node, bottleneck_value, generation,
    time_window_size=10
)

# BKB揮発
evaporate_bkb_values(graph, evaporation_rate=0.999)
```

---

### 2. `src/bandwidth_fluctuation_config.py` - 帯域変動設定モジュール

**責任**: ネットワークの帯域変動の設定と実行を提供

**提供する機能**:

- 帯域変動間隔の設定（`BANDWIDTH_UPDATE_INTERVAL`）
- 変動エッジの選択方法（ハブノード、ランダム、媒介中心性など）
- AR(1)モデルによる帯域変動の生成（`update_available_bandwidth_ar1`）
- AR(1)状態の初期化（`initialize_ar1_states`）

**特徴**:

- 帯域変動の「どのエッジが」「いつ」「どのように」変動するかを管理
- 選択方法は統一パラメータ（`FLUCTUATION_PERCENTAGE`）で制御

**使用例**:

```python
from bandwidth_fluctuation_config import (
    select_fluctuating_edges,
    initialize_ar1_states,
    update_available_bandwidth_ar1,
)

# 変動エッジの選択
fluctuating_edges = select_fluctuating_edges(graph, generation)

# 帯域変動
update_available_bandwidth_ar1(graph, fluctuating_edges)
```

---

### 3. `src/pheromone_update.py` - フェロモン更新モジュール

**責任**: フェロモンの付加・揮発処理を提供

**提供する機能**:

- フェロモン付加量の計算（シンプル版・統計版）
- フェロモン揮発処理（複数のモードに対応）
- フェロモン更新（BKB 更新と連携）

**特徴**:

- BKB 学習方法に依存しない設計（BKB 値は参照するが、学習方法には依存しない）
- パラメータを引数で注入する形式
- 揮発モードは引数で指定可能
- **揮発率は 3 つの要因で決定**：
  1. **基本揮発率**（`base_evaporation_rate`）：世代による一定割合の揮発
  2. **BKB ベースのペナルティ**（`penalty_factor`）：BKB を下回るエッジへのペナルティ（既存機能）
  3. **帯域変動パターンに基づく適応的揮発**（`adaptive_rate_func`）：エッジの可用帯域変動に応じた調整（将来実装予定）

**使用例**:

```python
from pheromone_update import (
    update_pheromone,
    volatilize_by_width,
    calculate_pheromone_increase_simple,
)

# フェロモン更新
update_pheromone(
    ant, graph, generation,
    max_pheromone=MAX_F,
    achievement_bonus=1.5,
    bkb_update_func=update_node_bkb_time_window_max,
)

# フェロモン揮発（基本版：BKBベースのペナルティのみ）
volatilize_by_width(
    graph,
    volatilization_mode=3,
    base_evaporation_rate=0.98,  # 世代による一定割合の揮発
    penalty_factor=0.5,  # BKBを下回るエッジへのペナルティ
    adaptive_rate_func=None,  # 帯域変動パターンに基づく適応的揮発（将来実装）
)

# フェロモン揮発（将来：帯域変動パターンに基づく適応的揮発を追加）
# from bandwidth_monitoring import calculate_adaptive_evaporation_rate
# volatilize_by_width(
#     graph,
#     volatilization_mode=3,
#     base_evaporation_rate=0.98,
#     penalty_factor=0.5,
#     adaptive_rate_func=calculate_adaptive_evaporation_rate,  # sin関数のような周期的変動を検出
# )
```

---

### 4. `src/bandwidth_monitoring.py` - 帯域監視・学習モジュール（将来実装予定）

**責任**: エッジの利用可能帯域を常時監視し、帯域変動パターンを学習

**想定する機能**:

- エッジの利用可能帯域の継続的監視
- 帯域変動パターンの学習（AR(1)パラメータの推定など）
- 変動予測に基づく適応的揮発率の計算
- 変動適応型 BKB 揮発への統合

**特徴**:

- `bkb_learning.py`とは別：BKB 値の更新ではなく、帯域変動パターンの学習
  - `bkb_learning.py`: ノードが知っている「最良のボトルネック帯域」を学習・記憶
  - `bandwidth_monitoring.py`: エッジの「利用可能帯域の変動パターン」を学習
- リングバッファとは異なる：継続的な監視データに基づく学習
  - リングバッファ：直近 N 個の観測値の最大値を記憶（BKB 値の更新用）
  - 帯域監視：エッジの帯域を定期的に観測し、変動パターン（統計特性）を学習
- 分散型システムを前提：各ノードが接続エッジの帯域を監視

**設計の方向性**:

- 各ノードは接続エッジの帯域を定期的に観測
  - 観測タイミング：アリがエッジを通過するたび、または定期的なサンプリング
  - 観測データ：エッジの利用可能帯域の時系列データ
- 観測データから変動パターン（平均、分散、自己相関）を推定
  - AR(1)モデルのパラメータ推定
  - 変動係数（CV: Coefficient of Variation）の計算
  - トレンド（増加傾向・減少傾向）の検出
- 変動パターンに基づいて、適応的な揮発率を計算
  - 高変動環境：揮発率を上げる（古い情報を早く忘れる）
  - 低変動環境：揮発率を下げる（長期的な情報を保持）
- この揮発率を `pheromone_update.py` に提供

**データ構造（想定）**:

```python
# グラフのエッジに追加される属性
graph[u][v]["bandwidth_history"] = [80, 75, 82, ...]  # 時系列データ
graph[u][v]["bandwidth_pattern"] = {
    "mean": 78.5,
    "variance": 12.3,
    "cv": 0.14,  # 変動係数
    "ar_coefficient": 0.8,  # AR(1)係数（推定値）
    "trend": "stable",  # "increasing", "decreasing", "stable"
    "periodicity": None,  # 周期的変動の検出（例：sin関数のような変動）
    "next_low_period": None,  # 次の低帯域時期の予測
}
```

**揮発率への統合**:

- `bandwidth_monitoring.py` で学習した変動パターンを `adaptive_rate_func` として提供
- `pheromone_update.py` の `apply_volatilization` 関数内で適用
- 例：周期的変動を検出した場合、次の低帯域時期を予測して揮発を促進（`adaptive_multiplier < 1.0`）
- 例：安定している場合は揮発を抑制（`adaptive_multiplier > 1.0`）

**将来の実装例（イメージ）**:

```python
from bandwidth_monitoring import (
    observe_edge_bandwidth,
    learn_bandwidth_pattern,
    calculate_adaptive_evaporation_rate,
)

# エッジの帯域を観測（アリがエッジを通過したとき）
observe_edge_bandwidth(graph, u, v, current_bandwidth)

# 変動パターンを学習（定期的に実行、または必要に応じて）
pattern = learn_bandwidth_pattern(graph, u, v)

# 適応的揮発率を計算
evaporation_rate = calculate_adaptive_evaporation_rate(
    graph, u, v, base_rate=0.98
)

# pheromone_update.py で使用
volatilize_by_width(
    graph,
    volatilization_mode=3,
    base_evaporation_rate=evaporation_rate,  # 適応的揮発率を使用
    penalty_factor=0.5,
)
```

**`bkb_learning.py` との違い**:
| 項目 | `bkb_learning.py` | `bandwidth_monitoring.py` |
|------|------------------|--------------------------|
| 学習対象 | ノードの BKB 値（最良のボトルネック帯域） | エッジの利用可能帯域の変動パターン |
| データ保存場所 | ノード属性（`graph.nodes[node]`） | エッジ属性（`graph.edges[u, v]`） |
| 学習方法 | リングバッファ（直近 N 個の最大値） | 時系列解析（AR(1)パラメータ推定など） |
| 用途 | BKB 値の更新、フェロモンボーナス計算 | 適応的揮発率の計算 |
| 更新タイミング | アリがノードを通過したとき | エッジの帯域を観測したとき |

---

## 各 ACO 実装ファイルの役割

### `src/aco_main_bkb_available_bandwidth_rfc.py`

- **目的**: ノードの学習方法を様々に試すための実験用ファイル
- **特徴**:
  - BKB 学習方法を柔軟に切り替え可能
  - 統計的 BKB 学習、リングバッファ学習など様々な手法を試せる
  - 学習方法のパラメータを調整して性能を比較

### `src/aco_main_bkb_available_bandwidth.py`

- **目的**: 通常の BKB 学習（MAX 値に着目）を使用
- **特徴**:
  - リングバッファサイズと忘却率を調整して実験
  - シンプルな MAX 値ベースの BKB 学習
  - 主に静的・動的環境での基本的な性能評価

### `src/aco_main_bkb_available_bandwidth_ave.py`

### `src/aco_main_bkb_available_bandwidth_ave_v1.py`

- **目的**: 帯域値の絶対値に着目し、平均的に良い経路を選択できるか評価
- **特徴**:
  - 最適解だけでなく、経路の「質」に着目
  - 平均ボトルネック帯域を記録・評価
  - リングバッファサイズ 1 の BKB 学習を使用

---

## データフロー

```
┌─────────────────────────────────────────────────────────┐
│ aco_main_*.py (メインループ)                             │
│  - 世代ループの管理                                        │
│  - アリの生成・移動                                        │
│  - ログ記録                                               │
└──────────────┬──────────────────────────────────────────┘
               │
               ├─→ bandwidth_fluctuation_config.py
               │   帯域変動の実行（AR(1)モデル）
               │
               ├─→ bkb_learning.py
               │   ノードのBKB更新（リングバッファ学習など）
               │
               ├─→ pheromone_update.py
               │   フェロモンの付加・揮発
               │   └─→ bkb_learning.py (BKB値を参照)
               │
               └─→ [将来] bandwidth_monitoring.py
                    エッジの帯域監視・学習
                    └─→ pheromone_update.py (適応的揮発率を提供)
```

---

## 設計原則

### 1. 関心の分離（Separation of Concerns）

- 各モジュールは単一の責任を持つ
- BKB 学習とフェロモン更新は独立
- 帯域変動と帯域監視は別モジュール

### 2. 依存性注入（Dependency Injection）

- パラメータは引数で注入
- グローバル変数への依存を最小化
- テスト容易性を向上

### 3. 分散型システムへの配慮

- ノードは世代数を知らない
- 観測値ベースの学習（世代ベースではない）
- ローカル情報のみを使用

### 4. 拡張性

- 新しい学習方法を追加しやすい
- 新しい揮発モードを追加しやすい
- モジュール間の結合度を低く保つ

---

## 実装の進め方

### Phase 1: フェロモン更新モジュールの統合（進行中）

1. ✅ `pheromone_update.py` の実装完了
2. ⏳ 各 `aco_main_*.py` ファイルを `pheromone_update.py` に依存させる形にリファクタリング
   - `aco_main_bkb_available_bandwidth.py`
   - `aco_main_bkb_available_bandwidth_rfc.py`
   - `aco_main_bkb_available_bandwidth_ave.py`
   - `aco_main_bkb_available_bandwidth_ave_v1.py`
3. ⏳ 重複コードの削除
   - `volatilize_by_width` 関数
   - `_apply_volatilization` 関数
   - `calculate_pheromone_increase` 関数
   - `update_pheromone` 関数

**実装方針**:

- 各ファイルの既存の関数を `pheromone_update.py` の関数呼び出しに置き換える
- パラメータ（揮発率、ボーナス係数など）は各ファイルで定義し、引数で注入
- BKB 更新関数はコールバックとして渡す

### Phase 2: 帯域監視モジュールの実装（将来）

1. `bandwidth_monitoring.py` の実装
   - エッジ帯域の継続的監視機能
   - 変動パターンの学習（AR(1)パラメータ推定）
   - 適応的揮発率の計算

**実装方針**:

- アリがエッジを通過するたびに帯域を観測
- 観測データをエッジ属性に保存（リングバッファ形式）
- 定期的に（または必要に応じて）変動パターンを学習
- 学習結果をエッジ属性に保存

2. `pheromone_update.py` との統合
   - `apply_volatilization` 関数に適応的揮発率のオプションを追加
   - `bandwidth_monitoring.py` から適応的揮発率を取得する機能を追加

### Phase 3: 変動適応型 BKB 揮発の統合（将来）

1. 変動適応型 BKB 揮発の実装
   - `bandwidth_monitoring.py` と `pheromone_update.py` の連携
   - 変動パターンに基づく動的揮発率調整
   - 性能評価とパラメータ最適化

**実装方針**:

- エッジの変動パターンに応じて揮発率を動的に調整
- 高変動環境では揮発を促進、低変動環境では揮発を抑制
- リングバッファの BKB 学習と組み合わせて使用

---

## モジュール間の依存関係

### 現在の依存関係

```
aco_main_*.py
  ├─→ bkb_learning.py
  │    (BKB学習・更新)
  │
  ├─→ bandwidth_fluctuation_config.py
  │    (帯域変動の設定・実行)
  │
  └─→ pheromone_update.py
       (フェロモンの付加・揮発)
       └─→ bkb_learning.py (BKB値を参照)
```

### 将来の依存関係（bandwidth_monitoring.py 実装後）

```
aco_main_*.py
  ├─→ bkb_learning.py
  │    (BKB学習・更新)
  │
  ├─→ bandwidth_fluctuation_config.py
  │    (帯域変動の設定・実行)
  │
  ├─→ bandwidth_monitoring.py
  │    (エッジ帯域の監視・変動パターン学習)
  │
  └─→ pheromone_update.py
       (フェロモンの付加・揮発)
       ├─→ bkb_learning.py (BKB値を参照)
       └─→ bandwidth_monitoring.py (適応的揮発率を参照)
```

**重要なポイント**:

- `pheromone_update.py` は `bkb_learning.py` に依存する（BKB 値を参照するため）
- `bandwidth_monitoring.py` は `bkb_learning.py` に依存しない（別の目的：エッジ帯域の変動パターン学習）
- `bandwidth_monitoring.py` と `bkb_learning.py` は独立したモジュール（データ保存場所も異なる）
- すべてのモジュールは `aco_main_*.py` から呼び出される
- `pheromone_update.py` は `bandwidth_monitoring.py` から適応的揮発率を受け取る（将来）

---

## 命名規則

- **BKB 学習**: `bkb_learning.py` で管理
- **帯域変動**: `bandwidth_fluctuation_config.py` で管理
- **フェロモン更新**: `pheromone_update.py` で管理
- **帯域監視**: `bandwidth_monitoring.py` で管理（将来）

各モジュールは明確な責任を持ち、他のモジュールの詳細に依存しないように設計されています。
