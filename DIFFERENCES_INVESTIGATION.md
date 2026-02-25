# 既存実装と新実装の違いの調査結果

## 調査日: 2025年12月21日

## ステータス: ✅ 修正完了

---

## 1. ログ記録の違い（修正済み）

### 既存実装（`aco_main_bkb_available_bandwidth.py`）

**ログ値**:
- `1`: 最適解を見つけた（`min(ant.width) >= current_optimal_bottleneck`）
- `0`: 最適解を見つけられなかった（ゴール到達したが最適解ではない、またはTTLで失敗）

### 新実装（修正前）

**ログ値**:
- `0`: 最適解を見つけた
- `-1`: ゴール未到達（TTLで失敗、または制約違反）
- `-2`: ゴール到達したが最適解ではない

### 新実装（修正後）✅

**ログ値**:
- `1`: 最適解を見つけた
- `0`: 失敗（ゴール未到達、またはゴール到達したが最適解ではない）

**修正対象ファイル**:
- `aco_moo_routing/src/aco_routing/algorithms/aco_solver.py`（proposed）
- `aco_moo_routing/src/aco_routing/algorithms/conventional_aco_solver.py`
- `aco_moo_routing/src/aco_routing/algorithms/previous_method_aco_solver.py`

---

## 2. 集計方法の違い（修正済み）

### 既存実装

```python
# その世代のチャンク内に1が一つでもあれば成功
if 1 in generation_chunk:
    generation_success_count += 1
```

### 新実装（修正前）

```python
# >= 0 を成功としてカウント（問題！）
final_success_rate = sum(1 for idx in ant_log if idx >= 0) / len(ant_log)
```

### 新実装（修正後）✅

```python
# 1 を成功としてカウント（既存実装と同じ）
final_success_rate = sum(1 for idx in ant_log if idx == 1) / len(ant_log)
```

**修正対象ファイル**:
- `aco_moo_routing/experiments/run_experiment.py`

---

## 3. CSVファイルの形式の違い（修正済み）

### 既存実装

- ファイル名: `log_ant.csv`
- 形式: **1行1シミュレーション**、各列がアリ（世代数×アリ数）
- 値: `1/0`

### 新実装（修正前）

- ファイル名: `ant_log.csv`
- 形式: **1行1アリ**、2列（unique_optimal, any_optimal）
- 値: `0/-1/-2`

### 新実装（修正後）✅

- ファイル名: `ant_log.csv`
- 形式: **1行1シミュレーション**、各列がアリ（世代数×アリ数）
- 値: `1/0`

**修正対象ファイル**:
- `aco_moo_routing/experiments/run_experiment.py`

---

## 4. 修正内容のまとめ

### 4.1. ログ値の変更

| 状態 | 修正前 | 修正後 |
|------|--------|--------|
| 最適解 | `0` | `1` |
| ゴール到達、非最適解 | `-2` | `0` |
| ゴール未到達 | `-1` | `0` |

### 4.2. 集計方法の変更

- 修正前: `idx >= 0` で成功をカウント
- 修正後: `idx == 1` で成功をカウント

### 4.3. CSV形式の変更

- 修正前: 1行1アリ、2列
- 修正後: 1行1シミュレーション、各列がアリ（既存の集計スクリプトと互換）

---

## 5. 既存の集計スクリプトとの互換性

修正後、以下の既存の集計スクリプトがそのまま使用可能:

- `csv_log_analysis_comparison_baseline_vs_proposed.py`
- `csv_log_analysis_percentage_of_optimal_solution_use_modified_dijkstra.py`
- その他の `csv_log_analysis_*.py` スクリプト

---

## 6. TTLチェックのタイミング（追加修正）

### 問題点

**既存実装**:
```python
if ant.current == ant.destination:
    # ゴール到達処理
elif len(ant.route) >= TTL:
    # TTL到達（失敗）
```
- **ゴール判定が優先される**
- ゴールに到達していればTTLに達していても成功

**新実装（修正前）**:
```python
if not ant.is_alive():  # TTLチェック
    ant_log.append(0)
    continue
# その後でゴール判定
```
- **TTLチェックが先に行われる**
- TTLに達したアリはゴール判定される前に失敗

### 修正内容 ✅

1. **TTLチェックのタイミング**: 移動後に行うように変更
2. **ゴール判定の優先**: ゴールに到達している場合はTTLチェックをスキップ
3. **TTL条件**: `remaining_ttl <= 1` で既存実装と同等の動作

```python
# 移動後にゴール判定
has_reached_any_goal = ant.current_node in goal_nodes_set

# ゴール未到達の場合のみTTLチェック
if not has_reached_any_goal and ant.remaining_ttl <= 1:
    ant_log.append(0)
    continue
```

**修正対象ファイル**:
- `aco_moo_routing/src/aco_routing/algorithms/aco_solver.py`
- `aco_moo_routing/src/aco_routing/algorithms/conventional_aco_solver.py`
- `aco_moo_routing/src/aco_routing/algorithms/previous_method_aco_solver.py`

---

## 7. 残りの調査項目（参考）

以下の項目は今回の修正対象外ですが、既存実装との違いとして認識しておく必要があります:

1. **最適解判定の型**: 既存実装は整数比較、新実装は浮動小数点数比較
   - フェロモン揮発時の比較は `int()` でキャストするように修正済み
   
2. **フェロモン更新のタイミング**: 両方とも即座に更新（同じ）

3. **フェロモン揮発のタイミング**: 両方とも世代終了時に揮発（同じ）

