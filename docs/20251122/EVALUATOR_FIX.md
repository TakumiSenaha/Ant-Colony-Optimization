# 評価関数の修正: 帯域のみ最適化への対応

## 🐛 問題

`target_objectives: ["bandwidth"]`の場合、評価関数が対応していなかった。

**エラー**:

```
ValueError: Unknown objectives: ['bandwidth']
```

## ✅ 修正内容

### 1. `evaluate()`メソッドに帯域のみのケースを追加

```python
# 帯域のみ最適化（既存実装との互換性）
if objectives == ["bandwidth"]:
    return self._evaluate_bandwidth_only(bandwidth)
```

### 2. `_evaluate_bandwidth_only()`メソッドを追加

```python
def _evaluate_bandwidth_only(self, bandwidth: float) -> float:
    """
    帯域のみの評価（既存実装との互換性）

    既存実装: calculate_pheromone_increase_simple(bottleneck_bandwidth) = bandwidth * 10
    新実装: evaluate() * bonus_factor で同じ結果になるように、bandwidth * 10 を返す
    """
    return bandwidth * 10.0
```

### 3. `check_bonus_condition()`メソッドに帯域のみのケースを追加

```python
# 帯域のみ最適化（既存実装との互換性）
if objectives == ["bandwidth"]:
    return b_ant >= k_j  # BKBのみを比較
```

## 📊 既存実装との互換性

### フェロモン付加量の計算

**既存実装**:

```python
# pheromone_update.py
def calculate_pheromone_increase_simple(bottleneck_bandwidth: int) -> float:
    return float(bottleneck_bandwidth * 10)

# ボーナスありの場合
delta_pheromone = calculate_pheromone_increase_simple(bottleneck) * achievement_bonus
# = (bottleneck * 10) * 1.5
```

**新実装**:

```python
# evaluator.py
def _evaluate_bandwidth_only(bandwidth: float) -> float:
    return bandwidth * 10.0

# ボーナスありの場合
delta_pheromone = evaluator.evaluate(bandwidth, delay, hops) * bonus_factor
# = (bandwidth * 10) * 1.5
```

**結論**: ✅ **完全に同じ**

## ✅ 確認結果

- ✅ 評価関数: `bandwidth * 10`を返す（既存実装と同じ）
- ✅ ボーナス条件: `b_ant >= k_j`のみをチェック（既存実装と同じ）
- ✅ 全 20 テストケース: 通過

## 🎯 対応済みの目的関数

1. ✅ `["bandwidth"]` - 帯域のみ（既存実装との互換性）
2. ✅ `["bandwidth", "hops"]` - Step 1
3. ✅ `["bandwidth", "delay"]` - Step 2
4. ✅ `["bandwidth", "delay", "hops"]` - Step 3

---

**修正日**: 2024-11-23
**状態**: ✅ 修正完了
