# BKB 揮発の問題点の分析

## ユーザーの指摘

**「リングバッファの値を揮発させないと毎回 max を取るから変わらない」**

---

## 現在の実装の動作

### 1. リングバッファと BKB 値の管理

```python
# src/bkb_learning.py - update_node_bkb_time_window_max()

# リングバッファに観測値を追加（揮発なし）
window_values.append(bottleneck)

# サイズを超えたら古いものを削除（FIFO）
while len(window_values) > time_window_size:
    window_values.pop(0)

# バッファ内の最大値をBKBとして設定（毎回上書き）
time_window_max = max(window_values) if window_values else 0
graph.nodes[node]["best_known_bottleneck"] = int(time_window_max)
```

### 2. BKB 値の揮発処理

```python
# src/bkb_learning.py - evaporate_bkb_values()

# BKB値に揮発を適用
new_value = graph.nodes[node]["best_known_bottleneck"] * evaporation_rate
graph.nodes[node]["best_known_bottleneck"] = new_value
```

### 3. 実行順序

```
世代N:
  [1] アリがノードXを通る
      → update_pheromone() が呼ばれる
      → update_node_bkb_time_window_max() が呼ばれる
      → リングバッファのmax値でBKBを上書き
      → BKB = max(観測値1, 観測値2, ..., 観測値10)

  [2] 世代の終わり
      → evaporate_bkb_values() が呼ばれる
      → BKB = BKB × 0.999（揮発）

世代N+1:
  [1] アリがノードXを通る
      → update_node_bkb_time_window_max() が呼ばれる
      → リングバッファのmax値でBKBを上書き（揮発の効果が消える）
      → BKB = max(観測値1, 観測値2, ..., 観測値10)  ← 元に戻る
```

---

## 問題点

### 1. リングバッファ内の値が揮発されない

- リングバッファ（`time_window_values`）には**生の観測値**が保存される
- これらの値は揮発されない
- したがって、`max(window_values)` は常に同じ（または大きい）値になる

### 2. BKB 値の揮発が即座に無効化される

- `evaporate_bkb_values()` で BKB 値に揮発を適用しても、次にアリがノードを通ると、**max 値で上書きされる**
- つまり、揮発の効果は**次にアリが通るまでの一瞬だけ**

### 3. 実質的に揮発が意味をなさない

```
世代0: BKB = 100（リングバッファのmax値）
       → 揮発後: BKB = 99.9

世代1: アリがノードを通る
       → update_node_bkb_time_window_max() が呼ばれる
       → BKB = max(観測値...) = 100（元に戻る）
       → 揮発後: BKB = 99.9

世代2: アリがノードを通る
       → BKB = max(観測値...) = 100（元に戻る）
       → 揮発後: BKB = 99.9

...

世代100: アリがノードを通る
         → BKB = max(観測値...) = 100（元に戻る）
         → 揮発後: BKB = 99.9
```

**「世代 0 で BKB=100 → 100 世代後 ≈ 90.5」という説明は間違っています。**

---

## 揮発が唯一効果を発揮するケース

### アリがそのノードを通らない期間

```
世代0: BKB = 100（リングバッファのmax値）
       → 揮発後: BKB = 99.9

世代1: アリがノードXを通らない
       → BKB更新なし（99.9のまま）
       → フェロモン揮発時にBKB = 99.9が参照される
       → 揮発後: BKB = 99.9 × 0.999 = 99.8

世代2: アリがノードXを通らない
       → BKB更新なし（99.8のまま）
       → 揮発後: BKB = 99.8 × 0.999 = 99.7

...

世代100: アリがノードXを通らない
         → BKB = 100 × 0.999^100 ≈ 90.5

世代101: アリがノードXを通る
         → update_node_bkb_time_window_max() が呼ばれる
         → BKB = max(観測値...) = 100（元に戻る）
         → 揮発の効果が消える
```

**しかし、このケースでも、次にアリが通った瞬間に元に戻るため、効果は一時的なもの**

---

## なぜ「精度が変わる」のか？

### 可能性 1: 実際には効果が小さい

- アリがノードを通らない期間中だけ効果がある
- しかし、次の世代にアリが通ると即座に元に戻る
- したがって、**実質的な効果は非常に限定的**

### 可能性 2: 別の要因が影響している

- リングバッファサイズが小さい（例: 10）場合、古い高い値が FIFO で削除される
- 帯域変動後、新しい低い観測値が追加される
- しかし、これは「揮発」ではなく「FIFO 削除」の効果

### 可能性 3: 観測値の分布による影響

- リングバッファ内の観測値が全て同じなら、max 値は変わらない
- しかし、観測値にバラツキがあれば、max 値も変動する
- 揮発とは別の要因で精度が変わる可能性がある

---

## 結論

**ユーザーの指摘は完全に正しい**

1. **リングバッファ内の値が揮発されない** → max 値は変わらない
2. **BKB 値に揮発を適用しても、次に max 値で上書きされる** → 効果が即座に無効化される
3. **実質的に揮発は意味をなさない** → アリがノードを通らない期間中だけ一時的に効果がある

**したがって、現在の実装では「世代 0 で BKB=100 → 100 世代後 ≈ 90.5」という説明は間違っています。**

---

## 正しい実装方針

### オプション 1: リングバッファ内の値に揮発を適用

```python
def update_node_bkb_time_window_max(...):
    # リングバッファ内の各値に揮発を適用
    for i in range(len(window_values)):
        window_values[i] *= evaporation_rate

    # 新しい観測値を追加
    window_values.append(bottleneck)

    # サイズを超えたら古いものを削除
    while len(window_values) > time_window_size:
        window_values.pop(0)

    # バッファ内の最大値をBKBとして設定
    time_window_max = max(window_values) if window_values else 0
    graph.nodes[node]["best_known_bottleneck"] = int(time_window_max)
```

### オプション 2: BKB 揮発を削除し、FIFO のみに依存

```python
# BKB揮発を削除
# evaporate_bkb_values() を呼ばない

# リングバッファのFIFO削除のみで「忘却」を実現
```

### オプション 3: 世代情報を記録して時間減衰

```python
# リングバッファに (値, 世代) のタプルを保存
window_values.append((bottleneck, generation))

# max値を計算する際に、世代差を考慮して重み付け
# 古い値ほど価値が低くなる
```

---

## 今後の対応

ユーザーの指摘に基づき、以下のいずれかを実装する必要がある：

1. **リングバッファ内の値に揮発を適用**（オプション 1）
2. **BKB 揮発を削除**（オプション 2）
3. **世代情報を記録して時間減衰**（オプション 3）
