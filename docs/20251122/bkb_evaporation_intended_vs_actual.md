# BKB揮発の期待動作と実際の実装の違い

## ユーザーの期待：時間減衰型の忘却

### 期待される動作

**観測値そのものが「時間とともに価値が減る」**

```
1世代目で観測した値100:
  1世代後: 100 × 0.999 = 99.9
  10世代後: 100 × 0.999^10 ≈ 99.0
  100世代後: 100 × 0.999^100 ≈ 90.5
  1000世代後: 100 × 0.999^1000 ≈ 36.8
```

**意味**: 
- 古い観測値は「陳腐化」する
- 1000世代前の観測値は、現在の帯域状況とは関係ない（帯域が変動しているため）
- 最近の観測値ほど重みが大きい

---

## 現在の実装：リングバッファのみで忘却

### 実際の動作

**リングバッファ内の観測値は揮発されない（そのまま保存）**

```
1世代目: 観測値100 → バッファ=[100], BKB=100
揮発後: BKB=100×0.999=99.9

2世代目: 観測値110 → バッファ=[100,110], BKB=max([100,110])=110
揮発後: BKB=110×0.999=109.89

...

1000世代後:
  バッファ=[最新10個の観測値]（100は削除済み）
  BKB=max(最新10個)
```

**意味**: 
- リングバッファのFIFOで古い値を削除（「忘却」）
- しかし、バッファ内の観測値は「そのまま」保存される
- 1000世代前の値は削除されているが、それは「忘却」ではなく「メモリ不足による削除」

---

## 違いの核心

### 期待される動作（時間減衰型）

```
観測値100を1世代目で記録:
  バッファ内の値: 100
  2世代目: バッファ内の値 = 100 × 0.999 = 99.9
  3世代目: バッファ内の値 = 99.9 × 0.999 = 99.8
  ...
  1000世代目: バッファ内の値 = 100 × 0.999^1000 ≈ 36.8

→ 古い観測値は「価値が減る」が「残っている」
→ 1000世代後でも、36.8という情報として存在する
```

### 現在の実装（FIFO削除型）

```
観測値100を1世代目で記録:
  バッファ内の値: 100（そのまま）
  2世代目: バッファ内の値 = 100（そのまま）
  ...
  11世代目: バッファから削除（FIFO）

→ 古い観測値は「削除される」か「そのまま残る」
→ 10世代を超えると完全に消える
```

---

## 修正が必要な理由

### 現在の実装の問題

1. **観測値が揮発されない**
   - 1000世代前の値がバッファに残っていれば、そのまま100として扱われる
   - 「古い情報」と「新しい情報」の区別がない

2. **本当の「忘却」ではない**
   - FIFO削除は「メモリ制限」であって「忘却」ではない
   - 帯域変動を考慮した「陳腐化」を表現できていない

3. **BKB揮発が意味をなさない**
   - BKB値はmax値で上書きされるため、揮発の効果がリセットされる

---

## 修正案：リングバッファ内の値も揮発させる

### 実装方針

**リングバッファ内の各観測値に、経過世代数を考慮した揮発を適用**

```python
def update_node_bkb_time_window_max(
    graph: nx.Graph,
    node: int,
    bottleneck: float,
    generation: int,
    time_window_size: int = 10,
    evaporation_rate: float = 0.999,
) -> None:
    # リングバッファの初期化
    if "time_window_values" not in graph.nodes[node]:
        graph.nodes[node]["time_window_values"] = []
    if "time_window_generations" not in graph.nodes[node]:
        graph.nodes[node]["time_window_generations"] = []  # 観測時の世代を記録

    window_values = graph.nodes[node]["time_window_values"]
    window_generations = graph.nodes[node]["time_window_generations"]

    # 新しい観測値を追加
    window_values.append(bottleneck)
    window_generations.append(generation)

    # ★★★ 各観測値に経過世代数を考慮した揮発を適用 ★★★
    for i in range(len(window_values)):
        generations_passed = generation - window_generations[i]
        # 経過世代数分だけ揮発を適用
        window_values[i] *= (evaporation_rate ** generations_passed)

    # サイズ制限（古いものを削除）
    while len(window_values) > time_window_size:
        window_values.pop(0)
        window_generations.pop(0)

    # ★★★ 揮発後のバッファ内の最大値をBKBとして使用 ★★★
    time_window_max = max(window_values) if window_values else 0
    graph.nodes[node]["best_known_bottleneck"] = int(time_window_max)
```

**しかし、この実装には問題があります**:
- 毎回全ての値を再計算する必要がある（計算量が増える）
- 世代を記録する必要がある

---

## より実用的な修正案

### 案1: 世代ごとにバッファ内の値に揮発を適用（簡易版）

```python
def update_node_bkb_time_window_max(...):
    # 観測値を追加
    window_values.append(bottleneck)
    
    # ★★★ バッファ内の全ての値に1世代分の揮発を適用 ★★★
    for i in range(len(window_values)):
        window_values[i] *= evaporation_rate  # 0.999
    
    # サイズ制限
    while len(window_values) > time_window_size:
        window_values.pop(0)
    
    # max値を計算
    time_window_max = max(window_values) if window_values else 0
    graph.nodes[node]["best_known_bottleneck"] = int(time_window_max)

# 揮発処理（evaporate_bkb_values）は削除
```

**動作**:
```
世代0: 観測値100 → バッファ=[100], BKB=100
世代1: 観測値110 → バッファ内の値に揮発適用 → [100×0.999=99.9]
        新しい値追加 → [99.9, 110]
        BKB=max([99.9, 110])=110
世代2: 観測値105 → バッファ内の値に揮発適用 → [99.9×0.999=99.8, 110×0.999=109.89]
        新しい値追加 → [99.8, 109.89, 105]
        BKB=max([99.8, 109.89, 105])=109.89

...

世代1000: 
  1000世代前の値100 → 100 × 0.999^1000 ≈ 36.8
  新しい値 → そのまま
  BKB=max(揮発後の値)
```

**メリット**:
- 古い観測値が「価値が減る」（ユーザーの期待通り）
- 1000世代前の値は、36.8程度の価値になる

---

### 案2: BKB更新時に揮発後の値とmax値の最小値を使う（現在の実装を活かす）

現在の実装を活かしつつ、揮発の効果を維持する方法：

```python
def update_node_bkb_time_window_max(...):
    # 現在のBKB値（揮発後）を取得
    current_bkb = graph.nodes[node].get("best_known_bottleneck", 0)
    
    # 観測値を追加してmax値を計算
    window_values.append(bottleneck)
    time_window_max = max(window_values) if window_values else 0
    
    # ★★★ 揮発後の値とmax値の最小値を使う ★★★
    # これにより、新しい観測値が来ても揮発の効果が維持される
    new_bkb = min(current_bkb, int(time_window_max))
    
    # ただし、新しい観測値が揮発後の値を超える場合は更新
    if time_window_max > current_bkb:
        new_bkb = int(time_window_max)
    
    graph.nodes[node]["best_known_bottleneck"] = new_bkb
```

**動作**:
```
世代0: 観測値100 → BKB=100, 揮発後=99.9
世代1: 観測値110 → BKB=max(99.9, 110)=110（新しい値で更新）
        揮発後=109.89
世代2: 観測値105 → BKB=max(109.89, 105)=109.89（揮発後の値が大きい）
        揮発後=109.78
世代3: 観測値95 → BKB=max(109.78, 95)=109.78（揮発後の値が大きい）
        揮発後=109.67
```

**問題**: これでも、新しい観測値が来ると上書きされてしまう。

---

## 推奨される修正

**案1を推奨**: リングバッファ内の値に毎世代揮発を適用する

### 実装の変更点

1. `update_node_bkb_time_window_max`を修正
   - 観測値を追加する前に、バッファ内の全値に揮発（0.999）を適用
2. `evaporate_bkb_values`の呼び出しを削除
   - BKB値はバッファ内のmax値なので、別途揮発不要

### 期待される動作

```
世代0: 観測値100 → バッファ=[100], BKB=100
世代1: 
  バッファ内の値に揮発 → [100×0.999=99.9]
  新しい観測値110追加 → [99.9, 110]
  BKB=max([99.9, 110])=110
世代2:
  バッファ内の値に揮発 → [99.9×0.999=99.8, 110×0.999=109.89]
  新しい観測値105追加 → [99.8, 109.89, 105]
  BKB=max([99.8, 109.89, 105])=109.89

...

世代1000:
  1000世代前の値100 → 100 × 0.999^1000 ≈ 36.8
  新しい観測値120 → 120
  BKB=max(揮発後の値)
```

**これで、1000世代前の観測値は「取るに足らない値」（36.8）になります！**

---

## まとめ

### ユーザーの期待

**「1世代目で観測した値100は、1000世代後には0.999^1000 ≈ 36.8という取るに足らない値になる」**

### 現在の実装

**「リングバッファ内の値は揮発されず、10世代を超えると削除される」**

### 修正案

**「リングバッファ内の各値に、毎世代揮発（0.999）を適用する」**

これにより、ユーザーの期待通りの「時間減衰型の忘却」が実現されます。




