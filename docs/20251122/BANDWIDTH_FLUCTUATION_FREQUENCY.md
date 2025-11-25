# 帯域変動の頻度設定

## 📍 設定場所

帯域変動の頻度は **`config/config.yaml`** で設定されます：

```yaml
graph:
  fluctuation:
    enabled: true
    model: "ar1"
    target_method: "hub"
    target_percentage: 0.1
    update_interval: 1 # ← ここで設定
```

## ⚙️ `update_interval`の意味

- **`update_interval: 1`**: 毎世代更新（デフォルト）
- **`update_interval: 2`**: 2 世代ごとに更新
- **`update_interval: 10`**: 10 世代ごとに更新
- **`update_interval: 100`**: 100 世代ごとに更新

## 🔧 実装箇所

帯域変動の更新は **`src/aco_routing/algorithms/aco_solver.py`** で制御されています：

```python
for generation in range(generations):
    # 帯域変動（update_intervalに応じて更新頻度を制御）
    if self.fluctuation_model is not None:
        update_interval = self.config["graph"]["fluctuation"].get("update_interval", 1)
        if generation % update_interval == 0:
            self.fluctuation_model.update(self.edge_states, generation)
```

## 📊 使用例

### 毎世代更新（最も動的）

```yaml
update_interval: 1
```

- 各世代で帯域が変動
- 動的環境のシミュレーションに適している

### 10 世代ごとに更新（中程度の動的環境）

```yaml
update_interval: 10
```

- 10 世代に 1 回だけ帯域が変動
- 比較的安定した環境のシミュレーションに適している

### 100 世代ごとに更新（比較的静的環境）

```yaml
update_interval: 100
```

- 100 世代に 1 回だけ帯域が変動
- ほぼ静的な環境のシミュレーションに適している

## ⚠️ 注意点

- `update_interval`が大きいほど、帯域変動の頻度が低くなります
- `update_interval: 0` は無効（毎世代更新される）
- デフォルト値は `1`（毎世代更新）
