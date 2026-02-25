# Ant-Colony-Optimization

本リポジトリは ACO（Ant Colony Optimization）を用いた経路最適化の実験/分析コードです。  
以下は典型的な利用手順です。

## 環境

- Python 3.11 以上を推奨
- 依存ライブラリは `requirements.txt` に準拠（未導入なら `pip install -r requirements.txt`）

## 実行手順

1. 設定ファイルを編集  
   `aco_moo_routing/config/config.yaml` で目的（`target_objectives`）、世代数、アリ数、遅延制約などを設定。
2. 実験実行  
   （推奨）リポジトリルートから `cd aco_moo_routing` して実行します。
   ```bash
   cd aco_moo_routing
   python experiments/run_experiment.py
   ```
   ※ルート直下から実行したい場合は `python aco_moo_routing/experiments/run_experiment.py` のようにパスを付けてください。
3. 出力先  
   `results/{method}/{environment}/{opt_type}/` に以下が生成されます。
   - `ant_solution_log.csv`（新ログ・各アリの詳細）
   - `generation_stats.csv`（新ログ・世代集計）
   - `ant_log.csv`（従来互換ログ）
   - 可視化画像（ranking など）

## 分析スクリプト（抜粋）

以下は「プロジェクトルートから実行」の例です。

### 4 手法の比較（`compare_methods.py`）

4 手法を任意に選択して比較できます。複数の手法と複数の環境を同時に描画可能です。

```bash
# 1. 提案手法のみを描画
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods proposed \
  --environments static \
  --opt-type bandwidth_only \
  --metric is_optimal

# 2. 4手法すべてを比較
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods basic_aco_no_heuristic basic_aco_with_heuristic previous proposed \
  --environments static \
  --opt-type bandwidth_only \
  --metric is_optimal

# 3. 複数環境を同時に描画
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods proposed \
  --environments static node_switching bandwidth_fluctuation \
  --opt-type bandwidth_only \
  --metric is_optimal

# 4. 複数手法×複数環境
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods previous proposed \
  --environments static bandwidth_fluctuation \
  --opt-type bandwidth_only \
  --metric is_optimal

# 5. 品質スコア（avg_quality）で比較
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods basic_aco_no_heuristic basic_aco_with_heuristic previous proposed \
  --environments static \
  --opt-type bandwidth_only \
  --metric avg_quality

# 6. 手動設定トポロジ環境（manual）で比較
python aco_moo_routing/analysis/compare_methods.py \
  --generations 1000 --ants 10 \
  --methods basic_aco_no_heuristic basic_aco_with_heuristic previous proposed \
  --environments manual \
  --opt-type bandwidth_only \
  --metric is_optimal
```

**比較可能な指標（--metric オプション）**:

- `is_optimal`: 最適解到達率 [%]
- `is_unique_optimal`: ユニーク最適解到達率 [%]
- `avg_quality`: 平均品質スコア (0.0 ~ 1.0)
- `max_quality`: 最大品質スコアの平均 (0.0 ~ 1.0)

### その他の可視化スクリプト

- **単一ログの可視化**（最適率/品質スコアなど）

  ```bash
  python aco_moo_routing/analysis/analyze_optimal_percentage.py \
    --csv aco_moo_routing/results/proposed/static/bandwidth_only/ant_solution_log.csv \
    --generations 1000 --ants 10 --metric optimal_rate
  ```

- **従来 vs 提案の比較**

  ```bash
  python aco_moo_routing/analysis/compare_conventional_vs_proposed.py \
    --generations 1000 --ants 10 \
    --environments static \
    --opt-type bandwidth_only \
    --metric is_optimal
  ```

- **遅延制約の値ごとの比較**（提案手法内で any/unique）
  ```bash
  python aco_moo_routing/analysis/compare_delay_constraint.py \
    --generations 1000 --ants 10 \
    --constraints 5 10 15 \
    --subdir-template "delay_constraint_{c}ms" \
    --opt-type delay_constraint
  ```

## ログの解釈

- `ant_solution_log.csv`  
  `generation, ant_id, bandwidth, delay, hops, is_optimal, optimal_index, is_unique_optimal, quality_score`
- `generation_stats.csv`  
  世代ごとの平均/最大/最小/標準偏差（bandwidth, delay, hops, quality_score）、到達アリ数、optimal_count、unique_optimal_count

## よくある確認ポイント

- `generations` と `num_ants` は実行設定と分析コマンドで一致させる
- 遅延制約 ON のときは出力パスが `delay_constraint` 系になる点に注意
