# ACO Multi-Objective Routing

多目的最適化（帯域・遅延・ホップ数）に対応した Ant Colony Optimization (ACO)ルーティング実装

## 📁 ディレクトリ構造

```
aco_moo_routing/
├── config/
│   └── config.yaml             # 全ての設定を一元管理
├── experiments/
│   └── run_experiment.py       # 実験実行スクリプト
├── results/                    # 結果出力先
└── src/
    └── aco_routing/            # メインパッケージ
        ├── core/               # 状態を持つ基本オブジェクト
        ├── modules/            # ロジック・機能コンポーネント
        ├── algorithms/         # アルゴリズム実装
        └── utils/              # ユーティリティ
```

## 🚀 実行方法

### 1. Conda 環境のセットアップ

```bash
# プロジェクトルートディレクトリで実行
cd /path/to/Ant-Colony-Optimization
conda env create -f conda-env.yaml
conda activate aco-env
```

または、pip で依存パッケージをインストール：

```bash
cd aco_moo_routing
pip install -r requirements.txt
```

### 2. 設定ファイルの編集

`config/config.yaml`を編集し、実験設定を調整します。

- **Step 1**: 帯域 vs ホップ数

  ```yaml
  target_objectives: ["bandwidth", "hops"]
  ```

- **Step 2**: 帯域 vs 遅延（デフォルト）

  ```yaml
  target_objectives: ["bandwidth", "delay"]
  ```

- **Step 3**: 帯域 vs 遅延 vs ホップ数
  ```yaml
  target_objectives: ["bandwidth", "delay", "hops"]
  ```

### 3. 実験の実行

```bash
cd experiments
python run_experiment.py
```

## 📊 評価指標

- **Pareto Discovery Rate**: ACO が真のパレート最適解を発見した割合
- **Dominance Rate**: ACO 解が真のパレート解に支配されない割合
- **Hypervolume**: パレートフロンティアが覆う超体積
- **Convergence Rate**: 世代ごとの収束率の推移

## 📈 結果の可視化

### 4手法の比較（`compare_methods.py`）

4手法（Basic ACO w/o Heuristic, Basic ACO w/ Heuristic, Previous Method, Proposed Method）を任意に選択して比較できます。

**基本的な使用例**:

```bash
# プロジェクトルートから実行

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

**利用可能な手法**:
- `basic_aco_no_heuristic`: Basic ACO w/o Heuristic (β=0)
- `basic_aco_with_heuristic`: Basic ACO w/ Heuristic (β=1)
- `previous`: Previous Method (Edge-based learning)
- `proposed`: Proposed Method (Node-based learning)

**利用可能な環境**:
- `manual`: 手動設定トポロジ（最適経路を100Mbpsに設定）
- `static`: 静的ランダムグラフ
- `node_switching`: コンテンツ要求ノード変動
- `bandwidth_fluctuation`: 帯域変動

**出力**:
- ファイル名: `comparison_{methods}_{environments}_{metric}.eps` と `.svg`
- 保存先: `results/analysis/` ディレクトリ（デフォルト）

### その他の可視化スクリプト

- `analyze_optimal_percentage.py`: 単一ログの可視化（最適率/品質スコアなど）
- `compare_conventional_vs_proposed.py`: 従来手法 vs 提案手法の比較
- `compare_delay_constraint.py`: 遅延制約の値ごとの比較

## 🎯 主要な機能

### 1. ノードの自律学習（BKB/BLD/BKH）

- 各ノードがリングバッファで過去の功績を記憶
- BKB (Best Known Bottleneck): 最大帯域
- BLD (Best Known Lowest Delay): 最小遅延
- BKH (Best Known Hops): 最小ホップ数

### 2. 功績ボーナス（Achievement Bonus）

- ノードの記憶値を更新した場合にフェロモンボーナスを付与
- 分散型の判断（各ノードでローカルに判定）

### 3. 動的帯域変動

- AR(1)モデルによる帯域変動
- 遅延も帯域と連動して変動（物理的整合性）

### 4. パレートフロンティア計算

- 多目的ラベリング法による厳密解の計算
- ACO の評価における「正解データ」として使用

## 📝 設定パラメータ（主要なもの）

```yaml
aco:
  alpha: 1.0 # フェロモンの影響度
  beta_bandwidth: 1.0 # 帯域のヒューリスティック重要度
  beta_delay: 1.0 # 遅延のヒューリスティック重要度
  epsilon: 0.1 # ε-Greedy法のランダム確率
  evaporation_rate: 0.02 # フェロモン揮発率

  learning:
    bkb_window_size: 10 # リングバッファサイズ
    bonus_factor: 1.5 # 功績ボーナス係数
    penalty_factor: 0.5 # BKBベースペナルティ係数
```

## 🔬 実装の特徴

- **オブジェクト指向設計**: 拡張性と可読性を重視
- **モジュール性**: 各コンポーネントが独立し、循環参照なし
- **型ヒント**: 全ての関数に型ヒントを付与
- **設定の一元管理**: YAML ファイルで全パラメータを管理

## 📖 参考文献

- Modified Dijkstra 法によるボトルネック最大化
- 多目的最適化におけるパレート支配
- Ant Colony Optimization (ACO)

## 🧪 テストの実行

```bash
# すべてのテストを実行
pytest tests/

# カバレッジ付きでテストを実行
pytest tests/ --cov=src/aco_routing --cov-report=html

# 特定のテストファイルのみ実行
pytest tests/test_core.py -v
```

## 🔍 コード品質チェック

### Linter（Flake8）

```bash
flake8 src/aco_routing tests/
```

### Formatter（Black）

```bash
# コードをフォーマット
black src/aco_routing tests/

# フォーマットチェックのみ（変更なし）
black --check src/aco_routing tests/
```

### 型チェック（Mypy）

```bash
mypy src/aco_routing
```

### 全チェックを一括実行

```bash
# フォーマット → Linter → 型チェック → テスト
black src/aco_routing tests/ && \
flake8 src/aco_routing tests/ && \
mypy src/aco_routing && \
pytest tests/ -v
```

## 📝 既存実装との互換性

### ボトルネック最大化のみの場合

新実装（`aco_moo_routing`）は、`target_objectives: ["bandwidth"]`または`["bandwidth", "hops"]`と設定した場合、既存実装（`src/aco_main_bkb_available_bandwidth.py`）と同じアルゴリズムで動作します。

**ヒューリスティック計算**:

- Step 1 (bandwidth vs hops): `η = bandwidth^β` （既存実装と同じ）
- Step 2 (bandwidth vs delay): `η = (bandwidth^β_B) / (delay^β_D)`
- Step 3 (3 目的): `η = (bandwidth^β_B) / (delay^β_D)`

## ライセンス

MIT License
