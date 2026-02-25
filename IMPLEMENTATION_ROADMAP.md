# 4 手法 ×4 環境 シミュレーション実装ロードマップ

## 概要

論文の評価実験に必要な、4 つの手法を 4 つの環境で実行できるようにするための実装計画です。

## 4 つの手法

1. **従来手法 1：基本 ACO（Basic ACO w/o Heuristic）**

   - β=0、ヒューリスティック情報（帯域幅）を使用しない
   - フェロモンのみで経路選択：`p_ij = [τ_ij]^α / Σ_l [τ_il]^α`
   - フェロモン更新：`Δτ = Q * B_k`（ボトルネック帯域に比例）

2. **従来手法 2：ヒューリスティック ACO（Basic ACO w/ Heuristic）**

   - β=1、ヒューリスティック情報（帯域幅）を使用
   - フェロモンと帯域幅で経路選択：`p_ij = [τ_ij]^α · [w_ij]^β / Σ_l [τ_il]^α · [w_il]^β`
   - フェロモン更新：`Δτ = Q * B_k`（ボトルネック帯域に比例）

3. **先行研究（Previous Method）**

   - エッジベースの学習：各エッジが`w_ij^min`と`w_ij^max`を保持
   - フェロモン揮発：`rate_ij = V × (w_ij - w_ij^min) / max(1, w_ij^max - w_ij^min)`
   - フェロモン更新：`local_min_bandwidth`と`local_max_bandwidth`を更新

4. **提案手法（Proposed Method）**
   - ノードベースの学習：各ノードが`B_v^ref`（参照値）を保持
   - 既に実装済み（`ACOSolver`）

## 4 つの環境

1. **Environment 1：手動設定トポロジ（Manual Topology）**

   - 特定の 1 経路のみ全リンクを 100Mbps に設定
   - それ以外のリンクは 10-90Mbps のランダム
   - 最適解のボトルネック帯域 = 100Mbps

2. **Environment 2：ランダムトポロジ（Static Random）**

   - 全リンクの帯域幅を 10-100Mbps の範囲でランダム設定
   - 既存の`static`環境で対応可能

3. **Environment 3：コンテンツ要求ノード変動（Node Switching）**

   - 100 世代ごとにスタートノードを切り替え
   - 以前のスタートノードをキャッシュ（ゴール）として追加
   - 既存の`node_switching`環境で対応可能

4. **Environment 4：帯域変動（Bandwidth Fluctuation）**
   - ハブノード（次数上位 10%）に接続されたリンクの帯域を変動
   - AR(1)モデルを使用
   - 既存の`bandwidth_fluctuation`環境で対応可能

## 実装タスク

### Phase 1: 従来手法 1・2 の実装

#### タスク 1.1: `ConventionalACOSolver`の修正

- **ファイル**: `aco_moo_routing/src/aco_routing/algorithms/conventional_aco_solver.py`
- **変更内容**:
  - `_probabilistic_selection`メソッドで`beta_bandwidth`を考慮
  - `beta_bandwidth=0`の場合はヒューリスティックを使用しない（従来手法 1）
  - `beta_bandwidth=1`の場合はヒューリスティックを使用（従来手法 2）
- **状態**: 未実装

#### タスク 1.2: 手法選択の拡張

- **ファイル**: `aco_moo_routing/config/config.yaml`, `aco_moo_routing/experiments/run_experiment.py`
- **変更内容**:
  - `aco.method`に`"basic_aco_no_heuristic"`と`"basic_aco_with_heuristic"`を追加
  - `run_experiment.py`で新しい手法を選択できるようにする
- **状態**: 未実装

### Phase 2: 先行研究（Previous Method）の実装

#### タスク 2.1: `PreviousMethodACOSolver`の作成

- **ファイル**: `aco_moo_routing/src/aco_routing/algorithms/previous_method_aco_solver.py`（新規作成）
- **実装内容**:
  - エッジベースの学習：`local_min_bandwidth`と`local_max_bandwidth`を更新
  - フェロモン揮発：`rate_ij = V × (w_ij - w_ij^min) / max(1, w_ij^max - w_ij^min)`
  - フェロモン更新：通過したパスのボトルネック帯域に基づいて`local_min/max_bandwidth`を更新
  - 参考実装：`src/aco_sim_caching_model_eval.py`, `src/pheromone_update.py`
- **状態**: 未実装

#### タスク 2.2: 手法選択への追加

- **ファイル**: `aco_moo_routing/config/config.yaml`, `aco_moo_routing/experiments/run_experiment.py`
- **変更内容**:
  - `aco.method`に`"previous"`を追加
  - `run_experiment.py`で`PreviousMethodACOSolver`を選択できるようにする
- **状態**: 未実装

### Phase 3: Environment 1（手動設定トポロジ）の実装

#### タスク 3.1: 手動設定トポロジ生成機能の追加

- **ファイル**: `aco_moo_routing/src/aco_routing/core/graph.py`
- **実装内容**:
  - `create_manual_topology`メソッドを追加
  - 特定の 1 経路のみ全リンクを 100Mbps に設定
  - それ以外のリンクは 10-90Mbps のランダム
  - 最適解のボトルネック帯域 = 100Mbps を保証
- **状態**: 未実装

#### タスク 3.2: 環境設定の拡張

- **ファイル**: `aco_moo_routing/config/config.yaml`
- **変更内容**:
  - `graph.graph_type`に`"manual"`を追加
  - 手動設定トポロジ用のパラメータを追加（最適経路の指定方法など）
- **状態**: 未実装

#### タスク 3.3: 環境判定の更新

- **ファイル**: `aco_moo_routing/experiments/run_experiment.py`
- **変更内容**:
  - `determine_environment`関数で`"manual"`環境を判定できるようにする
- **状態**: 未実装

### Phase 4: 統合テストと検証

#### タスク 4.1: 全組み合わせのテスト

- **内容**: 4 手法 ×4 環境の全 16 組み合わせでシミュレーションを実行
- **確認項目**:
  - 各手法が正しく動作するか
  - 各環境が正しく設定されるか
  - ログファイルが正しく生成されるか
- **状態**: 未実装

#### タスク 4.2: 結果の可視化

- **内容**: 既存の可視化スクリプトが新しい手法・環境に対応しているか確認
- **確認項目**:
  - `compare_conventional_vs_proposed.py`が新しい手法に対応しているか
  - グラフ生成が正しく動作するか
- **状態**: 未実装

## 実装の優先順位

1. **最優先**: Phase 1（従来手法 1・2 の実装）

   - 既存の`ConventionalACOSolver`を修正するだけなので比較的簡単
   - 論文の比較実験に必須

2. **次優先**: Phase 2（先行研究の実装）

   - エッジベースの学習は既に参考実装がある
   - 論文の比較実験に必須

3. **中優先**: Phase 3（Environment 1 の実装）

   - 新しい環境の実装が必要
   - 論文の評価シナリオ 1 に必要

4. **低優先**: Phase 4（統合テスト）
   - 実装完了後の検証

## 実装時の注意点

1. **後方互換性**: 既存の`conventional`手法との互換性を保つ
2. **設定ファイル**: `config.yaml`の構造を拡張する際は、既存の設定が壊れないようにする
3. **ログ形式**: 既存のログ形式（`ant_solution_log.csv`など）との互換性を保つ
4. **テスト**: 各手法が既存の環境（static, node_switching, bandwidth_fluctuation）で動作することを確認

## 参考実装

- **エッジベースの学習**: `src/aco_sim_caching_model_eval.py`, `src/pheromone_update.py`
- **フェロモン揮発**: `src/aco_main_csv_networkx.py`の`volatilize_by_width`関数
- **手動設定トポロジ**: 新規実装が必要（論文の記述を参考）

## 完了条件

- [x] 従来手法 1（Basic ACO w/o Heuristic）が実装され、4 環境で動作する
- [x] 従来手法 2（Basic ACO w/ Heuristic）が実装され、4 環境で動作する
- [x] 先行研究（Previous Method）が実装され、4 環境で動作する
- [x] 提案手法（Proposed Method）が 4 環境で動作する（既存）
- [x] Environment 1（手動設定トポロジ）が実装され、4 手法で動作する
- [x] 全 16 組み合わせでシミュレーションが実行できる
- [x] 結果が正しくログに記録される
- [x] 統合テストスクリプトが作成され、全組み合わせでテストが成功する

## 実装完了状況

✅ **Phase 1**: 従来手法1・2の実装完了
✅ **Phase 2**: 先行研究（Previous Method）の実装完了
✅ **Phase 3**: Environment 1（手動設定トポロジ）の実装完了
✅ **Phase 4**: 統合テスト完了（全16組み合わせでテスト成功）

## テスト結果

統合テストスクリプト（`aco_moo_routing/tests/test_all_methods_environments.py`）で全16組み合わせをテストし、すべて成功しました。

### テスト実行コマンド

```bash
# クイックテスト（最初の組み合わせのみ）
python aco_moo_routing/tests/test_all_methods_environments.py --quick

# 全16組み合わせをテスト（小規模グラフ、短い世代数）
python aco_moo_routing/tests/test_all_methods_environments.py --generations 5 --num-nodes 15

# 全16組み合わせをテスト（デフォルト設定）
python aco_moo_routing/tests/test_all_methods_environments.py
```

### テスト結果（全16組み合わせ）

✅ すべてのテストが成功しました：
- basic_aco_no_heuristic × 4環境: ✅
- basic_aco_with_heuristic × 4環境: ✅
- previous × 4環境: ✅
- proposed × 4環境: ✅
