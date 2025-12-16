"""
統合テストスクリプト

4手法×4環境の全16組み合わせでシミュレーションを実行し、エラーがないか確認します。

【テスト対象】
- 4つの手法：
  1. basic_aco_no_heuristic (従来手法1：基本ACO、β=0)
  2. basic_aco_with_heuristic (従来手法2：基本ACO、β=1)
  3. previous (先行研究：エッジベースの学習)
  4. proposed (提案手法：ノードベースの学習)

- 4つの環境：
  1. manual (手動設定トポロジ：最適経路を100Mbpsに設定)
  2. static (ランダムトポロジ：全リンクランダム)
  3. node_switching (ノード変動：スタートノード切り替え)
  4. bandwidth_fluctuation (帯域変動：AR1モデル)
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
aco_moo_root = Path(__file__).parent.parent
sys.path.insert(0, str(aco_moo_root / "src"))
sys.path.insert(0, str(aco_moo_root))

import yaml

# run_experimentをインポート（相対インポート）
from experiments.run_experiment import main as run_experiment_main


def create_test_config(
    method: str,
    environment: str,
    generations: int = 10,
    simulations: int = 1,
    num_nodes: int = 20,  # テスト用に小規模なグラフを使用
    num_edges: int = 3,  # テスト用に少ないエッジ数
) -> dict:
    """
    テスト用の設定を生成

    Args:
        method: ACO手法 ("basic_aco_no_heuristic", "basic_aco_with_heuristic", "previous", "proposed")
        environment: 環境タイプ ("manual", "static", "node_switching", "bandwidth_fluctuation")
        generations: 世代数（テスト用に短く設定）
        simulations: シミュレーション回数（テスト用に1回）

    Returns:
        設定辞書
    """
    config = {
        "experiment": {
            "name": f"test_{method}_{environment}",
            "generations": generations,
            "num_ants": 10,
            "simulations": simulations,
            "target_objectives": ["bandwidth"],
            "delay_constraint": {"enabled": False, "max_delay": 10.0},
            "start_switching": {
                "enabled": environment == "node_switching",
                "switch_interval": 100,
                "start_nodes": [],
            },
        },
        "graph": {
            "num_nodes": num_nodes,  # テスト用に小規模
            "num_edges": num_edges,  # テスト用に少ないエッジ数
            "graph_type": "manual" if environment == "manual" else "barabasi_albert",
            "bandwidth_range": [10, 100],
            "delay_range": [1, 10],
            "fluctuation": {
                "enabled": environment == "bandwidth_fluctuation",
                "model": "ar1",
                "target_method": "hub",
                "target_percentage": 0.1,
                "update_interval": 1,
            },
        },
        "aco": {
            "method": method,
            "q_factor": 1.0,
            "alpha": 1.0,
            "beta_bandwidth": 0 if method == "basic_aco_no_heuristic" else 1.0,
            "beta_delay": 1.0,
            "beta_hops": 1.0,
            "epsilon": 0.1,
            "evaporation_rate": 0.02,
            "min_pheromone": 100,
            "max_pheromone": 1000000000,
            "ttl": 100,
            "learning": {
                "bkb_window_size": 100,
                "bonus_factor": 2.0,
                "penalty_factor": 0.5,
                "volatilization_mode": 3,
                "bkb_evaporation_rate": 0.001,
                "delay_tolerance": 5.0,
            },
        },
        "pareto": {
            "enabled": True,
            "max_labels_per_node": 1000,
            "reference_point": [0, 1000, 200],
        },
        "evaluation": {
            "metrics": [
                "pareto_discovery_rate",
                "dominance_rate",
                "hypervolume",
                "convergence_rate",
            ]
        },
        "output": {
            "save_results": True,
            "save_interval": 100,
            "save_graphs": False,  # テスト時はグラフを保存しない
            "results_dir": "results",
            "log_level": "INFO",
        },
    }
    return config


def run_test_with_config(method: str, environment: str, config: dict) -> bool:
    """
    1つの組み合わせをテスト（設定辞書を直接受け取る）

    Args:
        method: ACO手法
        environment: 環境タイプ
        config: 設定辞書

    Returns:
        成功したらTrue、失敗したらFalse
    """
    print(f"\n{'='*80}")
    print(f"Testing: Method={method}, Environment={environment}")
    print(f"{'='*80}")

    try:
        # 実験を実行（設定辞書を直接渡す）
        run_experiment_main(config_dict=config)

        print(f"✅ Success: Method={method}, Environment={environment}")
        return True

    except Exception as e:
        print(f"❌ Error: Method={method}, Environment={environment}")
        print(f"   Error message: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """全組み合わせをテスト"""
    import argparse

    parser = argparse.ArgumentParser(description="Integration test for all method/environment combinations")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: only test first method and first environment",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations for testing (default: 10)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=20,
        help="Number of nodes for testing (default: 20)",
    )
    args = parser.parse_args()

    methods = [
        "basic_aco_no_heuristic",
        "basic_aco_with_heuristic",
        "previous",
        "proposed",
    ]
    environments = [
        "manual",
        "static",
        "node_switching",
        "bandwidth_fluctuation",
    ]

    # クイックテストモード：最初の組み合わせのみ
    if args.quick:
        methods = [methods[0]]
        environments = [environments[0]]
        print("⚠️  Quick test mode: Testing only first combination")

    results = {}
    total_tests = len(methods) * len(environments)
    passed_tests = 0
    failed_tests = 0

    print(f"\n{'='*80}")
    print(f"Integration Test: {total_tests} combinations")
    print(f"  Generations: {args.generations}")
    print(f"  Num nodes: {args.num_nodes}")
    print(f"{'='*80}")

    for method in methods:
        for environment in environments:
            key = f"{method}_{environment}"
            # テスト用設定を生成（引数で指定されたパラメータを使用）
            config = create_test_config(
                method,
                environment,
                generations=args.generations,
                simulations=1,
                num_nodes=args.num_nodes,
                num_edges=3,
            )
            success = run_test_with_config(method, environment, config)
            results[key] = success
            if success:
                passed_tests += 1
            else:
                failed_tests += 1

    # 結果サマリー
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"\nDetailed Results:")
    for key, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {key}")

    if failed_tests == 0:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
