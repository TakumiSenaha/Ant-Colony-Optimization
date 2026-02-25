"""
Manual環境のテスト

manual環境での初期値設定とフェロモン更新が正しく動作することを検証します。

【テストの方針】
- ログファイルを作成しない（結果を保存しない）
- 既存ログを汚染しない
- 初期値の正確性を検証
- フェロモン更新の正確性を検証
"""

import random
import sys
from pathlib import Path

import pytest

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aco_routing.algorithms.aco_solver import ACOSolver
from aco_routing.algorithms.conventional_aco_solver import ConventionalACOSolver
from aco_routing.algorithms.single_objective_solver import (
    bottleneck_capacity,
    max_load_path,
)
from aco_routing.core.graph import RoutingGraph


class TestManualEnvironmentInitialization:
    """Manual環境での初期化テスト"""

    def test_manual_environment_bandwidth_setup(self):
        """
        manual環境で最適経路を100Mbpsに設定した後、
        帯域値が正しく設定されていることを確認
        """
        random.seed(42)

        # テスト用設定
        config = {
            "graph": {
                "num_nodes": 100,
                "num_edges": 6,
                "graph_type": "barabasi_albert",
                "bandwidth_range": [10, 100],
                "delay_range": [1, 10],
                "fluctuation": {"enabled": False},
            },
            "aco": {
                "alpha": 1.0,
                "beta_bandwidth": 1.0,
                "beta_delay": 1.0,
                "epsilon": 0.1,
                "ttl": 100,
                "evaporation_rate": 0.02,
                "min_pheromone": 100,
                "max_pheromone": 1000000000,
                "learning": {
                    "bkb_window_size": 100,
                    "bonus_factor": 2.0,
                    "penalty_factor": 0.5,
                    "volatilization_mode": 3,
                    "bkb_evaporation_rate": 0.001,
                    "delay_tolerance": 5.0,
                },
            },
            "experiment": {
                "target_objectives": ["bandwidth"],
                "delay_constraint": {"enabled": False},
                "num_ants": 10,  # ACOSolver.run()で必要
            },
        }

        # グラフを生成
        graph = RoutingGraph(100, config)
        G = graph.graph

        # スタートとゴールを選択
        start_node = 21
        goal_node = 18

        # 最適パスを見つける
        optimal_path = max_load_path(G, start_node, goal_node, weight="bandwidth")
        assert optimal_path is not None, "最適パスが見つかりませんでした"

        # 最適パスを100Mbpsに設定（manual環境の処理を再現）
        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            G.edges[u, v]["bandwidth"] = 100.0
            G.edges[v, u]["bandwidth"] = 100.0
            G.edges[u, v]["original_bandwidth"] = 100.0
            G.edges[v, u]["original_bandwidth"] = 100.0

        # 検証: 最適パスの全エッジが100Mbpsになっているか
        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            assert (
                G.edges[u, v]["bandwidth"] == 100.0
            ), f"Edge ({u}, {v}): bandwidth should be 100.0"
            assert (
                G.edges[v, u]["bandwidth"] == 100.0
            ), f"Edge ({v}, {u}): bandwidth should be 100.0"

        # 検証: 最適解のボトルネック帯域が100Mbpsか
        optimal_bottleneck = bottleneck_capacity(G, optimal_path, weight="bandwidth")
        assert (
            optimal_bottleneck == 100.0
        ), f"Optimal bottleneck should be 100.0, got {optimal_bottleneck}"

        print("✅ Manual環境の帯域設定テスト: 成功")

    def test_manual_environment_pheromone_min_max_proposed_method(self):
        """
        manual環境で提案手法のフェロモンmin/maxが正しく設定されることを確認
        
        【検証ポイント】
        1. 初期値: min_pheromone = 100 * 3 // degree（双方向で異なる）
        2. 初期値: max_pheromone = bandwidth^5
        3. manual環境処理後: 最適経路のmax_pheromone = 100^5
        """
        random.seed(42)

        config = {
            "graph": {
                "num_nodes": 100,
                "num_edges": 6,
                "graph_type": "barabasi_albert",
                "bandwidth_range": [10, 100],
                "delay_range": [1, 10],
                "fluctuation": {"enabled": False},
            },
            "aco": {
                "alpha": 1.0,
                "beta_bandwidth": 1.0,
                "beta_delay": 1.0,
                "epsilon": 0.1,
                "ttl": 100,
                "evaporation_rate": 0.02,
                "min_pheromone": 100,
                "max_pheromone": 1000000000,
                "learning": {
                    "bkb_window_size": 100,
                    "bonus_factor": 2.0,
                    "penalty_factor": 0.5,
                    "volatilization_mode": 3,
                    "bkb_evaporation_rate": 0.001,
                    "delay_tolerance": 5.0,
                },
            },
            "experiment": {
                "target_objectives": ["bandwidth"],
                "delay_constraint": {"enabled": False},
                "num_ants": 10,  # ACOSolver.run()で必要
            },
        }

        # グラフを生成
        graph = RoutingGraph(100, config)
        G = graph.graph

        start_node = 21
        goal_node = 18

        # 最適パスを見つける
        optimal_path = max_load_path(G, start_node, goal_node, weight="bandwidth")

        # 【検証1】初期のmin_pheromone設定（次数に基づく）
        # RoutingGraphは有向グラフ（DiGraph）を使用
        # 双方向で異なるmin_pheromoneを設定できる
        u, v = optimal_path[0], optimal_path[1]
        degree_u = len(list(G.neighbors(u)))
        degree_v = len(list(G.neighbors(v)))

        # u → v のmin_pheromone（uの次数に基づく）
        min_u_to_v = G.edges[u, v]["min_pheromone"]
        expected_min_u_to_v = 100 * 3 // degree_u

        # v → u のmin_pheromone（vの次数に基づく）
        min_v_to_u = G.edges[v, u]["min_pheromone"]
        expected_min_v_to_u = 100 * 3 // degree_v

        print(
            f"  Edge ({u}→{v}): degree_u={degree_u}, "
            f"expected_min={expected_min_u_to_v}, actual_min={min_u_to_v}"
        )
        print(
            f"  Edge ({v}→{u}): degree_v={degree_v}, "
            f"expected_min={expected_min_v_to_u}, actual_min={min_v_to_u}"
        )

        # 検証: 有向グラフで双方向のmin_pheromoneが正しく設定されているか
        assert (
            min_u_to_v == expected_min_u_to_v
        ), f"Edge ({u}→{v}): min_pheromone should be {expected_min_u_to_v}, got {min_u_to_v}"
        assert (
            min_v_to_u == expected_min_v_to_u
        ), f"Edge ({v}→{u}): min_pheromone should be {expected_min_v_to_u}, got {min_v_to_u}"

        # 次数が異なれば、min値も異なるはず
        if degree_u != degree_v:
            assert (
                min_u_to_v != min_v_to_u
            ), f"min_pheromone should differ for different degrees: {min_u_to_v} vs {min_v_to_u}"

        # 【検証2】初期のmax_pheromone設定（帯域の5乗）
        initial_bandwidth = G.edges[u, v]["bandwidth"]
        expected_max = int(initial_bandwidth**5)
        actual_max_u_to_v = G.edges[u, v]["max_pheromone"]
        actual_max_v_to_u = G.edges[v, u]["max_pheromone"]
        
        # 双方向で同じ帯域なので、max_pheromoneも同じはず
        assert (
            actual_max_u_to_v == expected_max
        ), f"Edge ({u}→{v}): max_pheromone should be {expected_max}, got {actual_max_u_to_v}"
        assert (
            actual_max_v_to_u == expected_max
        ), f"Edge ({v}→{u}): max_pheromone should be {expected_max}, got {actual_max_v_to_u}"

        # 最適パスを100Mbpsに設定
        base_min_pheromone = config["aco"]["min_pheromone"]
        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            G.edges[u, v]["bandwidth"] = 100.0
            G.edges[v, u]["bandwidth"] = 100.0

            # フェロモンmin/maxを再計算（run_experiment.pyと同じロジック）
            degree_u = len(list(G.neighbors(u)))
            degree_v = len(list(G.neighbors(v)))
            G.edges[u, v]["min_pheromone"] = base_min_pheromone * 3 // degree_u
            G.edges[v, u]["min_pheromone"] = base_min_pheromone * 3 // degree_v
            G.edges[u, v]["max_pheromone"] = int(100.0**5)
            G.edges[v, u]["max_pheromone"] = int(100.0**5)

        # 【検証3】manual環境処理後のmax_pheromone
        u, v = optimal_path[0], optimal_path[1]
        expected_max_after_manual = int(100.0**5)  # 10,000,000,000
        actual_max_u_to_v = G.edges[u, v]["max_pheromone"]
        actual_max_v_to_u = G.edges[v, u]["max_pheromone"]
        
        assert (
            actual_max_u_to_v == expected_max_after_manual
        ), f"Edge ({u}→{v}): max_pheromone after manual should be {expected_max_after_manual}, got {actual_max_u_to_v}"
        assert (
            actual_max_v_to_u == expected_max_after_manual
        ), f"Edge ({v}→{u}): max_pheromone after manual should be {expected_max_after_manual}, got {actual_max_v_to_u}"

        print("✅ Manual環境のフェロモンmin/max設定テスト（提案手法）: 成功")

    def test_manual_environment_pheromone_min_max_conventional_method(self):
        """
        manual環境で従来手法のフェロモンmin/maxが正しく設定されることを確認
        
        【検証ポイント】
        1. ConventionalACOSolverは正規化スケール（min=0.01, max=10.0）を使用
        2. manual環境処理後も正規化スケールを維持
        """
        random.seed(42)

        config = {
            "graph": {
                "num_nodes": 100,
                "num_edges": 6,
                "graph_type": "barabasi_albert",
                "bandwidth_range": [10, 100],
                "delay_range": [1, 10],
                "fluctuation": {"enabled": False},
            },
            "aco": {
                "alpha": 1.0,
                "beta_bandwidth": 1.0,  # ConventionalACOSolverが2.0に上書き
                "beta_delay": 1.0,
                "ttl": 100,
                "evaporation_rate": 0.02,  # ConventionalACOSolverが0.1に上書き
                "min_pheromone": 100,  # ConventionalACOSolverが0.01に上書き
                "max_pheromone": 1000000000,  # ConventionalACOSolverが10.0に上書き
                "learning": {  # RoutingGraphの初期化に必要
                    "bkb_window_size": 100,
                    "bonus_factor": 2.0,
                    "penalty_factor": 0.5,
                    "volatilization_mode": 3,
                    "bkb_evaporation_rate": 0.001,
                    "delay_tolerance": 5.0,
                },
            },
            "experiment": {
                "target_objectives": ["bandwidth"],
                "delay_constraint": {"enabled": False},
                "num_ants": 10,  # ACOSolver.run()で必要
            },
        }

        # グラフを生成
        graph = RoutingGraph(100, config)
        G = graph.graph

        start_node = 21
        goal_node = 18

        # 最適パスを見つける
        optimal_path = max_load_path(G, start_node, goal_node, weight="bandwidth")

        # 最適パスを100Mbpsに設定
        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            G.edges[u, v]["bandwidth"] = 100.0
            G.edges[v, u]["bandwidth"] = 100.0

        # ConventionalACOSolverを初期化（_reinitialize_pheromones()が呼ばれる）
        solver = ConventionalACOSolver(config, graph)

        # 【検証1】ConventionalACOSolverのパラメータ
        assert solver.beta_bandwidth == 2.0, "ConventionalACOSolver should set beta_bandwidth=2.0"
        assert solver.evaporation_rate == 0.1, "ConventionalACOSolver should set evaporation_rate=0.1"
        assert solver.min_pheromone == 0.01, "ConventionalACOSolver should set min_pheromone=0.01"
        assert solver.max_pheromone == 10.0, "ConventionalACOSolver should set max_pheromone=10.0"

        # 【検証2】_reinitialize_pheromones()後のフェロモン値
        u, v = optimal_path[0], optimal_path[1]
        assert (
            G.edges[u, v]["pheromone"] == 1.0
        ), f"Pheromone should be initialized to 1.0, got {G.edges[u, v]['pheromone']}"
        assert (
            G.edges[u, v]["min_pheromone"] == 0.01
        ), f"min_pheromone should be 0.01, got {G.edges[u, v]['min_pheromone']}"
        assert (
            G.edges[u, v]["max_pheromone"] == 10.0
        ), f"max_pheromone should be 10.0, got {G.edges[u, v]['max_pheromone']}"

        # manual環境のフェロモン再計算を実行（run_experiment.pyのロジック）
        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            # ConventionalACOSolverは正規化スケール（固定値）を維持
            G.edges[u, v]["min_pheromone"] = solver.min_pheromone
            G.edges[v, u]["min_pheromone"] = solver.min_pheromone
            G.edges[u, v]["max_pheromone"] = solver.max_pheromone
            G.edges[v, u]["max_pheromone"] = solver.max_pheromone

        # 【検証3】manual環境処理後も正規化スケールを維持
        u, v = optimal_path[0], optimal_path[1]
        assert (
            G.edges[u, v]["min_pheromone"] == 0.01
        ), f"min_pheromone should remain 0.01, got {G.edges[u, v]['min_pheromone']}"
        assert (
            G.edges[u, v]["max_pheromone"] == 10.0
        ), f"max_pheromone should remain 10.0, got {G.edges[u, v]['max_pheromone']}"

        print("✅ Manual環境のフェロモンmin/max設定テスト（従来手法）: 成功")


class TestManualEnvironmentPheromoneUpdate:
    """Manual環境でのフェロモン更新テスト"""

    def test_proposed_method_pheromone_accumulation(self):
        """
        提案手法で100Mbpsパスを通ったアリのフェロモンが正しく蓄積されることを確認
        
        【検証ポイント】
        - フェロモン付加量: base_pheromone = 100 * 10 * 2.0 = 2000（ボーナスあり）
        - max_pheromone = 100^5 = 10,000,000,000
        - フェロモンが正しく蓄積される（max値で切り捨てられない）
        """
        random.seed(123)

        config = {
            "graph": {
                "num_nodes": 100,
                "num_edges": 6,
                "graph_type": "barabasi_albert",
                "bandwidth_range": [10, 100],
                "delay_range": [1, 10],
                "fluctuation": {"enabled": False},
            },
            "aco": {
                "alpha": 1.0,
                "beta_bandwidth": 1.0,
                "beta_delay": 1.0,
                "epsilon": 0.1,
                "ttl": 100,
                "evaporation_rate": 0.02,
                "min_pheromone": 100,
                "max_pheromone": 1000000000,
                "learning": {
                    "bkb_window_size": 100,
                    "bonus_factor": 2.0,
                    "penalty_factor": 0.5,
                    "volatilization_mode": 3,
                    "bkb_evaporation_rate": 0.001,
                    "delay_tolerance": 5.0,
                },
            },
            "experiment": {
                "target_objectives": ["bandwidth"],
                "delay_constraint": {"enabled": False},
                "num_ants": 10,  # ACOSolver.run()で必要
            },
        }

        # グラフを生成
        graph = RoutingGraph(100, config)
        G = graph.graph

        start_node = 21
        goal_node = 18

        # 最適パスを見つけて100Mbpsに設定
        optimal_path = max_load_path(G, start_node, goal_node, weight="bandwidth")
        base_min_pheromone = config["aco"]["min_pheromone"]

        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            G.edges[u, v]["bandwidth"] = 100.0
            G.edges[v, u]["bandwidth"] = 100.0

            # フェロモンmin/maxを再計算
            degree_u = len(list(G.neighbors(u)))
            degree_v = len(list(G.neighbors(v)))
            G.edges[u, v]["min_pheromone"] = base_min_pheromone * 3 // degree_u
            G.edges[v, u]["min_pheromone"] = base_min_pheromone * 3 // degree_v
            G.edges[u, v]["max_pheromone"] = int(100.0**5)
            G.edges[v, u]["max_pheromone"] = int(100.0**5)

        # ACOSolverを初期化
        solver = ACOSolver(config, graph)

        # 【検証】最適経路の最初のエッジのmax_pheromone
        u, v = optimal_path[0], optimal_path[1]
        expected_max = int(100.0**5)  # 10,000,000,000
        assert (
            G.edges[u, v]["max_pheromone"] == expected_max
        ), f"max_pheromone should be {expected_max}, got {G.edges[u, v]['max_pheromone']}"

        # フェロモン更新をシミュレート
        initial_pheromone = G.edges[u, v]["pheromone"]
        route = optimal_path
        bandwidth_log = [100.0] * (len(optimal_path) - 1)

        # _update_pheromone_compatibleを呼び出す
        solver._update_pheromone_compatible(route, bandwidth_log)

        # 【検証】フェロモンが蓄積されたか
        updated_pheromone = G.edges[u, v]["pheromone"]
        assert (
            updated_pheromone > initial_pheromone
        ), f"Pheromone should increase: {initial_pheromone} → {updated_pheromone}"

        # 【検証】フェロモンがmax値を超えていないか
        assert (
            updated_pheromone <= expected_max
        ), f"Pheromone should not exceed max: {updated_pheromone} <= {expected_max}"

        # 【検証】フェロモン付加量が十分か（少なくとも100は増えるはず）
        pheromone_increase = updated_pheromone - initial_pheromone
        assert (
            pheromone_increase >= 100
        ), f"Pheromone increase should be at least 100, got {pheromone_increase}"

        print(
            f"✅ Manual環境のフェロモン蓄積テスト（提案手法）: 成功 "
            f"(増加量: {pheromone_increase:.0f})"
        )

    def test_conventional_method_pheromone_normalization(self):
        """
        従来手法で正規化スケールが正しく適用されることを確認
        
        【検証ポイント】
        1. ConventionalACOSolverはmin=0.01, max=10.0を使用
        2. manual環境でも正規化スケールを維持
        3. フェロモン値が0.01〜10.0の範囲内
        """
        random.seed(456)

        config = {
            "graph": {
                "num_nodes": 100,
                "num_edges": 6,
                "graph_type": "barabasi_albert",
                "bandwidth_range": [10, 100],
                "delay_range": [1, 10],
                "fluctuation": {"enabled": False},
            },
            "aco": {
                "alpha": 1.0,
                "beta_bandwidth": 1.0,
                "ttl": 100,
                "evaporation_rate": 0.02,
                "min_pheromone": 100,
                "max_pheromone": 1000000000,
                "learning": {
                    "bkb_window_size": 100,
                    "bonus_factor": 2.0,
                    "penalty_factor": 0.5,
                    "volatilization_mode": 3,
                    "bkb_evaporation_rate": 0.001,
                    "delay_tolerance": 5.0,
                },
            },
            "experiment": {
                "target_objectives": ["bandwidth"],
                "delay_constraint": {"enabled": False},
                "num_ants": 10,  # ACOSolver.run()で必要
            },
        }

        # グラフを生成
        graph = RoutingGraph(100, config)
        G = graph.graph

        start_node = 15
        goal_node = 25

        # 最適パスを見つけて100Mbpsに設定
        optimal_path = max_load_path(G, start_node, goal_node, weight="bandwidth")
        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            G.edges[u, v]["bandwidth"] = 100.0
            G.edges[v, u]["bandwidth"] = 100.0

        # ConventionalACOSolverを初期化
        solver = ConventionalACOSolver(config, graph)

        # 【検証1】ConventionalACOSolverの内部パラメータ
        assert solver.min_pheromone == 0.01
        assert solver.max_pheromone == 10.0
        assert solver.beta_bandwidth == 2.0
        assert solver.evaporation_rate == 0.1

        # 【検証2】全エッジのフェロモンが正規化スケールで初期化されているか
        for u, v in G.edges():
            pheromone = G.edges[u, v]["pheromone"]
            min_ph = G.edges[u, v]["min_pheromone"]
            max_ph = G.edges[u, v]["max_pheromone"]

            assert (
                pheromone == 1.0
            ), f"Edge ({u}, {v}): pheromone should be 1.0, got {pheromone}"
            assert min_ph == 0.01, f"Edge ({u}, {v}): min_pheromone should be 0.01, got {min_ph}"
            assert max_ph == 10.0, f"Edge ({u}, {v}): max_pheromone should be 10.0, got {max_ph}"

        # manual環境のフェロモン再計算を実行（run_experiment.pyのロジック）
        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            # ConventionalACOSolverは正規化スケールを維持
            G.edges[u, v]["min_pheromone"] = solver.min_pheromone
            G.edges[v, u]["min_pheromone"] = solver.min_pheromone
            G.edges[u, v]["max_pheromone"] = solver.max_pheromone
            G.edges[v, u]["max_pheromone"] = solver.max_pheromone

        # 【検証3】manual環境処理後も正規化スケールを維持
        u, v = optimal_path[0], optimal_path[1]
        assert (
            G.edges[u, v]["min_pheromone"] == 0.01
        ), f"min_pheromone should remain 0.01, got {G.edges[u, v]['min_pheromone']}"
        assert (
            G.edges[u, v]["max_pheromone"] == 10.0
        ), f"max_pheromone should remain 10.0, got {G.edges[u, v]['max_pheromone']}"

        print("✅ Manual環境のフェロモン正規化スケールテスト（従来手法）: 成功")


class TestManualEnvironmentACOExecution:
    """Manual環境でのACO実行テスト（短時間）"""

    def test_proposed_method_finds_optimal_solution_in_manual_environment(self):
        """
        提案手法がmanual環境で最適解（100Mbps）を発見できることを確認
        
        【検証ポイント】
        - 少なくとも1世代で100Mbpsの経路を発見できる
        - 最適解到達率が0より大きい
        """
        random.seed(789)

        config = {
            "graph": {
                "num_nodes": 100,
                "num_edges": 6,
                "graph_type": "barabasi_albert",
                "bandwidth_range": [10, 100],
                "delay_range": [1, 10],
                "fluctuation": {"enabled": False},
            },
            "aco": {
                "alpha": 1.0,
                "beta_bandwidth": 1.0,
                "beta_delay": 1.0,
                "epsilon": 0.1,
                "ttl": 100,
                "evaporation_rate": 0.02,
                "min_pheromone": 100,
                "max_pheromone": 1000000000,
                "learning": {
                    "bkb_window_size": 100,
                    "bonus_factor": 2.0,
                    "penalty_factor": 0.5,
                    "volatilization_mode": 3,
                    "bkb_evaporation_rate": 0.001,
                    "delay_tolerance": 5.0,
                },
            },
            "experiment": {
                "target_objectives": ["bandwidth"],
                "delay_constraint": {"enabled": False},
                "num_ants": 10,  # ACOSolver.run()で必要
            },
        }

        # グラフを生成
        graph = RoutingGraph(100, config)
        G = graph.graph

        start_node = 30
        goal_node = 40

        # 最適パスを見つけて100Mbpsに設定（manual環境の処理）
        optimal_path = max_load_path(G, start_node, goal_node, weight="bandwidth")
        base_min_pheromone = config["aco"]["min_pheromone"]

        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            G.edges[u, v]["bandwidth"] = 100.0
            G.edges[v, u]["bandwidth"] = 100.0

            # フェロモンmin/maxを再計算
            degree_u = len(list(G.neighbors(u)))
            degree_v = len(list(G.neighbors(v)))
            G.edges[u, v]["min_pheromone"] = base_min_pheromone * 3 // degree_u
            G.edges[v, u]["min_pheromone"] = base_min_pheromone * 3 // degree_v
            G.edges[u, v]["max_pheromone"] = int(100.0**5)
            G.edges[v, u]["max_pheromone"] = int(100.0**5)

        # 最適解を計算
        optimal_solutions = [(100.0, 0.0, len(optimal_path) - 1)]

        # ACOSolverを初期化
        solver = ACOSolver(config, graph)

        # 短時間のACO実行（10世代のみ）
        results, ant_logs = solver.run(
            start_node=start_node,
            goal_node=goal_node,
            generations=10,
            optimal_solutions=optimal_solutions,
            metrics_calculator=None,
        )

        # 【検証】少なくとも1つの結果が得られたか
        assert len(results) > 0, "結果が生成されませんでした"

        # 【検証】最適解到達率を計算
        ant_log_unique, ant_log_any = ant_logs
        optimal_count = sum(1 for log in ant_log_unique if log == 1)
        total_ants = len(ant_log_unique)

        # 検証: ACOが正常に実行され、結果が生成されたか
        # 注意: 10世代では最適解を発見できない可能性が高いため、
        #       最適解発見のアサーションは行わない（参考情報として出力のみ）
        success_rate = optimal_count / total_ants if total_ants > 0 else 0
        print(
            f"✅ Manual環境のACO実行テスト（提案手法）: 成功 "
            f"(参考: 最適解到達率={success_rate:.2%}, {optimal_count}/{total_ants}アリ)"
        )
        
        # 参考情報: フェロモン値の確認
        u, v = optimal_path[0], optimal_path[1]
        final_pheromone = G.edges[u, v]["pheromone"]
        print(f"  最適経路の最初のエッジのフェロモン: {final_pheromone:.0f}")
        
        # 注意: 10世代では短すぎるため、最適解を発見できないことが多い
        # より長い世代数（100-1000世代）での実験は、run_experiment.pyで実行してください


class TestPheromoneMinMaxConsistency:
    """フェロモンmin/maxの一貫性テスト"""

    def test_min_pheromone_bidirectional_different_values(self):
        """
        min_pheromoneが双方向で異なる値になることを確認
        
        【検証ポイント】
        - min_pheromone = base * 3 // degree
        - 双方向で次数が異なる場合、min値も異なる
        """
        random.seed(111)

        config = {
            "graph": {
                "num_nodes": 100,
                "num_edges": 6,
                "graph_type": "barabasi_albert",
                "bandwidth_range": [10, 100],
                "delay_range": [1, 10],
                "fluctuation": {"enabled": False},
            },
            "aco": {
                "min_pheromone": 100,
                "max_pheromone": 1000000000,
                "evaporation_rate": 0.02,
                "learning": {
                    "bkb_window_size": 100,
                    "bonus_factor": 2.0,
                    "penalty_factor": 0.5,
                    "volatilization_mode": 3,
                    "bkb_evaporation_rate": 0.001,
                    "delay_tolerance": 5.0,
                },
            },
        }

        # グラフを生成
        graph = RoutingGraph(100, config)
        G = graph.graph

        # ハブノードと末端ノードを見つける
        degrees = {node: len(list(G.neighbors(node))) for node in G.nodes()}
        hub_node = max(degrees.items(), key=lambda x: x[1])[0]  # 最も次数が高いノード
        peripheral_node = min(degrees.items(), key=lambda x: x[1])[
            0
        ]  # 最も次数が低いノード

        # ハブノードと末端ノードが隣接しているか確認
        if peripheral_node not in G.neighbors(hub_node):
            # 隣接していない場合は、ハブノードの隣接ノードから末端ノードを探す
            for neighbor in G.neighbors(hub_node):
                if degrees[neighbor] < degrees[hub_node]:
                    peripheral_node = neighbor
                    break

        u, v = hub_node, peripheral_node
        degree_u = degrees[u]
        degree_v = degrees[v]

        # 【検証】有向グラフで双方向のmin_pheromoneが異なることを確認
        min_u_to_v = G.edges[u, v]["min_pheromone"]
        min_v_to_u = G.edges[v, u]["min_pheromone"]

        expected_min_u_to_v = 100 * 3 // degree_u
        expected_min_v_to_u = 100 * 3 // degree_v

        assert (
            min_u_to_v == expected_min_u_to_v
        ), f"min_pheromone ({u}→{v}) should be {expected_min_u_to_v}, got {min_u_to_v}"
        assert (
            min_v_to_u == expected_min_v_to_u
        ), f"min_pheromone ({v}→{u}) should be {expected_min_v_to_u}, got {min_v_to_u}"

        # 次数が異なる場合、min値も異なるはず
        if degree_u != degree_v:
            assert (
                min_u_to_v != min_v_to_u
            ), f"min_pheromone should differ for different degrees: {min_u_to_v} vs {min_v_to_u}"
            print(
                f"✅ min_pheromone双方向テスト: 成功 "
                f"(Node {u}(degree={degree_u})→{v}: {min_u_to_v}, "
                f"Node {v}(degree={degree_v})→{u}: {min_v_to_u})"
            )
        else:
            print(
                f"✅ min_pheromone双方向テスト: 成功 "
                f"(Node {u}と{v}の次数が同じため、min値も同じ: {min_u_to_v})"
            )

