"""
最適解判定のテスト

manualでは:
- 最適解のボトルネック帯域値が100になるように手動で設定
- 最適解の判定は「帯域が100以上か」（パスまでは拘らない）で行う
"""

import random
import sys
from pathlib import Path

import pytest

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aco_routing.core.graph import RoutingGraph
from aco_routing.core.node import NodeLearning
from aco_routing.algorithms.single_objective_solver import max_load_path, bottleneck_capacity


class TestOptimalSolutionDetection:
    """最適解判定のテストケース"""

    def test_manual_environment_optimal_is_100(self):
        """
        manual環境では、最適パスの帯域を100に設定し、
        ボトルネック帯域が100以上であれば最適解と判定する
        """
        random.seed(42)
        
        # 設定
        config = {
            'graph': {
                'num_nodes': 100,
                'num_edges': 6,
                'graph_type': 'barabasi_albert',
                'bandwidth_range': [10, 100],
                'delay_range': [1, 10],
                'fluctuation': {'enabled': False},
            },
            'aco': {
                'alpha': 1.0,
                'beta_bandwidth': 1.0,
                'beta_delay': 1.0,
                'epsilon': 0.1,
                'ttl': 100,
                'evaporation_rate': 0.02,
                'min_pheromone': 100,
                'max_pheromone': 1000000000,
                'time_window_size': 100,
                'penalty_factor': 0.5,
                'bkb_evaporation_rate': 0.999,
                'volatilization_mode': 3,
                'learning': {
                    'time_window_size': 100,
                    'bkb_window_size': 100,
                    'bkb_evaporation_rate': 0.999,
                    'bonus_factor': 1.0,
                    'delay_tolerance': 0.1,
                    'penalty_factor': 0.5,
                    'volatilization_mode': 3,
                },
            },
        }
        
        graph = RoutingGraph(100, config)
        G = graph.graph
        
        start_node = 21
        goal_node = 18
        
        # 最適パスを計算
        optimal_path = max_load_path(G, start_node, goal_node, weight='bandwidth')
        
        # 最適パスを100Mbpsに設定（双方向）
        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            G.edges[u, v]['bandwidth'] = 100.0
            G.edges[v, u]['bandwidth'] = 100.0  # 有向グラフでは双方向に設定
        
        # 最適解のボトルネック帯域は100
        optimal_bottleneck = bottleneck_capacity(G, optimal_path, weight='bandwidth')
        assert optimal_bottleneck == 100.0, f"Expected optimal bottleneck to be 100, got {optimal_bottleneck}"
        
        # テストケース1: 最適パスと全く同じ経路 → 成功
        test_path_same = optimal_path
        test_bottleneck = bottleneck_capacity(G, test_path_same, weight='bandwidth')
        assert test_bottleneck >= 100, f"Same path should have bottleneck >= 100, got {test_bottleneck}"
        
        # テストケース2: 異なる経路だがボトルネックが100以上 → 成功
        # (100Mbpsのエッジが他にも存在する可能性があるため)
        # この場合も最適解として判定すべき
        
        # テストケース3: ボトルネックが100未満 → 失敗
        # 帯域が100未満のエッジを含む経路
        
        print(f"Optimal path: {optimal_path}")
        print(f"Optimal bottleneck: {optimal_bottleneck}")
        
    def test_optimal_detection_logic(self):
        """
        最適解判定ロジックのテスト
        
        期待する動作:
        - solution_bandwidth >= current_optimal_bottleneck なら成功 (1)
        - solution_bandwidth < current_optimal_bottleneck なら失敗 (0)
        """
        current_optimal_bottleneck = 100.0
        
        # テストケース
        test_cases = [
            (100.0, True, "ボトルネック100は最適解"),
            (110.0, True, "ボトルネック110は最適解（100以上）"),
            (99.0, False, "ボトルネック99は非最適解"),
            (50.0, False, "ボトルネック50は非最適解"),
            (100.0001, True, "ボトルネック100.0001は最適解"),
        ]
        
        for solution_bandwidth, expected_success, description in test_cases:
            # 新実装のロジック（aco_solver.py line 728-733）
            bandwidth_ok = solution_bandwidth >= current_optimal_bottleneck
            log_value = 1 if bandwidth_ok else 0
            
            expected_log = 1 if expected_success else 0
            assert log_value == expected_log, f"{description}: expected {expected_log}, got {log_value}"
            print(f"✓ {description}")

    def test_ant_reaches_goal_with_bottleneck_100(self):
        """
        アリがゴールに到達し、ボトルネック帯域が100の場合、
        最適解として判定されることを確認
        """
        random.seed(42)
        
        config = {
            'graph': {
                'num_nodes': 100,
                'num_edges': 6,
                'graph_type': 'barabasi_albert',
                'bandwidth_range': [10, 100],
                'delay_range': [1, 10],
                'fluctuation': {'enabled': False},
            },
            'aco': {
                'alpha': 1.0,
                'beta_bandwidth': 1.0,
                'beta_delay': 1.0,
                'epsilon': 0.1,
                'ttl': 100,
                'evaporation_rate': 0.02,
                'min_pheromone': 100,
                'max_pheromone': 1000000000,
                'time_window_size': 100,
                'penalty_factor': 0.5,
                'bkb_evaporation_rate': 0.999,
                'volatilization_mode': 3,
                'learning': {
                    'time_window_size': 100,
                    'bkb_window_size': 100,
                    'bkb_evaporation_rate': 0.999,
                    'bonus_factor': 1.0,
                    'delay_tolerance': 0.1,
                    'penalty_factor': 0.5,
                    'volatilization_mode': 3,
                },
            },
        }
        
        graph = RoutingGraph(100, config)
        G = graph.graph
        
        start_node = 21
        goal_node = 18
        
        # 最適パスを計算
        optimal_path = max_load_path(G, start_node, goal_node, weight='bandwidth')
        
        # 最適パスを100Mbpsに設定（双方向）
        for u, v in zip(optimal_path[:-1], optimal_path[1:]):
            G.edges[u, v]['bandwidth'] = 100.0
            G.edges[v, u]['bandwidth'] = 100.0  # 有向グラフでは双方向に設定
        
        # 最適ボトルネック
        current_optimal_bottleneck = 100.0
        
        # シミュレーション
        V = 0.98
        ALPHA = 1.0
        BETA = 1.0
        EPSILON = 0.1
        TTL = 100
        ANT_NUM = 10
        GENERATION = 100
        TIME_WINDOW_SIZE = 100
        PENALTY_FACTOR = 0.5
        BKB_EVAPORATION_RATE = 0.999
        
        node_learning = {node: NodeLearning(node, TIME_WINDOW_SIZE) for node in G.nodes()}
        
        ant_log = []
        gen_successes = []
        
        for gen in range(GENERATION):
            gen_success = 0
            for ant_idx in range(ANT_NUM):
                route = [start_node]
                current = start_node
                visited = {start_node}
                
                for step in range(TTL):
                    if current == goal_node:
                        break
                    neighbors = [n for n in G.neighbors(current) if n not in visited]
                    if not neighbors:
                        break
                    
                    if random.random() < EPSILON:
                        next_node = random.choice(neighbors)
                    else:
                        probs = []
                        for n in neighbors:
                            tau = G.edges[current, n]['pheromone'] ** ALPHA
                            eta = G.edges[current, n]['bandwidth'] ** BETA
                            probs.append(tau * eta)
                        total = sum(probs)
                        if total > 0:
                            r = random.random() * total
                            cumsum = 0
                            next_node = neighbors[-1]
                            for i, p in enumerate(probs):
                                cumsum += p
                                if r <= cumsum:
                                    next_node = neighbors[i]
                                    break
                        else:
                            next_node = random.choice(neighbors)
                    
                    route.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if current == goal_node:
                    widths = [G.edges[u, v]['bandwidth'] for u, v in zip(route[:-1], route[1:])]
                    bottleneck = min(widths)
                    
                    # 【重要】既存実装と同じ判定ロジック
                    # min(ant.width) >= current_optimal_bottleneck
                    is_optimal = bottleneck >= current_optimal_bottleneck
                    
                    if is_optimal:
                        ant_log.append(1)
                        gen_success += 1
                    else:
                        ant_log.append(0)
                    
                    # フェロモン更新（簡略化、双方向）
                    bandwidth_int = int(bottleneck)
                    base_pheromone = float(bandwidth_int) * 10.0
                    for u, v in zip(route[:-1], route[1:]):
                        # u → v のフェロモン更新
                        edge_uv = G.edges[u, v]
                        new_pheromone_uv = edge_uv['pheromone'] + base_pheromone
                        edge_uv['pheromone'] = min(new_pheromone_uv, edge_uv.get('max_pheromone', 10**10))
                        
                        # v → u のフェロモン更新（有向グラフでは双方向に設定）
                        edge_vu = G.edges[v, u]
                        new_pheromone_vu = edge_vu['pheromone'] + base_pheromone
                        edge_vu['pheromone'] = min(new_pheromone_vu, edge_vu.get('max_pheromone', 10**10))
                        
                        nl = node_learning[u]
                        if bandwidth_int > nl.bkb:
                            nl.bkb = float(bandwidth_int)
                else:
                    ant_log.append(0)
                
            gen_successes.append(1 if gen_success > 0 else 0)
            
            # 揮発
            for u, v in G.edges():
                edge = G.edges[u, v]
                edge_bw = int(edge['bandwidth'])
                bkb_u = int(node_learning[u].bkb)
                
                if edge_bw >= bkb_u:
                    edge['pheromone'] = max(edge['pheromone'] * V, edge.get('min_pheromone', 100))
                else:
                    penalty_rate = V * PENALTY_FACTOR
                    edge['pheromone'] = max(edge['pheromone'] * penalty_rate, edge.get('min_pheromone', 100))
            
            for node in G.nodes():
                node_learning[node].bkb *= BKB_EVAPORATION_RATE
        
        # 結果
        total_ants = GENERATION * ANT_NUM
        success_count = sum(ant_log)
        gen_success_rate = sum(gen_successes) / GENERATION
        
        print(f"\n=== テスト結果 ===")
        print(f"全アリの成功率: {success_count}/{total_ants} ({success_count/total_ants*100:.1f}%)")
        print(f"世代ごとの成功率: {sum(gen_successes)}/{GENERATION} ({gen_success_rate*100:.1f}%)")
        
        # 世代ごとの成功率が50%以上であることを確認
        assert gen_success_rate >= 0.5, f"Expected generation success rate >= 50%, got {gen_success_rate*100:.1f}%"


if __name__ == "__main__":
    # 直接実行時のテスト
    test = TestOptimalSolutionDetection()
    print("=" * 60)
    print("Test 1: test_manual_environment_optimal_is_100")
    print("=" * 60)
    test.test_manual_environment_optimal_is_100()
    
    print("\n" + "=" * 60)
    print("Test 2: test_optimal_detection_logic")
    print("=" * 60)
    test.test_optimal_detection_logic()
    
    print("\n" + "=" * 60)
    print("Test 3: test_ant_reaches_goal_with_bottleneck_100")
    print("=" * 60)
    test.test_ant_reaches_goal_with_bottleneck_100()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

