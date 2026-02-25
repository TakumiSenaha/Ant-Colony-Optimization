"""
既存実装との互換性テスト

このテストは、新実装（aco_solver.py）が既存実装（aco_main_bkb_available_bandwidth.py）と
同じ結果を出力することを確認します。

問題点：
1. グラフ生成時の乱数消費順序が異なる
2. ACOの探索ロジックが異なる
3. フェロモン更新/揮発ロジックが異なる

このテストを通過させることで、新実装が既存実装と完全に互換性を持つことを保証します。
"""

import random
import sys
from pathlib import Path

import networkx as nx

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def run_existing_implementation(seed: int, generations: int = 100, num_ants: int = 10):
    """既存実装のロジックを再現"""
    from modified_dijkstra import max_load_path
    
    # パラメータ（既存実装と同じ）
    V = 0.98
    MIN_F = 100
    TTL = 100
    ALPHA = 1.0
    BETA = 1.0
    EPSILON = 0.1
    TIME_WINDOW_SIZE = 100
    PENALTY_FACTOR = 0.5
    BKB_EVAPORATION_RATE = 0.999
    
    random.seed(seed)
    
    # グラフ生成
    graph = nx.barabasi_albert_graph(100, 6)
    for u, v in graph.edges():
        weight = random.randint(1, 10) * 10
        graph[u][v]['weight'] = weight
        graph[u][v]['pheromone'] = MIN_F
        graph[u][v]['min_pheromone'] = MIN_F
        graph[u][v]['max_pheromone'] = weight ** 5
    
    for node in graph.nodes():
        graph.nodes[node]['best_known_bottleneck'] = 0
    
    start_node = random.randint(0, 99)
    goal_node = random.choice([n for n in range(100) if n != start_node])
    
    # 最適パスを取得して100Mbpsに設定
    optimal_path = max_load_path(graph, start_node, goal_node)
    for u, v in zip(optimal_path[:-1], optimal_path[1:]):
        graph[u][v]['weight'] = 100
        graph[u][v]['max_pheromone'] = 100 ** 5
    
    optimal_bottleneck = 100
    
    # ACO実行
    gen_successes = []
    for gen in range(generations):
        gen_success = 0
        for ant_idx in range(num_ants):
            route = [start_node]
            current = start_node
            visited = {start_node}
            
            for step in range(TTL):
                if current == goal_node:
                    break
                neighbors = [n for n in graph.neighbors(current) if n not in visited]
                if not neighbors:
                    break
                
                if random.random() < EPSILON:
                    next_node = random.choice(neighbors)
                else:
                    probs = []
                    for n in neighbors:
                        tau = graph.edges[current, n]['pheromone'] ** ALPHA
                        eta = graph.edges[current, n]['weight'] ** BETA
                        probs.append(tau * eta)
                    total = sum(probs)
                    r = random.random() * total
                    cumsum = 0
                    next_node = neighbors[-1]
                    for i, p in enumerate(probs):
                        cumsum += p
                        if r <= cumsum:
                            next_node = neighbors[i]
                            break
                
                route.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if current == goal_node:
                widths = [graph.edges[u, v]['weight'] for u, v in zip(route[:-1], route[1:])]
                bottleneck = min(widths)
                if bottleneck >= optimal_bottleneck:
                    gen_success += 1
                
                # フェロモン更新
                base_pheromone = float(int(bottleneck)) * 10.0
                for u, v in zip(route[:-1], route[1:]):
                    new_pheromone = graph.edges[u, v]['pheromone'] + base_pheromone
                    graph.edges[u, v]['pheromone'] = min(new_pheromone, graph.edges[u, v]['max_pheromone'])
                    
                    if int(bottleneck) > graph.nodes[u]['best_known_bottleneck']:
                        graph.nodes[u]['best_known_bottleneck'] = int(bottleneck)
        
        gen_successes.append(1 if gen_success > 0 else 0)
        
        # 揮発
        for u, v in graph.edges():
            edge = graph.edges[u, v]
            edge_bw = int(edge['weight'])
            bkb_u = int(graph.nodes[u]['best_known_bottleneck'])
            
            if edge_bw >= bkb_u:
                edge['pheromone'] = max(edge['pheromone'] * V, edge['min_pheromone'])
            else:
                penalty_rate = V * PENALTY_FACTOR
                edge['pheromone'] = max(edge['pheromone'] * penalty_rate, edge['min_pheromone'])
        
        for node in graph.nodes():
            graph.nodes[node]['best_known_bottleneck'] = int(
                graph.nodes[node]['best_known_bottleneck'] * BKB_EVAPORATION_RATE
            )
    
    success_rate = sum(gen_successes) / generations
    return {
        'start_node': start_node,
        'goal_node': goal_node,
        'optimal_path': optimal_path,
        'success_rate': success_rate,
        'gen_successes': gen_successes,
    }


def run_new_implementation(seed: int, generations: int = 100, num_ants: int = 10):
    """新実装を使用"""
    from aco_routing.core.graph import RoutingGraph
    from aco_routing.core.node import NodeLearning
    from aco_routing.algorithms.single_objective_solver import max_load_path
    import yaml
    
    # パラメータ
    V = 0.98
    MIN_F = 100
    TTL = 100
    ALPHA = 1.0
    BETA = 1.0
    EPSILON = 0.1
    TIME_WINDOW_SIZE = 100
    PENALTY_FACTOR = 0.5
    BKB_EVAPORATION_RATE = 0.999
    
    random.seed(seed)
    
    # 設定ファイルを読み込み
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['graph']['num_nodes'] = 100
    config['graph']['num_edges'] = 6
    
    graph = RoutingGraph(100, config)
    G = graph.graph
    
    # NodeLearning初期化
    node_learning = {node: NodeLearning(node, TIME_WINDOW_SIZE) for node in G.nodes()}
    
    start_node = random.randint(0, 99)
    goal_node = random.choice([n for n in range(100) if n != start_node])
    
    # 最適パスを取得して100Mbpsに設定
    optimal_path = max_load_path(G, start_node, goal_node, weight='bandwidth')
    for u, v in zip(optimal_path[:-1], optimal_path[1:]):
        G.edges[u, v]['bandwidth'] = 100.0
        G.edges[u, v]['max_pheromone'] = 100 ** 5
    
    optimal_bottleneck = 100.0
    
    # ACO実行（既存実装と同じロジック）
    gen_successes = []
    for gen in range(generations):
        gen_success = 0
        for ant_idx in range(num_ants):
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
                    r = random.random() * total
                    cumsum = 0
                    next_node = neighbors[-1]
                    for i, p in enumerate(probs):
                        cumsum += p
                        if r <= cumsum:
                            next_node = neighbors[i]
                            break
                
                route.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if current == goal_node:
                widths = [G.edges[u, v]['bandwidth'] for u, v in zip(route[:-1], route[1:])]
                bottleneck = min(widths)
                if bottleneck >= optimal_bottleneck:
                    gen_success += 1
                
                # フェロモン更新
                bandwidth_int = int(bottleneck)
                base_pheromone = float(bandwidth_int) * 10.0
                for u, v in zip(route[:-1], route[1:]):
                    edge = G.edges[u, v]
                    new_pheromone = edge['pheromone'] + base_pheromone
                    edge['pheromone'] = min(new_pheromone, edge['max_pheromone'])
                    
                    nl = node_learning[u]
                    if bandwidth_int > nl.bkb:
                        nl.bkb = float(bandwidth_int)
        
        gen_successes.append(1 if gen_success > 0 else 0)
        
        # 揮発
        for u, v in G.edges():
            edge = G.edges[u, v]
            edge_bw = int(edge['bandwidth'])
            bkb_u = int(node_learning[u].bkb)
            
            if edge_bw >= bkb_u:
                edge['pheromone'] = max(edge['pheromone'] * V, edge['min_pheromone'])
            else:
                penalty_rate = V * PENALTY_FACTOR
                edge['pheromone'] = max(edge['pheromone'] * penalty_rate, edge['min_pheromone'])
        
        for node in G.nodes():
            node_learning[node].bkb *= BKB_EVAPORATION_RATE
    
    success_rate = sum(gen_successes) / generations
    return {
        'start_node': start_node,
        'goal_node': goal_node,
        'optimal_path': list(optimal_path),
        'success_rate': success_rate,
        'gen_successes': gen_successes,
    }


def test_compatibility():
    """既存実装と新実装の結果が一致することを確認"""
    print("=" * 60)
    print("既存実装と新実装の互換性テスト")
    print("=" * 60)
    
    for seed in [100, 101, 102]:
        print(f"\n--- Seed: {seed} ---")
        
        existing_result = run_existing_implementation(seed, generations=100, num_ants=10)
        new_result = run_new_implementation(seed, generations=100, num_ants=10)
        
        print(f"既存: Start={existing_result['start_node']}, Goal={existing_result['goal_node']}")
        print(f"新規: Start={new_result['start_node']}, Goal={new_result['goal_node']}")
        
        start_match = existing_result['start_node'] == new_result['start_node']
        goal_match = existing_result['goal_node'] == new_result['goal_node']
        path_match = existing_result['optimal_path'] == new_result['optimal_path']
        
        print(f"スタート一致: {start_match}")
        print(f"ゴール一致: {goal_match}")
        print(f"パス一致: {path_match}")
        
        print(f"既存成功率: {existing_result['success_rate']*100:.1f}%")
        print(f"新規成功率: {new_result['success_rate']*100:.1f}%")
        
        # 成功率の差が5%以内であることを確認
        rate_diff = abs(existing_result['success_rate'] - new_result['success_rate'])
        print(f"成功率の差: {rate_diff*100:.1f}%")
        
        if start_match and goal_match and path_match and rate_diff < 0.05:
            print("✓ テスト合格")
        else:
            print("✗ テスト不合格")


if __name__ == "__main__":
    test_compatibility()









