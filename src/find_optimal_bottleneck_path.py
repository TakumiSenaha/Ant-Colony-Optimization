import networkx as nx


# ボトルネックリンク容量を計算する関数
def bottleneck(graph: nx.Graph, path: list) -> float:
    """経路上のリンク容量の最小値（ボトルネック容量）を返す"""
    return min(graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


# 全探索による最適経路を見つけ、ボトルネック容量を求める
def find_optimal_bottleneck(graph: nx.Graph, start_node: int, goal_node: int):
    """
    全ての経路を探索し、最適なボトルネック経路を見つける
    """
    try:
        all_paths = list(
            nx.all_simple_paths(graph, source=start_node, target=goal_node)
        )
    except nx.NetworkXNoPath:
        print(f"No path exists between {start_node} and {goal_node}")
        return None, None

    optimal_path = None
    optimal_bottleneck = -float("inf")  # 初期値は非常に小さな値
    for path in all_paths:
        # 各経路のボトルネック容量を計算
        path_bottleneck = bottleneck(graph, path)
        if path_bottleneck > optimal_bottleneck:
            optimal_bottleneck = path_bottleneck
            optimal_path = path

    return optimal_path, optimal_bottleneck


# 結果の記録と出力
def save_optimal_path(optimal_path, optimal_bottleneck, start_node, goal_node):
    if optimal_path:
        print(f"最適な経路: {optimal_path}")
        print(f"ボトルネック帯域: {optimal_bottleneck}")
    else:
        print(f"{start_node} と {goal_node} の間に経路が見つかりませんでした。")


# 既存のネットワークを読み込む関数
def load_graph(file_name: str) -> nx.Graph:
    """保存されたエッジリスト形式のグラフを読み込む"""
    graph = nx.read_edgelist(file_name, data=[("weight", float)], nodetype=int)
    return graph


# Main 処理部分の追加
if __name__ == "__main__":
    # グラフのファイルを指定
    file_name = "ba_model_graph"  # 例: 保存されたエッジリスト形式のファイル名
    graph = load_graph(file_name)  # 既存のグラフを読み込む

    # 開始ノードと終了ノードを指定
    START_NODE = 6  # 例: 開始ノード
    GOAL_NODE = 9  # 例: 終了ノード

    # 全探索で最適経路とボトルネック帯域を計算
    optimal_path, optimal_bottleneck = find_optimal_bottleneck(
        graph, START_NODE, GOAL_NODE
    )

    # 結果を保存・表示
    save_optimal_path(optimal_path, optimal_bottleneck, START_NODE, GOAL_NODE)

    # グラフの可視化などの処理があればここに追加
    # visualize_graph(graph, "network_graph.pdf")
