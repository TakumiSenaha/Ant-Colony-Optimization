import random

import networkx as nx


# BAモデルのグラフを生成
def ba_graph(num_nodes: int, num_edges: int = 3, lb: int = 1, ub: int = 10) -> nx.Graph:
    """Barabási-Albertモデルでグラフを生成"""
    graph = nx.barabasi_albert_graph(num_nodes, num_edges)
    for u, v in graph.edges():
        graph[u][v]["weight"] = (
            random.randint(lb, ub) * 10
        )  # リンクの容量（重み）を設定
    return graph


# 全てのリンクについて、容量の逆数をラベルに追加
def add_inverse_weight_label(graph: nx.Graph) -> None:
    for u, v in graph.edges():
        graph[u][v]["inv"] = 1 / graph[u][v]["weight"]


# 経路の重みの合計を計算
def path2weight(graph: nx.Graph, path: list) -> float:
    return sum(graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


# ボトルネックリンク容量を計算
def bottleneck(graph: nx.Graph, path: list) -> float:
    return min(graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


# グラフ生成
num_node = 100
G = ba_graph(num_node, num_edges=3)

# 容量の逆数ラベルを追加
add_inverse_weight_label(G)

# 始点と終点の選択
node_list = list(G.nodes())
src_node = random.choice(node_list)
node_list.remove(src_node)
dst_node = random.choice(node_list)

# 'inv'に対するDijkstra法
v2 = nx.dijkstra_path(G, src_node, dst_node, weight="inv")

# 結果の表示
print(f"経路: {v2}")
print(f"経路の重みの合計: {path2weight(G, v2)}")
print(f"ボトルネックリンク容量: {bottleneck(G, v2)}")

# グラフの保存（エッジリスト形式）
file_name = "ba_model_graph"  # 拡張子なしのファイル名
nx.write_edgelist(
    G, file_name, comments="#", delimiter=" ", data=["weight"], encoding="utf-8"
)

print(f"グラフが '{file_name}' というファイルに保存されました。")
