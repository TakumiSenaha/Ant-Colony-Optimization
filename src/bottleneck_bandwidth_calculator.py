import networkx as nx

# 指定されたパス
path = [30, 6, 11, 59, 35, 58, 5, 32]


def load_graph(file_name: str) -> nx.Graph:
    """保存されたエッジリスト形式のグラフを読み込む"""
    graph = nx.read_edgelist(file_name, data=[("weight", float)], nodetype=int)
    return graph


def log_bandwidth_per_move(graph: nx.Graph, path: list[int]) -> None:
    """
    指定されたパスの各移動ごとに帯域幅を表示し、ボトルネック帯域を計算する
    """
    bottleneck = float("inf")  # 初期値を無限大に設定

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if graph.has_edge(u, v):
            weight = graph[u][v]["weight"]  # エッジの帯域幅を取得
            print(f"{u}-{v}: 帯域幅 = {weight}")
            bottleneck = min(bottleneck, weight)  # 最小帯域幅を更新
        else:
            print(f"エッジが見つかりません: {u} - {v}")
            return None  # エッジが存在しない場合、Noneを返す

    print(f"指定されたパスのボトルネック帯域は: {bottleneck}")


# グラフをロードする
graph = load_graph("ba_model_graph")  # 適切なファイル名を指定してください

# 指定されたパスの各移動での帯域幅をログに出力し、ボトルネック帯域を確認
log_bandwidth_per_move(graph, path)
