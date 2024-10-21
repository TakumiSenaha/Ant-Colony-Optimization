import random

import matplotlib.pyplot as plt
import networkx as nx


# グラフを保存されたエッジリストから読み込む
def load_graph(file_name: str) -> nx.Graph:
    """保存されたエッジリスト形式のグラフを読み込む"""
    graph = nx.read_edgelist(file_name, data=[("weight", float)], nodetype=int)
    return graph


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


# グラフを描画してPDFに保存
def save_graph_as_pdf(graph: nx.Graph, path: list, pdf_file: str):
    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(graph)  # グラフのレイアウトを生成

    # ノードとエッジを描画
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        font_weight="bold",
    )

    # エッジラベル（重み）を描画
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    # 経路上のエッジをハイライト
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="r", width=2.5)

    # 経路の情報を表示
    path_weight = path2weight(graph, path)
    path_bottleneck = bottleneck(graph, path)
    plt.title(
        f"経路: {path}\n総帯域幅: {path_weight}, ボトルネック帯域: {path_bottleneck}"
    )

    # PDFに保存
    plt.savefig(pdf_file)
    print(f"グラフが {pdf_file} に保存されました。")
    plt.close()


# メイン処理
if __name__ == "__main__":
    use_existing_graph = input("既存のグラフを使用しますか？(y/n): ").lower() == "y"

    if use_existing_graph:
        # 既存のグラフを読み込む
        file_name = "ba_model_graph"
        G = load_graph(file_name)
        print(f"グラフ '{file_name}' を読み込みました。")
        # 始点と終点の選択
        node_list = list(G.nodes())
        src_node = random.choice(node_list)
        node_list.remove(src_node)
        dst_node = random.choice(node_list)

    else:
        # 新しいグラフを生成
        num_node = 10
        G = ba_graph(num_node, num_edges=3)
        print("新しいBAモデルのグラフを生成しました。")
        # 始点と終点の選択
        node_list = list(G.nodes())
        src_node = random.choice(node_list)
        node_list.remove(src_node)
        dst_node = random.choice(node_list)

    # 容量の逆数ラベルを追加
    add_inverse_weight_label(G)

    # 'inv'に対するDijkstra法
    v2 = nx.dijkstra_path(G, src_node, dst_node, weight="inv")

    # 結果の表示
    print(f"経路: {v2}")
    print(f"経路の重みの合計: {path2weight(G, v2)}")
    print(f"ボトルネックリンク容量: {bottleneck(G, v2)}")

    # グラフをPDFに保存
    pdf_file_name = "network_graph.pdf"
    save_graph_as_pdf(G, v2, pdf_file_name)

    if not use_existing_graph:
        # グラフの保存（エッジリスト形式）
        file_name = "ba_model_graph"  # 拡張子なしのファイル名
        nx.write_edgelist(
            G, file_name, comments="#", delimiter=" ", data=["weight"], encoding="utf-8"
        )
        print(f"グラフが '{file_name}' というファイルに保存されました。")
