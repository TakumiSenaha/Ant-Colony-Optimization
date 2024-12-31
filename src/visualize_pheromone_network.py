import networkx as nx
from networkx.drawing.nx_agraph import to_agraph


def load_graph_with_pheromone(file_name: str) -> nx.Graph:
    """
    フェロモン情報付きのエッジリスト形式のグラフを読み込む
    フォーマット: source target weight pheromone
    """
    graph = nx.read_edgelist(
        file_name,
        data=[("weight", float), ("pheromone", float)],
        nodetype=int,
        create_using=nx.DiGraph,
    )
    print(f"グラフをロードしました: {file_name}")
    return graph


def visualize_pheromone_graph(graph: nx.Graph, output_filename="pheromone_network.pdf"):
    """
    グラフをフェロモン値と帯域幅を基に視覚化し、PDFとして出力する
    """
    A = to_agraph(graph)  # NetworkXグラフをGraphvizに変換

    # レイアウト設定
    A.graph_attr.update(
        {
            "overlap": "scale",  # ノード間の重なりを防ぐ
            "splines": "true",  # エッジの線を滑らかにする
            "nodesep": "2.0",  # ノード間の距離を広げる
            "ranksep": "2.0",  # 階層間の距離を広げる
            "start": "random",  # ノード配置をランダムに初期化
        }
    )

    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        pheromone = data.get("pheromone", 1.0)

        edge = A.get_edge(u, v)
        edge.attr["penwidth"] = str(max(1.0, weight / 50))  # エッジの太さを調整
        edge.attr["label"] = f"{weight:.1f} | {pheromone:.1f}"  # 帯域幅とフェロモン値
        edge.attr["color"] = (
            f"0.0 {max(0.1, pheromone / 1000.0)} 1.0"  # フェロモン値に基づく色
        )

    A.node_attr["shape"] = "circle"
    A.layout("sfdp")  # sfdpレイアウトを使用
    A.draw(output_filename, format="pdf")
    print(f"グラフを可視化し、{output_filename} に保存しました。")


if __name__ == "__main__":
    # 入力ファイル名（フェロモン情報付きネットワークファイル）
    input_file = "ba_model_graph_with_pheromone"
    output_file = "pheromone_network_visualization.pdf"

    # グラフの読み込み
    graph = load_graph_with_pheromone(input_file)

    # グラフの可視化と保存
    visualize_pheromone_graph(graph, output_file)
