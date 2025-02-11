import random
import time

import matplotlib.pyplot as plt
import networkx as nx


def load_graph(file_name: str) -> nx.Graph:
    """
    保存されたエッジリスト形式のグラフを読み込む。

    Parameters:
    -----------
    file_name : str
        読み込むエッジリストファイルのパス。

    Returns:
    --------
    nx.Graph
        読み込まれた NetworkX グラフオブジェクト。
    """
    graph = nx.Graph()
    with open(file_name, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    u, v, weight = int(parts[0]), int(parts[1]), float(parts[2])
                    graph.add_edge(u, v, weight=weight)
                except ValueError as e:
                    print(f"無視された行: {line.strip()} - エラー: {e}")
    return graph


def generate_ba_graph(
    num_nodes: int, num_edges: int = 3, lb: int = 10, ub: int = 100
) -> nx.Graph:
    """
    Barabási-Albertモデルを用いたランダムグラフを生成し、エッジに帯域幅の重みを設定する。

    Parameters:
    -----------
    num_nodes : int
        ノード数。
    num_edges : int, optional (default=3)
        各新規ノードが既存ノードに接続するエッジの数。
    lb : int, optional (default=10)
        帯域幅の最小値 (Mbps)。
    ub : int, optional (default=100)
        帯域幅の最大値 (Mbps)。

    Returns:
    --------
    nx.Graph
        生成されたランダムグラフ。
    """
    graph = nx.barabasi_albert_graph(num_nodes, num_edges)
    for u, v in graph.edges():
        graph[u][v]["weight"] = random.randint(lb, ub)
    return graph


def save_graph(graph: nx.Graph, file_name: str) -> None:
    """
    グラフをエッジリスト形式で保存する。

    Parameters:
    -----------
    graph : nx.Graph
        保存するグラフ。
    file_name : str
        保存するファイルのパス（.txt）。
    """
    nx.write_edgelist(graph, file_name, delimiter=" ", data=["weight"])
    print(f"グラフが '{file_name}' に保存されました。")


def add_inverse_weight_label(graph: nx.Graph) -> None:
    """
    すべてのエッジについて、容量の逆数 (1/weight) を追加する。

    Parameters:
    -----------
    graph : nx.Graph
        更新対象のグラフ。
    """
    for u, v in graph.edges():
        graph[u][v]["inv"] = 1 / graph[u][v]["weight"]


def path_weight(graph: nx.Graph, path: list) -> float:
    """
    経路の重みの合計を計算する。

    Parameters:
    -----------
    graph : nx.Graph
        計算対象のグラフ。
    path : list
        ノードのリスト（経路）。

    Returns:
    --------
    float
        経路の総重み（エッジの重みの合計）。
    """
    return sum(graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


def bottleneck(graph: nx.Graph, path: list) -> float:
    """
    経路上の最小リンク容量（ボトルネック）を計算する。

    Parameters:
    -----------
    graph : nx.Graph
        計算対象のグラフ。
    path : list
        ノードのリスト（経路）。

    Returns:
    --------
    float
        ボトルネック帯域値（経路上で最小のエッジ重み）。
    """
    return min(graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


def save_graph_as_pdf(graph: nx.Graph, path: list, pdf_file: str) -> None:
    """
    グラフを可視化し、指定した PDF ファイルに保存する。

    Parameters:
    -----------
    graph : nx.Graph
        可視化するグラフ。
    path : list
        ハイライト表示する経路（ノードのリスト）。
    pdf_file : str
        保存する PDF ファイルのパス。
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)

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

    # 経路上のエッジを赤色でハイライト
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="r", width=2.5)

    # 経路の情報を表示
    plt.title(
        f"経路: {path}\n総帯域幅: {path_weight(graph, path)}, ボトルネック帯域: {bottleneck(graph, path)}"
    )

    # PDFに保存
    plt.savefig(pdf_file)
    print(f"グラフが {pdf_file} に保存されました。")
    plt.close()


def generate_graph_and_save(num_nodes: int, file_prefix: str = "graph") -> str:
    """
    グラフを作成し、エッジリスト形式でファイルに保存する。

    Parameters:
    -----------
    num_nodes : int
        ノード数。
    file_prefix : str, optional (default="graph")
        ファイル名の接頭辞。

    Returns:
    --------
    str
        保存されたファイル名。
    """
    G = generate_ba_graph(num_nodes)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"{file_prefix}_{timestamp}.txt"
    save_graph(G, file_name)
    return file_name
