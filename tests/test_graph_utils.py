import networkx as nx
import pytest

from src.utils.graph_utils import generate_ba_graph, load_graph


def test_load_graph():
    # サンプルエッジリストの作成
    file_name = "test_graph.txt"
    with open(file_name, "w") as f:
        f.write("0 1 30\n1 2 20\n2 3 10\n")

    G = load_graph(file_name)

    assert len(G.nodes) == 4  # ノード数が4
    assert len(G.edges) == 3  # エッジ数が3
    assert G[0][1]["weight"] == 30  # エッジの重みが正しい


def test_generate_ba_graph():
    G = generate_ba_graph(10)
    assert len(G.nodes) == 10  # 10ノードがあることを確認
    assert len(G.edges) > 0  # エッジが存在することを確認
