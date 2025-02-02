import networkx as nx
import pytest
from networkx.utils import pairwise

from src.modified_dijkstra import max_load_path  # 自前の関数をimport


def validate_max_load_path(G, s, t, expected_bottleneck, path, weight="weight"):
    """
    経路が始点 s から終点 t まで適切につながっており、
    かつ経路上の最小エッジ重み（ボトルネック）が期待値と一致するかを確認する。
    """
    assert path[0] == s, f"経路の開始点が{s}であるべき"
    assert path[-1] == t, f"経路の終点が{t}であるべき"

    if callable(weight):
        weight_f = weight
    else:
        if G.is_multigraph():

            def weight_f(u, v, d):
                return min(e.get(weight, 1) for e in d.values())

        else:

            def weight_f(u, v, d):
                return d.get(weight, 1)

    computed_bottleneck = min(weight_f(u, v, G[u][v]) for u, v in pairwise(path))
    assert (
        expected_bottleneck == computed_bottleneck
    ), f"期待されるボトルネック {expected_bottleneck} だが、計算値 {computed_bottleneck} であった。"


class TestMaxLoadPath:
    def setup_method(self):
        """
        テスト用グラフをセットアップ。

        - **直接経路**: `S → X → T` （帯域 10 Mbps）
        - **遠回り経路**: `S → A → B → T` （帯域 100 Mbps）

        遠回りの方がボトルネック帯域が大きいので、`max_load_path` がそれを選択すべき。
        """
        self.G = nx.DiGraph()
        # 低帯域の直接経路
        self.G.add_edge("S", "X", weight=10)
        self.G.add_edge("X", "T", weight=10)
        # 高帯域の遠回り経路
        self.G.add_edge("S", "A", weight=100)
        self.G.add_edge("A", "B", weight=100)
        self.G.add_edge("B", "T", weight=100)
        # その他の補助エッジ
        self.G.add_edge("X", "A", weight=50)
        self.G.add_edge("B", "X", weight=50)

    def test_detour_selected(self):
        """
        直接経路（10 Mbps）よりも遠回り経路（100 Mbps）の方がボトルネック帯域が大きいため、
        `max_load_path` は遠回りを選択するはず。
        """
        path = max_load_path(self.G, "S", "T")
        validate_max_load_path(self.G, "S", "T", 100, path, weight="weight")
        assert path == ["S", "A", "B", "T"]

    def test_source_equals_target(self):
        """始点と終点が同じ場合、経路は1ノードだけになるべき。"""
        path = max_load_path(self.G, "S", "S")
        assert path == ["S"]

    def test_no_path(self):
        """経路が存在しない場合、NetworkXNoPath 例外を発生させるべき。"""
        H = nx.DiGraph()
        H.add_edge("S", "X", weight=10)
        H.add_edge("A", "T", weight=100)
        with pytest.raises(nx.NetworkXNoPath):
            max_load_path(H, "S", "T")
