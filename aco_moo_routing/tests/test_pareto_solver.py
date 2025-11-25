"""
パレートソルバーのテスト
"""

import pytest
from aco_routing.algorithms.pareto_solver import Label, ParetoSolver


class TestLabel:
    """Labelクラスのテスト"""

    def test_dominates(self):
        """支配関係のテスト"""
        label1 = Label(bandwidth=100, delay=10, hops=3, path=[0, 1, 2])
        label2 = Label(bandwidth=80, delay=15, hops=4, path=[0, 3, 2])

        # label1がlabel2を支配する
        assert label1.dominates(label2) is True
        # label2はlabel1を支配しない
        assert label2.dominates(label1) is False

    def test_dominates_equal(self):
        """同等の場合は支配しない"""
        label1 = Label(bandwidth=100, delay=10, hops=3, path=[0, 1, 2])
        label2 = Label(bandwidth=100, delay=10, hops=3, path=[0, 3, 2])

        assert label1.dominates(label2) is False
        assert label2.dominates(label1) is False

    def test_dominates_partial(self):
        """部分的に優れている場合は支配しない"""
        label1 = Label(bandwidth=100, delay=10, hops=3, path=[0, 1, 2])
        label2 = Label(bandwidth=90, delay=8, hops=4, path=[0, 3, 2])

        # label1は帯域で優れているが、遅延で劣る
        assert label1.dominates(label2) is False
        assert label2.dominates(label1) is False

    def test_is_dominated_by_any(self):
        """ラベルリストによる支配のテスト"""
        label1 = Label(bandwidth=80, delay=15, hops=4, path=[0, 3, 2])
        label2 = Label(bandwidth=100, delay=10, hops=3, path=[0, 1, 2])
        label3 = Label(bandwidth=90, delay=12, hops=3, path=[0, 4, 2])

        # label1はlabel2に支配される
        assert label1.is_dominated_by_any([label2, label3]) is True

        # label2はどれにも支配されない
        assert label2.is_dominated_by_any([label1, label3]) is False


class TestParetoSolver:
    """ParetoSolverクラスのテスト"""

    @pytest.fixture
    def simple_graph(self):
        """簡単なテスト用グラフを生成"""
        import networkx as nx

        graph = nx.Graph()
        graph.add_edge(0, 1, bandwidth=100, delay=5)
        graph.add_edge(1, 2, bandwidth=80, delay=3)
        graph.add_edge(0, 2, bandwidth=60, delay=2)

        return graph

    def test_find_pareto_frontier_simple(self, simple_graph):
        """シンプルなグラフでのパレートフロンティア計算"""
        solver = ParetoSolver(simple_graph)
        pareto_frontier = solver.find_pareto_frontier(source=0, destination=2)

        # 2つのパレート最適解が見つかるはず
        # 1. 経路 0->2: bandwidth=60, delay=2, hops=1
        # 2. 経路 0->1->2: bandwidth=80, delay=8, hops=2
        assert len(pareto_frontier) >= 1

        # ボトルネック帯域の確認
        bandwidths = [pf[0] for pf in pareto_frontier]
        assert 60 in bandwidths or 80 in bandwidths

    def test_is_pareto_optimal(self, simple_graph):
        """パレート最適解の判定テスト"""
        solver = ParetoSolver(simple_graph)
        pareto_frontier = solver.find_pareto_frontier(source=0, destination=2)

        # パレート最適解のいずれかを選択
        if pareto_frontier:
            pf_solution = pareto_frontier[0]
            solution = (pf_solution[0], pf_solution[1], pf_solution[2])

            # 完全一致の判定
            assert solver.is_pareto_optimal(solution, pareto_frontier) is True

    def test_dominance_check(self, simple_graph):
        """支配チェックのテスト"""
        solver = ParetoSolver(simple_graph)
        pareto_frontier = solver.find_pareto_frontier(source=0, destination=2)

        if pareto_frontier:
            # パレート最適解は支配されない
            pf_solution = pareto_frontier[0]
            solution = (pf_solution[0], pf_solution[1], pf_solution[2])
            assert solver.dominance_check(solution, pareto_frontier) is True

            # 明らかに劣る解は支配される
            bad_solution = (10, 100, 10)  # 低帯域、高遅延、多ホップ
            assert solver.dominance_check(bad_solution, pareto_frontier) is False
