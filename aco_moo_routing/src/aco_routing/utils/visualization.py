"""
可視化モジュール

パレートフロンティア、ACO解の散布図、収束率の推移などを可視化します。
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


class Visualizer:
    """可視化を行うクラス"""

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = output_dir

    def plot_pareto_frontier_2d(
        self,
        optimal_solutions: List[Tuple[float, float, int]],
        aco_solutions: List[Tuple[float, float, int]],
        filename: str = "pareto_frontier_2d.svg",
    ) -> None:
        """
        2次元パレートフロンティアを可視化（Bandwidth vs Delay）

        Args:
            optimal_solutions: 最適解のリスト [(bandwidth, delay, hops), ...]
            aco_solutions: ACO解のリスト [(bandwidth, delay, hops), ...]
            filename: 保存するファイル名
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # ACO解をプロット
        if aco_solutions:
            aco_bandwidths = [s[0] for s in aco_solutions]
            aco_delays = [s[1] for s in aco_solutions]
            ax.scatter(
                aco_delays,
                aco_bandwidths,
                c="blue",
                alpha=0.6,
                s=50,
                label="ACO Solutions",
            )

        # 最適解をプロット
        if optimal_solutions:
            opt_bandwidths = [s[0] for s in optimal_solutions]
            opt_delays = [s[1] for s in optimal_solutions]
            ax.scatter(
                opt_delays,
                opt_bandwidths,
                c="red",
                marker="*",
                s=200,
                label="Optimal Solutions",
                zorder=5,
            )

        ax.set_xlabel("Delay [ms]")
        ax.set_ylabel("Bandwidth [Mbps]")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_pareto_frontier_3d(
        self,
        optimal_solutions: List[Tuple[float, float, int]],
        aco_solutions: List[Tuple[float, float, int]],
        filename: str = "pareto_frontier_3d.svg",
    ) -> None:
        """
        3次元パレートフロンティアを可視化（Bandwidth vs Delay vs Hops）

        Args:
            optimal_solutions: 最適解のリスト [(bandwidth, delay, hops), ...]
            aco_solutions: ACO解のリスト [(bandwidth, delay, hops), ...]
            filename: 保存するファイル名
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # ACO解をプロット
        if aco_solutions:
            aco_bandwidths = [s[0] for s in aco_solutions]
            aco_delays = [s[1] for s in aco_solutions]
            aco_hops = [s[2] for s in aco_solutions]
            ax.scatter(
                aco_delays,
                aco_bandwidths,
                aco_hops,
                c="blue",
                alpha=0.6,
                s=50,
                label="ACO Solutions",
            )

        # 最適解をプロット
        if optimal_solutions:
            opt_bandwidths = [s[0] for s in optimal_solutions]
            opt_delays = [s[1] for s in optimal_solutions]
            opt_hops = [s[2] for s in optimal_solutions]
            ax.scatter(
                opt_delays,
                opt_bandwidths,
                opt_hops,
                c="red",
                marker="*",
                s=200,
                label="Optimal Solutions",
            )

        ax.set_xlabel("Delay [ms]")
        ax.set_ylabel("Bandwidth [Mbps]")
        ax.set_zlabel("Hops")
        ax.legend()

        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_convergence_rate(
        self,
        pareto_discovery_rates: List[float],
        filename: str = "convergence_rate.svg",
    ) -> None:
        """
        収束率の推移を可視化

        Args:
            pareto_discovery_rates: 各世代のパレート発見率のリスト
            filename: 保存するファイル名
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        generations = range(len(pareto_discovery_rates))
        ax.plot(generations, pareto_discovery_rates, marker="o", linewidth=2)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Pareto Discovery Rate")
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_metrics_summary(
        self,
        metrics: dict,
        filename: str = "metrics_summary.png",
    ) -> None:
        """
        評価指標のサマリーを可視化

        Args:
            metrics: 評価指標の辞書 {"Pareto Discovery Rate": float, "Dominance Rate": float, ...}
            filename: 保存するファイル名
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        names = list(metrics.keys())
        values = list(metrics.values())

        bars = ax.bar(names, values, color=["#1f77b4", "#ff7f0e"])
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.0)

        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, format="png", bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_optimal_selection_rate(
        self,
        ant_logs: List[List[int]],
        num_ants: int,
        filename: str = "optimal_selection_rate.svg",
    ) -> None:
        """
        最適解選択率の推移を可視化

        Args:
            ant_logs: 各シミュレーションのant_logのリスト
            num_ants: 1世代あたりのアリの数
            filename: 保存するファイル名
        """
        if not ant_logs:
            print("Warning: No ant_logs data to plot")
            return

        num_simulations = len(ant_logs)

        # 世代ごとの最適解選択率を計算
        if num_ants == 1:
            # ANT_NUM = 1 の場合
            if not ant_logs[0]:
                return
            num_generations = len(ant_logs[0])

            optimal_selection_rates = []
            for gen_idx in range(num_generations):
                optimal_count = 0
                for sim_log in ant_logs:
                    if sim_log[gen_idx] >= 0:
                        optimal_count += 1
                rate = (optimal_count / num_simulations) * 100
                optimal_selection_rates.append(rate)
        else:
            # ANT_NUM > 1 の場合（チャンク処理）
            if not ant_logs[0]:
                return
            total_log_entries = len(ant_logs[0])
            num_generations = total_log_entries // num_ants

            optimal_selection_rates = []
            for gen_idx in range(num_generations):
                optimal_count = 0
                total_count = 0
                for sim_log in ant_logs:
                    start_index = gen_idx * num_ants
                    end_index = start_index + num_ants
                    generation_chunk = sim_log[start_index:end_index]
                    for idx in generation_chunk:
                        total_count += 1
                        if idx >= 0:
                            optimal_count += 1
                rate = (optimal_count / total_count * 100) if total_count > 0 else 0
                optimal_selection_rates.append(rate)

        # グラフを描画
        fig, ax = plt.subplots(figsize=(12, 6))
        generations = range(len(optimal_selection_rates))
        ax.plot(generations, optimal_selection_rates, linewidth=2)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Optimal Selection Rate [%]")
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_optimal_solution_selection_stacked(
        self,
        ant_logs: List[List[int]],
        num_ants: int,
        optimal_solutions: List[Tuple[float, float, int]],
        filename: str = "optimal_solution_selection_stacked.svg",
    ) -> None:
        """
        各パス（最適解 + その他のパス）の選択率を線グラフで表示

        Args:
            ant_logs: 各シミュレーションのant_logのリスト
                    各ant_logは各アリの最適解インデックスを記録（-1=非最適解, 0以上=最適解のインデックス）
            num_ants: 1世代あたりのアリの数
            optimal_solutions: 最適解のリスト [(bandwidth, delay, hops), ...]
            filename: 保存するファイル名
        """
        if not ant_logs or not optimal_solutions:
            print("Warning: No ant_logs or optimal_solutions data to plot")
            return

        num_optimal_solutions = len(optimal_solutions)

        # 世代ごとの各最適解への到達率を計算
        if num_ants == 1:
            # ANT_NUM = 1 の場合
            if not ant_logs[0]:
                return
            num_generations = len(ant_logs[0])

            # 各世代、各パスへの到達率（その他のパスも含む）
            generation_rates = []
            for gen_idx in range(num_generations):
                # 各パスへの到達割合をカウント
                # num_ants == 1 の場合、1世代で1匹のアリがゴール
                # 100回のシミュレーションで、各パスに到達したアリの割合を計算
                optimal_counts_sum = [0.0] * num_optimal_solutions
                other_paths_count_sum = 0.0  # その他のパスのカウント合計
                total_ants_count = 0  # 総アリ数

                for sim_log in ant_logs:
                    optimal_index = sim_log[gen_idx]
                    # 統一形式: 0以上 = 最適解のインデックス、-1 = ゴール未到達、-2 = 非最適解
                    if optimal_index >= 0 and optimal_index < num_optimal_solutions:
                        # 最適解に到達
                        optimal_counts_sum[optimal_index] += 1
                    else:
                        # 最適解以外のパス（-1=ゴール未到達、-2=非最適解、範囲外）
                        other_paths_count_sum += 1
                    total_ants_count += 1

                # 到達率に変換（%）
                # 100回のシミュレーションで、各パスに到達したアリの割合を計算
                if total_ants_count > 0:
                    rates = [
                        float((count_sum / total_ants_count) * 100)
                        for count_sum in optimal_counts_sum
                    ]
                    # その他のパスの到達率も追加
                    other_rate = float((other_paths_count_sum / total_ants_count) * 100)
                    rates.append(other_rate)
                else:
                    rates = [0.0] * (num_optimal_solutions + 1)
                generation_rates.append(rates)
        else:
            # ANT_NUM > 1 の場合（チャンク処理）
            if not ant_logs[0]:
                return
            total_log_entries = len(ant_logs[0])
            num_generations = total_log_entries // num_ants

            generation_rates = []
            for gen_idx in range(num_generations):
                # 各パスへの到達割合をカウント
                # 1世代で10匹のアリがゴールする場合、各パスに到達したアリの数をカウント
                # 100回のシミュレーションで、各パスに到達したアリの割合の平均を計算
                optimal_counts_sum = [0.0] * num_optimal_solutions
                other_paths_count_sum = 0.0  # その他のパスのカウント合計
                total_ants_count = 0  # 総アリ数

                for sim_log in ant_logs:
                    start_index = gen_idx * num_ants
                    end_index = start_index + num_ants
                    generation_chunk = sim_log[start_index:end_index]

                    # その世代のチャンク内で、各パスに到達したアリの数をカウント
                    # 統一形式: 0以上 = 最適解のインデックス、-1 = ゴール未到達、-2 = 非最適解
                    # 例（帯域のみ最適化）: [0, 0, -2, 0, 0, -2, -1, 0, ...]
                    #     → 最適解（インデックス0）: 6回
                    #     → その他のパス（-1, -2）: 2回
                    # 例（パレート最適化）: [0, 0, 1, -1, 0, 2, -2, 1, 0, -1]
                    #     → 最適解0: 4回
                    #     → 最適解1: 2回
                    #     → 最適解2: 1回
                    #     → その他のパス: 3回
                    for opt_idx in range(num_optimal_solutions):
                        count = generation_chunk.count(opt_idx)
                        optimal_counts_sum[opt_idx] += count
                    # その他のパス（最適解以外のパス）のカウント
                    # -1（ゴール未到達）、-2（非最適解）、範囲外のインデックスを含む
                    optimal_indices_in_chunk = [
                        idx
                        for idx in generation_chunk
                        if 0 <= idx < num_optimal_solutions
                    ]
                    other_count = len(generation_chunk) - len(optimal_indices_in_chunk)
                    other_paths_count_sum += other_count
                    total_ants_count += len(generation_chunk)

                # 到達率に変換（%）
                # 100回のシミュレーション × num_ants匹のアリで、各パスに到達したアリの割合を計算
                if total_ants_count > 0:
                    rates = [
                        float((count_sum / total_ants_count) * 100)
                        for count_sum in optimal_counts_sum
                    ]
                    # その他のパスの到達率も追加
                    other_rate = float((other_paths_count_sum / total_ants_count) * 100)
                    rates.append(other_rate)
                else:
                    rates = [0.0] * (num_optimal_solutions + 1)
                generation_rates.append(rates)

        if not generation_rates:
            print("Warning: No generation_rates calculated")
            return

        # 線グラフを描画（パスごとの選択率）
        fig, ax = plt.subplots(figsize=(14, 7))
        generations = range(len(generation_rates))

        # 各最適解の線を描画
        colors = plt.cm.tab10(range(num_optimal_solutions))
        for opt_idx in range(num_optimal_solutions):
            rates = [gen_rates[opt_idx] for gen_rates in generation_rates]
            b, d, h = optimal_solutions[opt_idx]
            label = f"S{opt_idx+1}: B={b:.0f}, D={d:.0f}, H={h}"
            ax.plot(
                generations,
                rates,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=label,
                color=colors[opt_idx],
            )

        # その他のパスの線を描画
        other_rates = [gen_rates[-1] for gen_rates in generation_rates]
        ax.plot(
            generations,
            other_rates,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Other Paths",
            color="gray",
            linestyle="--",
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Path Selection Rate [%]")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        # グリッドの設定
        ax.tick_params(
            which="major",
            direction="out",
            length=5,
            width=1.5,
            color="black",
        )
        ax.tick_params(
            which="minor",
            direction="out",
            length=3,
            width=1.0,
            color="black",
        )

        ax.minorticks_on()

        # 保存（凡例のスペースを確保）
        output_path = self.output_dir / filename
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 右側にスペースを確保
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def calculate_path_selection_ranking(
        self,
        all_results: List[List[Dict]],
        num_simulations: int,
        generations: int,
        num_ants: int,
        all_optimal_solutions: Optional[List[List[Tuple[float, float, int]]]] = None,
    ) -> List[Tuple[Tuple[float, float, int], float, int, bool]]:
        """
        全パスへの選択率を計算し、ランキングを作成

        Args:
            all_results: 全シミュレーションのresultsリスト
                        各resultsは[{"generation": int, "solutions": List[Tuple[float, float, int]]}, ...]
            num_simulations: シミュレーション数
            generations: 世代数
            num_ants: 1世代あたりのアリの数
            all_optimal_solutions: 各シミュレーションの最適解のリスト
                                  [[(bandwidth, delay, hops), ...], ...]

        Returns:
            ランキングリスト: [(path, selection_rate, count, is_optimal), ...]
            path: (bandwidth, delay, hops)のタプル
            selection_rate: 選択率（%）
            count: 選択回数
            is_optimal: 正解（Pareto最適解）と一致する場合True（各シミュレーションで平均）
        """
        # 全パスへの選択回数をカウント
        path_counter: Counter[Tuple[float, float, int]] = Counter()
        total_ants = 0

        # 各パスが各シミュレーションで最適解と一致した回数をカウント
        path_optimal_matches: Dict[Tuple[float, float, int], int] = {}

        for sim_idx, results in enumerate(all_results):
            # このシミュレーションの最適解を取得
            sim_optimal_solutions = (
                all_optimal_solutions[sim_idx]
                if all_optimal_solutions and sim_idx < len(all_optimal_solutions)
                else []
            )

            for result in results:
                solutions = result.get("solutions", [])
                for solution in solutions:
                    # 解をタプルとして扱い、PIDとして使用
                    path = tuple(solution)  # (bandwidth, delay, hops)
                    path_counter[path] += 1
                    total_ants += 1

                    # このシミュレーションの最適解と一致するかチェック
                    if sim_optimal_solutions:
                        b, d, h = path
                        for opt_b, opt_d, opt_h in sim_optimal_solutions:
                            # 完全一致（誤差を許容）
                            bandwidth_match = abs(b - opt_b) < max(0.01, opt_b * 0.01)
                            delay_match = abs(d - opt_d) < max(0.1, opt_d * 0.01)
                            hops_match = h == opt_h
                            if bandwidth_match and delay_match and hops_match:
                                path_optimal_matches[path] = (
                                    path_optimal_matches.get(path, 0) + 1
                                )
                                break

        # 選択率を計算してランキングを作成
        ranking = []
        for path, count in path_counter.items():
            selection_rate = (count / total_ants * 100) if total_ants > 0 else 0.0

            # 正解マッチング判定（各シミュレーションで一致した割合）
            # このパスが最適解と一致したシミュレーションの割合
            match_count = path_optimal_matches.get(path, 0)
            is_optimal = (
                match_count / num_simulations
            ) >= 0.5  # 50%以上のシミュレーションで一致

            ranking.append((path, selection_rate, count, is_optimal))

        # 選択率で降順にソート
        ranking.sort(key=lambda x: x[1], reverse=True)

        return ranking

    def plot_path_selection_ranking(
        self,
        ranking: List[Tuple[Tuple[float, float, int], float, int, bool]],
        top_n: int = 20,
        filename: str = "path_selection_ranking.svg",
    ) -> None:
        """
        パス選択率ランキングを可視化

        Args:
            ranking: ランキングリスト [(path, selection_rate, count, is_optimal), ...]
            top_n: 表示する上位N個のパス
            filename: 保存するファイル名
        """
        if not ranking:
            print("Warning: No ranking data to plot")
            return

        # 上位N個のパスを取得
        top_paths = ranking[:top_n]

        # データを準備
        labels = []
        rates = []
        colors = []
        for i, (path, rate, count, is_optimal) in enumerate(top_paths, 1):
            b, d, h = path
            marker = "✓" if is_optimal else ""
            label = f"P{i}: B={b:.0f}, D={d:.0f}, H={h} {marker}"
            labels.append(label)
            rates.append(rate)
            # 正解マッチングの場合は緑色、そうでない場合は青色
            colors.append("#2ecc71" if is_optimal else "#3498db")

        # グラフを作成
        fig, ax = plt.subplots(figsize=(14, max(8, len(top_paths) * 0.4)))
        y_pos = range(len(labels))

        # 横棒グラフ（正解マッチングの場合は緑色）
        bars = ax.barh(y_pos, rates, align="center", color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # 上位を上に表示
        ax.set_xlabel("Selection Rate [%]")
        ax.set_xlim(0, max(rates) * 1.1 if rates else 100)

        # 値をバーの右側に表示
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            ax.text(
                rate + max(rates) * 0.01 if rates else 1,
                i,
                f"{rate:.2f}%",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def filter_pareto_solutions(
        self,
        candidates: List[Tuple[Tuple[float, float, int], float, int, bool]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Tuple[float, float, int], float, int, bool]]:
        """
        事後的パレートフィルタリング（A Posteriori Pareto Filtering）

        候補パスから非支配解のみを抽出する。
        支配関係の定義:
        - 帯域は大きい方が良い
        - 遅延は小さい方が良い
        - ホップ数は小さい方が良い

        Args:
            candidates: 候補パスのリスト [(path, selection_rate, count, is_optimal), ...]
                       path: (bandwidth, delay, hops)のタプル
            top_n: フィルタリング前に上位N個に絞る（Noneの場合は全候補を使用）

        Returns:
            フィルタリング後のパレート最適解のリスト
        """
        if not candidates:
            return []

        # 上位N個に絞る（計算量削減のため）
        if top_n is not None and top_n > 0:
            candidates = candidates[:top_n]

        # パレート最適解を抽出
        pareto_front = []

        for i, candidate_a in enumerate(candidates):
            path_a, rate_a, count_a, is_opt_a = candidate_a
            b_a, d_a, h_a = path_a
            is_dominated = False

            for j, candidate_b in enumerate(candidates):
                if i == j:
                    continue

                path_b, _, _, _ = candidate_b
                b_b, d_b, h_b = path_b

                # BがAを支配しているかチェック
                # 支配の定義:
                # - Bの帯域 >= Aの帯域
                # - Bの遅延 <= Aの遅延
                # - Bのホップ数 <= Aのホップ数
                # かつ、少なくとも1つでBがAより優れている
                better_or_equal = (b_b >= b_a) and (d_b <= d_a) and (h_b <= h_a)
                strictly_better = (b_b > b_a) or (d_b < d_a) or (h_b < h_a)

                if better_or_equal and strictly_better:
                    is_dominated = True
                    break  # Aは負けたので終了

            # 誰にも負けなかったら採用
            if not is_dominated:
                pareto_front.append(candidate_a)

        # 選択率で降順にソート（フィルタリング後も人気順を維持）
        pareto_front.sort(key=lambda x: x[1], reverse=True)

        return pareto_front
