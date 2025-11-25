"""
可視化モジュール

パレートフロンティア、ACO解の散布図、収束率の推移などを可視化します。
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


class Visualizer:
    """可視化を行うクラス"""

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_pareto_frontier_2d(
        self,
        pareto_frontier: List[Tuple[float, float, int, List[int]]],
        aco_solutions: List[Tuple[float, float, int]],
        filename: str = "pareto_frontier_2d.png",
    ) -> None:
        """
        パレートフロンティアとACO解の散布図（2次元: 帯域 vs 遅延）

        Args:
            pareto_frontier: 真のパレートフロンティア
            aco_solutions: ACOが見つけた解のリスト
            filename: 保存するファイル名
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # パレートフロンティア（赤色）
        if pareto_frontier:
            pf_delays = [pf[1] for pf in pareto_frontier]
            pf_bandwidths = [pf[0] for pf in pareto_frontier]
            ax.scatter(
                pf_delays,
                pf_bandwidths,
                c="red",
                marker="o",
                s=100,
                label="Pareto Frontier",
                alpha=0.7,
                edgecolors="black",
            )

        # ACO解（青色）
        if aco_solutions:
            aco_delays = [sol[1] for sol in aco_solutions]
            aco_bandwidths = [sol[0] for sol in aco_solutions]
            ax.scatter(
                aco_delays,
                aco_bandwidths,
                c="blue",
                marker="x",
                s=50,
                label="ACO Solutions",
                alpha=0.5,
            )

        ax.set_xlabel("Delay (ms)", fontsize=12)
        ax.set_ylabel("Bandwidth (Mbps)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 保存
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_pareto_frontier_3d(
        self,
        pareto_frontier: List[Tuple[float, float, int, List[int]]],
        aco_solutions: List[Tuple[float, float, int]],
        filename: str = "pareto_frontier_3d.png",
    ) -> None:
        """
        パレートフロンティアとACO解の散布図（3次元: 帯域 vs 遅延 vs ホップ数）

        Args:
            pareto_frontier: 真のパレートフロンティア
            aco_solutions: ACOが見つけた解のリスト
            filename: 保存するファイル名
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # パレートフロンティア（赤色）
        if pareto_frontier:
            pf_bandwidths = [pf[0] for pf in pareto_frontier]
            pf_delays = [pf[1] for pf in pareto_frontier]
            pf_hops = [pf[2] for pf in pareto_frontier]
            ax.scatter(
                pf_delays,
                pf_bandwidths,
                pf_hops,
                c="red",
                marker="o",
                s=100,
                label="Pareto Frontier",
                alpha=0.7,
                edgecolors="black",
            )

        # ACO解（青色）
        if aco_solutions:
            aco_bandwidths = [sol[0] for sol in aco_solutions]
            aco_delays = [sol[1] for sol in aco_solutions]
            aco_hops = [sol[2] for sol in aco_solutions]
            ax.scatter(
                aco_delays,
                aco_bandwidths,
                aco_hops,
                c="blue",
                marker="x",
                s=50,
                label="ACO Solutions",
                alpha=0.5,
            )

        ax.set_xlabel("Delay (ms)", fontsize=12)
        ax.set_ylabel("Bandwidth (Mbps)", fontsize=12)
        ax.set_zlabel("Hops", fontsize=12)
        ax.legend()

        # 保存
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_convergence_rate(
        self,
        convergence_rates: List[float],
        filename: str = "convergence_rate.png",
    ) -> None:
        """
        収束率の推移をプロット

        Args:
            convergence_rates: 世代ごとの収束率のリスト
            filename: 保存するファイル名
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        generations = list(range(len(convergence_rates)))
        ax.plot(generations, convergence_rates, marker="o", linestyle="-", linewidth=2)

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Pareto Discovery Rate", fontsize=12)
        ax.grid(True, alpha=0.3)

        # 保存
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_metrics_summary(
        self,
        metrics: dict,
        filename: str = "metrics_summary.png",
    ) -> None:
        """
        評価指標のサマリーを棒グラフで表示

        Args:
            metrics: 評価指標の辞書
            filename: 保存するファイル名
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        ax.bar(metric_names, metric_values, color="skyblue", edgecolor="black")
        ax.set_ylabel("Value", fontsize=12)
        ax.set_ylim(0, 1.1)  # 0.0 ~ 1.0の範囲

        # 値をバーの上に表示
        for i, v in enumerate(metric_values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

        # 保存
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_optimal_selection_rate(
        self,
        ant_logs: List[List[int]],
        num_ants: int,
        filename: str = "optimal_selection_rate.svg",
    ) -> None:
        """
        最適解選択率の遷移をプロット（参考: csv_log_analysis_percentage_of_optimal_solution_use_modified_dijkstra.py）

        Args:
            ant_logs: 各シミュレーションのant_logのリスト（各ant_logは各アリの成功/失敗を記録）
            num_ants: 1世代あたりのアリの数
            filename: 保存するファイル名
        """
        if not ant_logs:
            print("Warning: No ant_logs data to plot")
            return

        # 世代ごとの最適解選択率を計算
        num_simulations = len(ant_logs)
        optimal_percentages = []

        if num_ants == 1:
            # ANT_NUM = 1 の場合
            if not ant_logs[0]:
                return
            num_generations = len(ant_logs[0])

            for gen_idx in range(num_generations):
                count_optimal = sum(log[gen_idx] >= 0 for log in ant_logs)
                percentage = (count_optimal / num_simulations) * 100
                optimal_percentages.append(percentage)
        else:
            # ANT_NUM > 1 の場合（チャンク処理）
            if not ant_logs[0]:
                return
            total_log_entries = len(ant_logs[0])
            num_generations = total_log_entries // num_ants

            for gen_idx in range(num_generations):
                # 1世代で最適解に到達したアリの数をカウント
                total_optimal_count = 0
                total_ants_count = 0
                for sim_log in ant_logs:
                    start_index = gen_idx * num_ants
                    end_index = start_index + num_ants
                    generation_chunk = sim_log[start_index:end_index]

                    # その世代のチャンク内で最適解に到達したアリの数をカウント
                    optimal_count = sum(1 for idx in generation_chunk if idx >= 0)
                    total_optimal_count += optimal_count
                    total_ants_count += len(generation_chunk)

                # 到達率を計算（%）
                if total_ants_count > 0:
                    percentage = (total_optimal_count / total_ants_count) * 100
                else:
                    percentage = 0.0
                optimal_percentages.append(percentage)

        if not optimal_percentages:
            print("Warning: No optimal_percentages calculated")
            return

        # グラフ描画（論文標準形式：箱型）
        fig, ax = plt.subplots(figsize=(10, 7))  # 白銀比に近い比率

        x_values = list(range(len(optimal_percentages)))
        y_values = optimal_percentages

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linestyle="-",
            color="black",
            linewidth=2.0,  # 線幅を太く（0.02cm以上相当）
            markersize=5,  # マーカーサイズを適度に
        )

        ax.set_ylim((0, 105))
        ax.set_xlim(left=0)
        ax.set_xlabel("Generation", fontsize=28)
        ax.set_ylabel("Optimal Path Selection Ratio [%]", fontsize=28)

        # 論文標準の軸設定（箱型：全ての枠線を表示）
        ax.spines["top"].set_visible(True)  # 上枠線を表示
        ax.spines["right"].set_visible(True)  # 右枠線を表示
        ax.spines["left"].set_visible(True)  # 左枠線を表示
        ax.spines["bottom"].set_visible(True)  # 下枠線を表示

        # 全ての枠線を黒色、適切な線幅に設定
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.5)  # 枠線の線幅

        # 目盛りの設定
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=24,  # 目盛りラベルのフォントサイズ
            direction="out",  # 目盛りを外向きに
            length=6,  # 主目盛りの長さ
            width=1.5,  # 目盛り線の太さ
            color="black",
        )

        # 副目盛りの設定
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=3,  # 副目盛りの長さ（主目盛りより短く）
            width=1.0,  # 副目盛り線の太さ
            color="black",
        )

        # 副目盛りを有効化
        ax.minorticks_on()

        # 保存
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, format="svg")
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
        各最適解への到達率を積み上げ棒グラフで表示

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

        num_simulations = len(ant_logs)
        num_optimal_solutions = len(optimal_solutions)

        # 世代ごとの各最適解への到達率を計算
        if num_ants == 1:
            # ANT_NUM = 1 の場合
            if not ant_logs[0]:
                return
            num_generations = len(ant_logs[0])

            # 各世代、各最適解への到達率
            generation_rates = []
            for gen_idx in range(num_generations):
                # 各最適解への到達数をカウント
                optimal_counts = [0] * num_optimal_solutions
                for sim_log in ant_logs:
                    optimal_index = sim_log[gen_idx]
                    if optimal_index >= 0:  # 最適解に到達
                        optimal_counts[optimal_index] += 1

                # 到達率に変換（%）
                rates = [
                    float((count / num_simulations) * 100) for count in optimal_counts
                ]
                generation_rates.append(rates)
        else:
            # ANT_NUM > 1 の場合（チャンク処理）
            if not ant_logs[0]:
                return
            total_log_entries = len(ant_logs[0])
            num_generations = total_log_entries // num_ants

            generation_rates = []
            for gen_idx in range(num_generations):
                # 各最適解への到達数をカウント
                # 1世代で複数のアリが同じ最適解に到達している場合、その到達数をカウント
                optimal_counts = [0] * num_optimal_solutions
                for sim_log in ant_logs:
                    start_index = gen_idx * num_ants
                    end_index = start_index + num_ants
                    generation_chunk = sim_log[start_index:end_index]

                    # その世代のチャンク内で、各最適解に到達したアリの数をカウント
                    # 例: [0, 0, 1, -1, 0, 2, -1, 1, 0, -1]
                    #     → 最適解0: 4回, 最適解1: 2回, 最適解2: 1回
                    for opt_idx in range(num_optimal_solutions):
                        count = generation_chunk.count(opt_idx)
                        optimal_counts[opt_idx] += count

                # 到達率に変換（%）
                # 1世代でnum_ants匹のアリがゴールするので、到達率 = (到達数 / (num_simulations * num_ants)) * 100
                total_ants = num_simulations * num_ants
                rates = [float((count / total_ants) * 100) for count in optimal_counts]
                generation_rates.append(rates)

        if not generation_rates:
            print("Warning: No generation_rates calculated")
            return

        # 積み上げ棒グラフを描画
        fig, ax = plt.subplots(figsize=(12, 7))

        generations = list(range(len(generation_rates)))

        # 各最適解のラベルを生成（英語形式）
        optimal_labels = []
        for idx, opt in enumerate(optimal_solutions):
            b, d, h = opt
            # 凡例: "Solution 1: Bandwidth=140.00 Mbps, Delay=16.25 ms, Hops=3"
            label = (
                f"Solution {idx+1}: Bandwidth={b:.0f} Mbps, Delay={d:.0f} ms, Hops={h}"
            )
            optimal_labels.append(label)

        # 積み上げ棒グラフのデータを準備
        # 各最適解のデータを転置（世代ごとの値のリスト）
        stacked_data = []
        for opt_idx in range(num_optimal_solutions):
            opt_rates = [gen_rates[opt_idx] for gen_rates in generation_rates]
            stacked_data.append(opt_rates)

        # 積み上げ棒グラフを描画
        bottom = [0.0] * len(generations)
        colors = plt.cm.tab10(range(num_optimal_solutions))

        for opt_idx in range(num_optimal_solutions):
            ax.bar(
                generations,
                stacked_data[opt_idx],
                bottom=bottom,
                label=optimal_labels[opt_idx],
                color=colors[opt_idx],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            # 次の積み上げのためのbottomを更新
            bottom = [
                bottom[i] + stacked_data[opt_idx][i] for i in range(len(generations))
            ]

        ax.set_xlabel("Generation", fontsize=28)
        ax.set_ylabel("Optimal Solution Selection Rate [%]", fontsize=28)
        ax.set_ylim(0, 105)
        ax.set_xlim(left=-0.5, right=len(generations) - 0.5)
        ax.legend(loc="upper left", fontsize=12, ncol=min(3, num_optimal_solutions))

        # 論文標準の軸設定（箱型）
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.5)

        # 目盛りの設定
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=24,
            direction="out",
            length=6,
            width=1.5,
            color="black",
        )

        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=3,
            width=1.0,
            color="black",
        )

        ax.minorticks_on()

        # 保存
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, format="svg")
        plt.close()
        print(f"Saved: {output_path}")
