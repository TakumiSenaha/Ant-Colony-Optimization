#!/usr/bin/env python3
"""
二段階BKB学習の詳細分析スクリプト

分析内容:
1. 追従率の時系列変化
2. 短期/長期EMAの乖離度
3. 成功率の分布
4. 学習速度の評価
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定（SVG出力のためASCIIエンコード不要）
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    # japanize_matplotlibがない場合は、システムフォントを使用
    plt.rcParams["font.sans-serif"] = [
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

# ===== グラフ描画設定（論文形式） =====
AXIS_LABEL_FONTSIZE = 28  # 軸ラベルのフォントサイズ
TICK_LABEL_FONTSIZE = 24  # 目盛りラベルのフォントサイズ
LEGEND_FONTSIZE = 12  # 凡例のフォントサイズ


def load_detailed_log(filepath: str) -> dict:
    """詳細ログを読み込む"""
    data = {
        "simulation": [],
        "generation": [],
        "optimal_bw": [],
        "goal_ultra_short_bkb": [],
        "goal_short_bkb": [],
        "goal_long_bkb": [],
        "goal_effective_bkb": [],
        "goal_var": [],
        "confidence": [],
        "tracking_rate_ultra_short": [],
        "tracking_rate_short": [],
        "tracking_rate_effective": [],
        "success_rate": [],
    }

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["simulation"].append(int(row["simulation"]))
            data["generation"].append(int(row["generation"]))
            data["optimal_bw"].append(float(row["optimal_bw"]))
            data["goal_ultra_short_bkb"].append(float(row["goal_ultra_short_bkb"]))
            data["goal_short_bkb"].append(float(row["goal_short_bkb"]))
            data["goal_long_bkb"].append(float(row["goal_long_bkb"]))
            data["goal_effective_bkb"].append(float(row["goal_effective_bkb"]))
            data["goal_var"].append(float(row["goal_var"]))
            data["confidence"].append(float(row["confidence"]))
            data["tracking_rate_ultra_short"].append(
                float(row["tracking_rate_ultra_short"])
            )
            data["tracking_rate_short"].append(float(row["tracking_rate_short"]))
            data["tracking_rate_effective"].append(
                float(row["tracking_rate_effective"])
            )
            data["success_rate"].append(float(row["success_rate"]))

    return data


def analyze_overall_performance(data: dict):
    """全体的な性能を分析"""
    print("=" * 80)
    print("📊 三段階BKB学習の全体性能分析")
    print("=" * 80)

    # 追従率の統計
    tracking_effective = np.array(data["tracking_rate_effective"])
    print(f"\n🎯 実効BKB追従率:")
    print(
        f"   平均: {np.mean(tracking_effective):.3f} ({np.mean(tracking_effective)*100:.1f}%)"
    )
    print(
        f"   中央値: {np.median(tracking_effective):.3f} ({np.median(tracking_effective)*100:.1f}%)"
    )
    print(
        f"   最小: {np.min(tracking_effective):.3f} ({np.min(tracking_effective)*100:.1f}%)"
    )
    print(
        f"   最大: {np.max(tracking_effective):.3f} ({np.max(tracking_effective)*100:.1f}%)"
    )
    print(f"   標準偏差: {np.std(tracking_effective):.3f}")

    # 成功率の統計
    success_rate = np.array(data["success_rate"])
    print(f"\n✅ 成功率（直近10世代）:")
    print(f"   平均: {np.mean(success_rate):.3f} ({np.mean(success_rate)*100:.1f}%)")
    print(
        f"   中央値: {np.median(success_rate):.3f} ({np.median(success_rate)*100:.1f}%)"
    )
    print(f"   最大: {np.max(success_rate):.3f} ({np.max(success_rate)*100:.1f}%)")

    # 確信度の統計
    confidence = np.array(data["confidence"])
    print(f"\n🔒 確信度:")
    print(f"   平均: {np.mean(confidence):.3f}")
    print(f"   中央値: {np.median(confidence):.3f}")

    # 短期と長期のギャップ
    short_long_gap = []
    for i in range(len(data["goal_short_bkb"])):
        short = data["goal_short_bkb"][i]
        long = data["goal_long_bkb"][i]
        optimal = data["optimal_bw"][i]
        if optimal > 0:
            gap = abs(short - long) / optimal
            short_long_gap.append(gap)

    print(f"\n📏 短期/長期EMAの乖離度（対最適値比）:")
    print(
        f"   平均: {np.mean(short_long_gap):.3f} ({np.mean(short_long_gap)*100:.1f}%)"
    )
    print(
        f"   中央値: {np.median(short_long_gap):.3f} ({np.median(short_long_gap)*100:.1f}%)"
    )
    print(f"   最大: {np.max(short_long_gap):.3f} ({np.max(short_long_gap)*100:.1f}%)")

    return tracking_effective, success_rate, confidence, short_long_gap


def analyze_by_generation(data: dict):
    """世代別の性能を分析"""
    print("\n" + "=" * 80)
    print("📈 世代別性能分析（初期 vs 中期 vs 後期）")
    print("=" * 80)

    # 世代を3つに分割
    generations = np.array(data["generation"])
    tracking = np.array(data["tracking_rate_effective"])
    success = np.array(data["success_rate"])

    # 初期（0-300世代）
    early_mask = generations < 300
    early_tracking = tracking[early_mask]
    early_success = success[early_mask]

    # 中期（300-700世代）
    mid_mask = (generations >= 300) & (generations < 700)
    mid_tracking = tracking[mid_mask]
    mid_success = success[mid_mask]

    # 後期（700-1000世代）
    late_mask = generations >= 700
    late_tracking = tracking[late_mask]
    late_success = success[late_mask]

    print(f"\n🌱 初期（0-300世代）:")
    print(
        f"   追従率: {np.mean(early_tracking):.3f} ({np.mean(early_tracking)*100:.1f}%)"
    )
    print(
        f"   成功率: {np.mean(early_success):.3f} ({np.mean(early_success)*100:.1f}%)"
    )

    print(f"\n🌿 中期（300-700世代）:")
    print(f"   追従率: {np.mean(mid_tracking):.3f} ({np.mean(mid_tracking)*100:.1f}%)")
    print(f"   成功率: {np.mean(mid_success):.3f} ({np.mean(mid_success)*100:.1f}%)")

    print(f"\n🌳 後期（700-1000世代）:")
    print(
        f"   追従率: {np.mean(late_tracking):.3f} ({np.mean(late_tracking)*100:.1f}%)"
    )
    print(f"   成功率: {np.mean(late_success):.3f} ({np.mean(late_success)*100:.1f}%)")

    # 改善率を計算
    early_tracking_mean = np.mean(early_tracking)
    early_success_mean = np.mean(early_success)
    if early_tracking_mean > 0:
        improvement_tracking = (
            (np.mean(late_tracking) - early_tracking_mean) / early_tracking_mean * 100
        )
    else:
        improvement_tracking = 0.0 if np.mean(late_tracking) == 0 else float("inf")

    if early_success_mean > 0:
        improvement_success = (
            (np.mean(late_success) - early_success_mean) / early_success_mean * 100
        )
    else:
        improvement_success = 0.0 if np.mean(late_success) == 0 else float("inf")

    print(f"\n📊 改善率:")
    if improvement_tracking == float("inf"):
        print(f"   追従率: N/A (初期値が0で後期に改善)")
    else:
        print(f"   追従率: {improvement_tracking:+.1f}%")
    if improvement_success == float("inf"):
        print(f"   成功率: N/A (初期値が0で後期に改善)")
    else:
        print(f"   成功率: {improvement_success:+.1f}%")


def analyze_learning_speed(data: dict):
    """学習速度を分析"""
    print("\n" + "=" * 80)
    print("⚡ 学習速度分析")
    print("=" * 80)

    # 各シミュレーションで追従率が50%に達するまでの世代数
    sim_ids = sorted(set(data["simulation"]))
    generations_to_50 = []

    for sim_id in sim_ids:
        sim_mask = np.array(data["simulation"]) == sim_id
        sim_generations = np.array(data["generation"])[sim_mask]
        sim_tracking = np.array(data["tracking_rate_effective"])[sim_mask]

        # 50%を超えた最初の世代を探す
        over_50 = sim_tracking >= 0.5
        if np.any(over_50):
            first_over_50_idx = np.argmax(over_50)
            generations_to_50.append(sim_generations[first_over_50_idx])
        else:
            generations_to_50.append(1000)  # 到達しなかった

    reached_count = sum(1 for g in generations_to_50 if g < 1000)

    print(f"\n🎯 追従率50%到達:")
    print(f"   到達シミュレーション数: {reached_count}/{len(sim_ids)}")
    if reached_count > 0:
        reached_generations = [g for g in generations_to_50 if g < 1000]
        print(f"   平均到達世代: {np.mean(reached_generations):.0f}世代")
        print(f"   最速到達世代: {np.min(reached_generations):.0f}世代")

    # 成功率が10%に達するまでの世代数
    generations_to_10_success = []

    for sim_id in sim_ids:
        sim_mask = np.array(data["simulation"]) == sim_id
        sim_generations = np.array(data["generation"])[sim_mask]
        sim_success = np.array(data["success_rate"])[sim_mask]

        over_10 = sim_success >= 0.1
        if np.any(over_10):
            first_over_10_idx = np.argmax(over_10)
            generations_to_10_success.append(sim_generations[first_over_10_idx])
        else:
            generations_to_10_success.append(1000)

    success_reached_count = sum(1 for g in generations_to_10_success if g < 1000)

    print(f"\n✅ 成功率10%到達:")
    print(f"   到達シミュレーション数: {success_reached_count}/{len(sim_ids)}")
    if success_reached_count > 0:
        reached_success_generations = [g for g in generations_to_10_success if g < 1000]
        print(f"   平均到達世代: {np.mean(reached_success_generations):.0f}世代")


def create_visualization(data: dict, output_dir: str):
    """可視化を作成"""
    print("\n" + "=" * 80)
    print("📊 グラフ生成中...")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # シミュレーション1の詳細を可視化
    sim1_mask = np.array(data["simulation"]) == 1
    sim1_gen = np.array(data["generation"])[sim1_mask]
    sim1_optimal = np.array(data["optimal_bw"])[sim1_mask]
    sim1_ultra_short = np.array(data["goal_ultra_short_bkb"])[sim1_mask]
    sim1_short = np.array(data["goal_short_bkb"])[sim1_mask]
    sim1_long = np.array(data["goal_long_bkb"])[sim1_mask]
    sim1_effective = np.array(data["goal_effective_bkb"])[sim1_mask]
    sim1_tracking = np.array(data["tracking_rate_effective"])[sim1_mask]
    sim1_success = np.array(data["success_rate"])[sim1_mask]

    # グラフ1: 帯域値の推移（シミュレーション1）
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(sim1_gen, sim1_optimal, "k-", linewidth=2.0, label="Optimal BKB", alpha=0.7)
    ax.plot(
        sim1_gen,
        sim1_ultra_short,
        "orange",
        linewidth=2.0,
        label="Ultra-Short EMA (α=0.9)",
        alpha=0.8,
    )
    ax.plot(
        sim1_gen, sim1_short, "r-", linewidth=2.0, label="Short EMA (α=0.5)", alpha=0.8
    )
    ax.plot(
        sim1_gen, sim1_long, "b-", linewidth=2.0, label="Long EMA (α=0.125)", alpha=0.8
    )
    ax.plot(
        sim1_gen, sim1_effective, "g--", linewidth=2.0, label="Effective BKB", alpha=0.8
    )
    ax.set_xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Bottleneck Bandwidth [Mbps]", fontsize=AXIS_LABEL_FONTSIZE)

    # 論文標準の軸設定（箱型：全ての枠線を表示）
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        ncol=2,
    )
    ax.grid(True, alpha=0.3)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONTSIZE,
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

    plt.tight_layout()
    output_path = output_dir / "three_phase_learning_sim1_bandwidth.svg"
    plt.savefig(output_path, format="svg")
    print(f"✅ 保存: {output_path}")
    plt.show()

    # グラフ2: 追従率（シミュレーション1）
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(sim1_gen, sim1_tracking * 100, "g-", linewidth=2.0, label="Tracking Rate")
    ax.axhline(
        y=50,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="50% Target",
        alpha=0.7,
    )
    ax.axhline(
        y=80, color="red", linestyle="--", linewidth=1.5, label="80% Target", alpha=0.7
    )
    ax.set_xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Tracking Rate [%]", fontsize=AXIS_LABEL_FONTSIZE)

    # 論文標準の軸設定
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONTSIZE,
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

    plt.tight_layout()
    output_path = output_dir / "three_phase_learning_sim1_tracking.svg"
    plt.savefig(output_path, format="svg")
    print(f"✅ 保存: {output_path}")
    plt.show()

    # グラフ3: 成功率（シミュレーション1）
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(
        sim1_gen,
        sim1_success * 100,
        "m-",
        linewidth=2.0,
        label="Success Rate (Recent 10 gens)",
    )
    ax.axhline(
        y=10,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="10% Target",
        alpha=0.7,
    )
    ax.set_xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Success Rate [%]", fontsize=AXIS_LABEL_FONTSIZE)

    # 論文標準の軸設定
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONTSIZE,
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

    plt.tight_layout()
    output_path = output_dir / "three_phase_learning_sim1_success.svg"
    plt.savefig(output_path, format="svg")
    print(f"✅ 保存: {output_path}")
    plt.show()

    # グラフ4: 全シミュレーションの平均追従率
    sim_ids = sorted(set(data["simulation"]))
    generations_unique = sorted(set(data["generation"]))

    tracking_by_gen = {gen: [] for gen in generations_unique}
    success_by_gen = {gen: [] for gen in generations_unique}

    for sim_id in sim_ids:
        sim_mask = np.array(data["simulation"]) == sim_id
        sim_gen = np.array(data["generation"])[sim_mask]
        sim_tracking = np.array(data["tracking_rate_effective"])[sim_mask]
        sim_success = np.array(data["success_rate"])[sim_mask]

        for gen, track, succ in zip(sim_gen, sim_tracking, sim_success):
            tracking_by_gen[gen].append(track)
            success_by_gen[gen].append(succ)

    mean_tracking = [np.mean(tracking_by_gen[g]) * 100 for g in generations_unique]
    std_tracking = [np.std(tracking_by_gen[g]) * 100 for g in generations_unique]
    mean_success = [np.mean(success_by_gen[g]) * 100 for g in generations_unique]
    std_success = [np.std(success_by_gen[g]) * 100 for g in generations_unique]

    # 追従率の平均と標準偏差
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(
        generations_unique,
        mean_tracking,
        "g-",
        linewidth=2.0,
        label="Mean Tracking Rate",
    )
    ax.fill_between(
        generations_unique,
        np.array(mean_tracking) - np.array(std_tracking),
        np.array(mean_tracking) + np.array(std_tracking),
        alpha=0.3,
        color="green",
        label="±1σ",
    )
    ax.axhline(
        y=50,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="50% Target",
        alpha=0.7,
    )
    ax.axhline(
        y=60, color="red", linestyle="--", linewidth=1.5, label="60% Target", alpha=0.7
    )
    ax.set_xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Tracking Rate [%]", fontsize=AXIS_LABEL_FONTSIZE)

    # 論文標準の軸設定
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        ncol=2,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONTSIZE,
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

    plt.tight_layout()
    output_path = output_dir / "three_phase_learning_average_tracking.svg"
    plt.savefig(output_path, format="svg")
    print(f"✅ 保存: {output_path}")
    plt.show()

    # グラフ5: 全シミュレーションの平均成功率
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(
        generations_unique, mean_success, "m-", linewidth=2.0, label="Mean Success Rate"
    )
    ax.fill_between(
        generations_unique,
        np.array(mean_success) - np.array(std_success),
        np.array(mean_success) + np.array(std_success),
        alpha=0.3,
        color="magenta",
        label="±1σ",
    )
    ax.axhline(
        y=10,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="10% Target",
        alpha=0.7,
    )
    ax.set_xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Success Rate [%]", fontsize=AXIS_LABEL_FONTSIZE)

    # 論文標準の軸設定
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 55)
    ax.set_xlim(left=0)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONTSIZE,
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

    plt.tight_layout()
    output_path = output_dir / "three_phase_learning_average_success.svg"
    plt.savefig(output_path, format="svg")
    print(f"✅ 保存: {output_path}")
    plt.show()


def provide_recommendations(data: dict):
    """改善提案を提供"""
    print("\n" + "=" * 80)
    print("💡 改善提案")
    print("=" * 80)

    tracking_mean = np.mean(data["tracking_rate_effective"])
    success_mean = np.mean(data["success_rate"])

    print(f"\n現状:")
    print(f"   平均追従率: {tracking_mean*100:.1f}%")
    print(f"   平均成功率: {success_mean*100:.1f}%")

    print(f"\n🎯 目標:")
    print(f"   追従率: 60% 以上")
    print(f"   成功率: 10% 以上")

    print(f"\n📋 問題点:")
    if tracking_mean < 0.6:
        print(f"   ❌ 追従率が低い（現在{tracking_mean*100:.1f}%、目標60%）")
        print(f"      → 学習速度が環境変化に追いついていない")
    if success_mean < 0.1:
        print(f"   ❌ 成功率が低い（現在{success_mean*100:.1f}%、目標10%）")
        print(f"      → アリが最適経路を選択できていない")

    print(f"\n🔧 改善案:")
    print(f"\n【案1】短期EMAの学習率をさらに上げる")
    print(f"   現在: α_short = 0.5")
    print(f"   提案: α_short = 0.7～0.8")
    print(f"   理由: より急激な変化に素早く追従")

    print(f"\n【案2】実効BKBの計算方法を変更")
    print(f"   現在: max(短期, 長期)")
    print(f"   提案: 重み付き平均")
    print(f"   式: w×短期 + (1-w)×長期 (w=0.7)")
    print(f"   理由: 短期と長期のバランスを調整可能")

    print(f"\n【案3】フェロモン更新の強化")
    print(f"   提案: achievement_bonus を2.0→3.0に増加")
    print(f"   提案: penalty_factor を0.5→0.3に減少")
    print(f"   理由: より強い学習シグナルを提供")

    print(f"\n【案4】揮発率の調整")
    print(f"   現在: V = 0.99（フェロモン）")
    print(f"   提案: V = 0.95～0.90")
    print(f"   理由: 過去の情報をより早く忘れて新環境に適応")

    print(f"\n【案5】三段階学習モデル")
    print(f"   超短期（α=0.9）: 最新の変化を即座に捉える")
    print(f"   短期（α=0.5）: 直近の傾向を把握")
    print(f"   長期（α=0.125）: 安定した基準を保持")
    print(f"   実効BKB = max(超短期, 短期, 長期)")


if __name__ == "__main__":
    log_file = "./simulation_result/log_detailed_tracking_rfc.csv"
    output_dir = "./simulation_result"

    print("\n🚀 三段階BKB学習の詳細分析を開始します...")
    print(f"📁 ログファイル: {log_file}\n")

    # データ読み込み
    data = load_detailed_log(log_file)
    print(f"✅ データ読み込み完了: {len(data['simulation'])}レコード")
    print(f"   シミュレーション数: {len(set(data['simulation']))}")
    print(f"   世代数: {max(data['generation']) + 1}")

    # 分析実行
    tracking_eff, success, confidence, gap = analyze_overall_performance(data)
    analyze_by_generation(data)
    analyze_learning_speed(data)

    # 可視化
    create_visualization(data, output_dir)

    # 改善提案
    provide_recommendations(data)

    print("\n" + "=" * 80)
    print("✅ 分析完了！")
    print("=" * 80)
