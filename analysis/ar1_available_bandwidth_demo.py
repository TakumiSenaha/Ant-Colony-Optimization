import csv
import math
import os
import random
from typing import List, Tuple

MEAN_UTILIZATION: float = 0.4  # (根拠: ISPの一般的な運用マージン)
AR_COEFFICIENT: float = 0.95  # (根拠: ネットワークトラフィックの高い自己相関)
NOISE_VARIANCE: float = 0.000975  # (根拠: 上記2値から逆算した値)


def map_utilization_to_bandwidth(utilization: float, capacity: int = 100) -> int:
    """
    標準的な可用帯域計算: キャパシティ × (1 - 使用率)
    """
    bandwidth = int(round(capacity * (1.0 - utilization)))
    # 10Mbps刻みに丸め
    bandwidth = ((bandwidth + 5) // 10) * 10
    return bandwidth


def map_utilization_with_capacity(utilization: float, capacity_mbps: int) -> int:
    """
    標準的な可用帯域計算: キャパシティ × (1 - 使用率)
    """
    bandwidth = int(round(capacity_mbps * (1.0 - utilization)))
    # 10Mbps刻みに丸め
    bandwidth = ((bandwidth + 5) // 10) * 10
    return bandwidth


def simulate_ar1_series(
    generations: int, seed: int = 0, initial_utilization: float | None = None
) -> Tuple[List[float], List[int]]:
    random.seed(seed)
    utilizations: List[float] = []
    bandwidths: List[int] = []

    # 初期利用率を設定
    current_util = (
        initial_utilization
        if initial_utilization is not None
        else random.uniform(0.3, 0.5)
    )

    for _ in range(generations):
        # AR(1): X_t = (1-phi)*mu + phi*X_{t-1} + epsilon
        noise = random.gauss(0.0, math.sqrt(NOISE_VARIANCE))
        new_util = (
            (1 - AR_COEFFICIENT) * MEAN_UTILIZATION
            + AR_COEFFICIENT * current_util
            + noise
        )
        new_util = max(0.05, min(0.95, new_util))
        current_util = new_util

        bw = map_utilization_to_bandwidth(current_util, 100)
        utilizations.append(current_util)
        bandwidths.append(bw)

    return utilizations, bandwidths


def main() -> None:
    generations = 200
    seed = 0

    utilizations, bandwidths_abs = simulate_ar1_series(generations, seed)

    # キャパシティ依存バージョン（cap=100, 50, 20）を同一利用率系列から生成
    bw_cap100 = [map_utilization_with_capacity(u, 100) for u in utilizations]
    bw_cap50 = [map_utilization_with_capacity(u, 50) for u in utilizations]
    bw_cap20 = [map_utilization_with_capacity(u, 20) for u in utilizations]

    # CSVに保存
    out_csv = os.path.join(
        os.path.dirname(__file__), "..", "simulation_result", "ar1_demo.csv"
    )
    out_csv = os.path.abspath(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation",
                "utilization",
                "bw_cap100",
                "bw_cap50",
                "bw_cap20",
            ]
        )
        for i, u in enumerate(utilizations):
            writer.writerow([i, f"{u:.4f}", bw_cap100[i], bw_cap50[i], bw_cap20[i]])

    # 可能ならPNGで保存
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        # 上段: 現行方式（常に10〜100）
        ax_top.plot(
            range(generations), bandwidths_abs, label="ABS 10-100", color="tab:blue"
        )
        ax_top.set_ylim(0, 110)
        ax_top.set_ylabel("Bandwidth (Mbps)")
        ax_top.legend(loc="upper right")

        # 下段: キャパシティ依存（cap=100 / 50）
        ax_bottom.plot(
            range(generations), bw_cap100, label="Capacity 100", color="tab:green"
        )
        ax_bottom.plot(
            range(generations), bw_cap50, label="Capacity 50", color="tab:red"
        )
        ax_bottom.set_ylim(0, 110)
        ax_bottom.set_ylabel("Bandwidth (Mbps)")
        ax_bottom.set_xlabel("t")
        ax_bottom.legend(loc="upper right")

        fig.tight_layout(h_pad=1.2)
        out_png = os.path.join(os.path.dirname(out_csv), "ar1_demo.png")
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        # matplotlibがない場合はPNG生成をスキップ
        pass


if __name__ == "__main__":
    main()
