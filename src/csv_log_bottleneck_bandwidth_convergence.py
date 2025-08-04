import csv
import os

import japanize_matplotlib
import matplotlib.pyplot as plt

# ===== 設定 =====
log_file = "./simulation_result/log_ant.csv"
export_image_name = "./simulation_result/bottleneck_bandwidth_convergence.pdf"
ANT_NUM = 10  # 1世代あたりのアリの数
GENERATION = 1000  # 総世代数


def read_bottleneck_bandwidth_log(file_path, ant_num, generation):
    """
    1回分のボトルネック帯域幅ログ（1行）を読み込み、世代ごとに平均値・最大値を返す
    """
    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        return [], []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                values = [float(val) for val in row]
                avg_list = []
                max_list = []
                for i in range(generation):
                    chunk = values[i * ant_num : (i + 1) * ant_num]
                    avg_list.append(sum(chunk) / ant_num)
                    max_list.append(max(chunk))
                return avg_list, max_list
    print(f"データが空です: {file_path}")
    return [], []


def plot_bottleneck_bandwidth_convergence(avg_bandwidths, max_bandwidths, output_file):
    x_values = list(range(len(avg_bandwidths)))
    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values,
        avg_bandwidths,
        marker="o",
        linestyle="-",
        color="dimgray",
        label="Average among 10 ants",
    )
    plt.plot(
        x_values,
        max_bandwidths,
        marker="s",
        linestyle="--",
        color="black",
        label="Maximum among 10 ants",
    )
    plt.xlabel("Generation", fontsize=20)
    plt.ylabel("Bottleneck Bandwidth (Mbps)", fontsize=20)
    plt.ylim(0, 105)
    plt.xlim(left=0)
    plt.legend(fontsize=15, loc="lower right")
    ax = plt.gca()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.5)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_frame_on(True)
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"グラフを {output_file} にPDF形式で保存しました。")
    plt.show()


if __name__ == "__main__":
    avg_bandwidths, max_bandwidths = read_bottleneck_bandwidth_log(
        log_file, ANT_NUM, GENERATION
    )
    if avg_bandwidths and max_bandwidths:
        plot_bottleneck_bandwidth_convergence(
            avg_bandwidths, max_bandwidths, export_image_name
        )
