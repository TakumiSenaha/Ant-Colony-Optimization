import matplotlib.pyplot as plt
import numpy as np


# 関数を定義
def pheromone_rate(width):
    return 0.99 * (0.8 ** ((100 - width) / 10))


# 0から100までの幅値の配列を作成
width_values = np.linspace(0, 100, 500)
# 対応するフェロモンレートを計算
rate_values = pheromone_rate(width_values)

# グラフを描画
plt.figure(figsize=(10, 6))
plt.plot(width_values, rate_values, label="rate")
plt.grid(False)  # グリッドをオフ
plt.legend()

# 縦軸と横軸の囲いを表示
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)

# フォントサイズを2倍に設定
ax.tick_params(axis="both", which="major", labelsize=20)

plt.show()
