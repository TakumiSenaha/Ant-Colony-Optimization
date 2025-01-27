import numpy as np
import plotly.graph_objects as go


# 揮発率計算関数
def calculate_rate(weight_uv, local_min_bandwidth, local_max_bandwidth, mode=1):
    if local_max_bandwidth == local_min_bandwidth:
        # 未使用エッジの場合
        rate = 1 - (1 / max(1, weight_uv))
    else:
        if mode == 1:  # モード1: 最小値・最大値基準
            rate = (weight_uv - local_min_bandwidth) / max(
                1, (local_max_bandwidth - local_min_bandwidth)
            )
            rate = 0.99 * rate
        elif mode == 2:  # モード2: 平均・分散基準
            avg_bandwidth = 0.5 * (local_min_bandwidth + local_max_bandwidth)
            std_dev = max(abs(local_max_bandwidth - avg_bandwidth), 1)
            gamma = 1.0
            rate = np.exp(-gamma * (avg_bandwidth - weight_uv) / std_dev)
    return np.clip(rate, 0, 1)  # 必ず [0, 1] の範囲に収める


# 3Dグラフのプロット
def plot_3d_rate(mode=1):
    weight_uv = np.linspace(1, 100, 50)  # エッジ帯域幅
    local_min_bandwidth = 10
    local_max_bandwidth_range = np.linspace(20, 100, 50)
    X, Y = np.meshgrid(weight_uv, local_max_bandwidth_range)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = calculate_rate(X[i, j], local_min_bandwidth, Y[i, j], mode)

    fig = go.Figure(
        data=[
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                hovertemplate=(
                    "weight_uv (帯域幅): %{x:.2f}<br>"
                    "local_min_bandwidth (最小帯域幅): 10.00<br>"  # 固定値
                    "local_max_bandwidth (最大帯域幅): %{y:.2f}<br>"
                    "rate (揮発率): %{z:.4f}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Rate Distribution (Mode {mode})",
        scene=dict(
            xaxis=dict(title="weight_uv (帯域幅)", range=[1, 100]),
            yaxis=dict(title="local_max_bandwidth (最大帯域幅)", range=[20, 100]),
            zaxis=dict(title="rate (揮発率)", range=[0, 1]),
        ),
    )
    fig.show()


# メイン処理
if __name__ == "__main__":
    print("3Dグラフを表示中...")
    plot_3d_rate(mode=1)  # Mode 1 の3Dグラフ
