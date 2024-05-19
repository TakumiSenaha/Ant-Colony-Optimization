import psycopg2
import traceback
import pprint
import matplotlib.pyplot as plt

try:
    # データベースに接続
    conn = psycopg2.connect(
        dbname="test_db",
        user="user",
        password="password",
        host="localhost",
        port="5432"
    )
    # カーソルを作成
    cur = conn.cursor()

    parameter_id = 1
    generation_limit = 500
    width_step = 10
    max_width = 100

    # SQLクエリを一括で実行
    cur.execute(f"""
        SELECT generations.generation_count, ants.routebottleneck, COUNT(ants.routebottleneck)
        FROM ants
        JOIN generations ON ants.generationid = generations.generationid
        JOIN simulations ON generations.simulationid = simulations.simulationid
        WHERE simulations.parameterid = {parameter_id}
        GROUP BY generations.generation_count, ants.routebottleneck
        ORDER BY generations.generation_count, ants.routebottleneck;
    """)

    # 結果を取得
    rows = cur.fetchall()

    # 二次元配列を初期化
    width_counts_matrix = [[0] * (max_width // width_step + 1) for _ in range(generation_limit)]

    # 結果を二次元配列に格納
    for generation, bottleneck, count in rows:
        width_counts_matrix[generation][(max_width - bottleneck) // width_step] = count

    pprint.pprint(width_counts_matrix)

    # 縦列→世代(昇順)、横行→widthを降順(100,90,80...0)、要素→その世代におけるそのwidthの割合
    proportions = [[0] * (max_width // width_step + 1) for _ in range(generation_limit)]
    for generation, width_count_list in enumerate(width_counts_matrix):
        total = sum(width_count_list)
        if total > 0:
            for i, width in enumerate(width_count_list):
                proportions[generation][i] = width * 100 / total

    pprint.pprint(proportions)

    # 転置する
    # 縦列→widthを降順(100,90,80...0)、横行→世代(昇順)、要素→その世代におけるそのwidthの割合
    transpose = list(map(list, zip(*proportions)))

    # グラフ描写
    # 棒グラフの棒のカラー
    color = ['#4F71BE', '#DE8344', '#A5A5A5', '#F1C242', '#6A99D0',
             '#7EAB54', '#2D4374', '#934D21', '#636363', '#937424', '#355D8D']

    # 表示するラベルの用意(世代数)
    labels = list(range(len(transpose[0])))

    # bottomの準備(積み上げ用の変数)
    bottom = [0] * len(transpose[0])

    # プロットする色用のカウンタ
    color_count = 0

    for row in transpose:
        plt.bar(labels, row, width=1.0, bottom=bottom,
                color=color[color_count])
        bottom = [sum(x) for x in zip(bottom, row)]
        color_count += 1

    # グラフの設定
    plt.ylim((0, 100))
    plt.xlabel('Search Count')
    plt.ylabel('Percentage')
    plt.savefig("test.SVG")
    plt.show()

except Exception as e:
    print(e)
    conn.rollback()
    print(traceback.format_exc())

finally:
    # カーソルと接続を閉じる
    cur.close()
    conn.close()
