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
    num_generations = 100
    width_ranges = range(0, 101, 10)
    width_counts_matrix = [[0] * len(width_ranges) for _ in range(num_generations)]

    # 各世代のwidthの出現回数を一度に取得
    cur.execute(f"""
        SELECT generations.generation_count, interests.routebottleneck, COUNT(*)
        FROM interests
        JOIN generations ON interests.generationid = generations.generationid
        JOIN simulations ON generations.simulationid = simulations.simulationid
        WHERE simulations.parameterid = {parameter_id}
        GROUP BY generations.generation_count, interests.routebottleneck;
    """)
    
    rows = cur.fetchall()

    # 結果を二次元配列に格納
    for row in rows:
        generation = row[0]
        bottleneck = row[1]
        count = row[2]
        width_counts_matrix[generation][10 - bottleneck // 10] = count

    pprint.pprint(width_counts_matrix)

    # 各世代の割合を計算
    proportions = [[0] * len(width_ranges) for _ in range(num_generations)]
    for generation, width_count_list in enumerate(width_counts_matrix):
        total = sum(width_count_list)
        for i, width in enumerate(width_count_list):
            proportions[generation][i] = width * 100 / total if total > 0 else 0

    pprint.pprint(proportions)

    # 転置する
    transpose = list(map(list, zip(*proportions)))

    # グラフ描写
    color = ['#4F71BE', '#DE8344', '#A5A5A5', '#F1C242', '#6A99D0',
             '#7EAB54', '#2D4374', '#934D21', '#636363', '#937424', '#355D8D']
    
    labels = list(range(len(transpose[0])))
    bottom = [0] * len(transpose[0])
    color_count = 0

    for row in transpose:
        plt.bar(labels, row, width=1.0, bottom=bottom, color=color[color_count])
        bottom = [sum(x) for x in zip(bottom, row)]
        color_count += 1

    # グラフの設定
    plt.ylim((0, 100))
    plt.xlabel('Generation')
    plt.ylabel('Percentage')
    plt.title('Width Distribution by Generation')
    plt.savefig("test.SVG")
    plt.show()

except Exception as e:
    print(e)
    conn.rollback()
    print(traceback.format_exc())

finally:
    cur.close()
    conn.close()
