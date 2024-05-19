import csv
import matplotlib.pyplot as plt
import pprint

csv_file_name = './simulation_result/log_interest.csv'
export_image_name = './simulation_result/result_previous.SVG'

# ! 結果の集計---------------------------------------------------------------

# CSVファイルを読み込む
with open('./simulation_result/log_interest.csv', 'r') as f:
    reader = csv.reader(f)
    data = [list(map(int, row)) for row in reader]


# 縦列が探索回数(1,2,3...)・横行(0,10,20...100)がその探索回数におけるwidthの出現回数
counts = [[0]*11 for _ in range(len(data[0]))]


# 集計
for row in data:
    for search_count, width in enumerate(row):
        counts[search_count][width//10] += 1


# 横行のwidthの出現回数の総和
totals = [sum(col) for col in counts]


# for col, total in zip(counts, totals):
#   print (f"{col} - {total}")


# 縦列が探索回数・横行がその探索回数におけるwidthの割合
ratios = [[count * 100 / total for count in row]
          for row, total in zip(counts, totals)]
# print(ratios)


# ! グラフ描写---------------------------------------------------------------
# 棒グラフの棒のカラー
color = ['#4F71BE', '#DE8344', '#A5A5A5', '#F1C242', '#6A99D0',
         '#7EAB54', '#2D4374', '#934D21', '#636363', '#937424', '#355D8D']

# データの用意
data = ratios

# データの左右反転
reversed_data = [row[::-1] for row in data]

# 表示するラベルの用意
labels = list(range(len(data)))

# 各行の合計値を計算し、それを元に割合を計算
totals = [sum(row) for row in reversed_data]
proportions = [[val * 100 / total for val in row]
               for total, row in zip(totals, reversed_data)]

print(*reversed_data, end='\n')
print("\n")
print(*proportions, end='\n')


# 積み上げ棒グラフを描画(width = 100 -> width = 10)の順に描写
# 積み上げのために二次元配列を転置
transpose = list(map(list, (zip(*proportions))))
# pprint.pprint(transpose, width=80)
# print(transpose)

bottom = [0] * len(labels)
color_count = 0

for row in transpose:
    plt.bar(labels, row, width=1.0, bottom=bottom, color=color[color_count])
    bottom = [sum(x) for x in zip(bottom, row)]
    color_count += 1


# グラフの設定
plt.ylim((0, 100))
plt.xlabel('Search Count')
plt.ylabel('Percentage')
plt.savefig(export_image_name)
plt.show()
