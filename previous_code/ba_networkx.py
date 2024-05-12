import matplotlib.pyplot as plt
import networkx as nx
import random
__TRAINING_COUNT__ = 100

node = [x for x in range(__TRAINING_COUNT__)]
degree= [1,1] # 先頭から各ノードの次数を格納
edge=[(0,1)]  # タプルでノードとノードの繋がりを格納

for i in range(__TRAINING_COUNT__ ):

    # sumの２つ目の引数[]でリストの連結を指示
    # i番目のノードの次数はp個.iをp個生成し配列に格納
    board = sum([[i] * p for i,p in enumerate(degree)], []) 

    # boardの中からランダムに接続先を選ぶことで次数の高いノードを確立的に選ぶ
    # select_nodeには選ばれたノード番号が格納
    select_node = random.choice(board)

    # 選ばれたノードの次数を1つ追加
    degree[select_node] = degree[select_node] + 1

    # degreeに新たなノードの次数である1をappend
    degree.append(1)

    # 選ばれたselect_nodeと新たなノードの番号をタプルでedgeにappend
    edge.append((select_node, len(degree) - 1))

print("degree:{}, edge:{}".format(degree, edge))

# Graphオブジェクトの作成
G = nx.Graph()
 
# nodeデータの追加
G.add_nodes_from(node)
 
# edgeデータの追加
G.add_edges_from(edge)
 
# ネットワークの可視化
nx.draw(G, with_labels = True)
plt.show()
