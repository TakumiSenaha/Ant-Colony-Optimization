import networkx as nx
import matplotlib.pyplot as plt

# Graphオブジェクトの作成
G = nx.MultiDiGraph()
 
# nodeデータの追加
G.add_nodes_from([1, 2, 3, 4, 5, 6])
 
# edgeデータの追加
G.add_edges_from([(0,1),(1,0),(0,4),(1,2),(1,5),(2,3),(2,6),(3,7),(4,5),(4,8),(5,6),(5,9),(6,7),(6,10),(7,11),(8,9),(8,12),(9,10),(9,13),(10,11),(10,14),(11,15),(12,13),(13,14),(14,15)])
 
# ネットワークの可視化
nx.draw(G, with_labels = True)
plt.show()
