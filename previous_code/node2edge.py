import math
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from cairosvg import svg2png

N=10 # 総ノード数(実際は最初に用意する２つのノードで+2)
V=0.8 # フェロモン揮発量
M=10 # フェロモン最小値
F=1 # フェロモン値決定用定数
TTL=8 # 蟻のTime to Live
W=1000 # 帯域幅初期値
node_list=[] # Nodeオブジェクト格納リスト
ant_list=[] # Antオブジェクト格納リスト
interest_list=[] # Interestオブジェクト格納リスト
edge=[] # networkxで表示するためのエッジリスト
interest_log=[] # interestのログ用リスト


class Node():
  def __init__(self,connection=np.empty([0,3],dtype=int)):
    # 経路表として[接続先ノード,フェロモン,帯域幅]のセットを二次元配列で保持
    self.connection = connection

def show_node_info(node_list):
  for i in range(len(node_list)):
    print(node_list[i].connection)

def node2edge(node_list):
  # node_listからネットワークのグラフ表示のためのedgeのリストを返す
  edges=[]
  for i in range(len(node_list)):
    # i番目ノードの接続先を取得
    line0 = node_list[i].connection[:,0]
    # i番目ノードのフェロモン値を取得
    line1 = node_list[i].connection[:,1]
    # i番目ノードの帯域幅を取得
    line2 = node_list[i].connection[:,2]

    sum_line1 = sum(line1)

    for j in range(len(line0)):

      edge=(i,line0[j],{"len":"5.0", "label": str(line2[j])+":"+str(line1[j]), "color": "0.000 " + str(round(math.log(line1[j],1000),3)) + " 1.000", "penwidth":str(line2[j]/20)})
      edges.append(edge)

  return edges



if __name__ == "__main__":

  #4×4のネットワークを生成
  node_list.append(Node(np.array([[1,M,100],[4,M,10]])))
  node_list.append(Node(np.array([[0,M,10],[5,M,100],[2,M,10]])))
  node_list.append(Node(np.array([[1,M,10],[6,M,10],[3,M,10]])))
  node_list.append(Node(np.array([[2,M,10],[7,M,10]])))
  node_list.append(Node(np.array([[0,M,10],[5,M,10],[8,M,10]])))
  node_list.append(Node(np.array([[1,M,10],[4,M,10],[6,M,100],[9,M,10]])))
  node_list.append(Node(np.array([[2,M,10],[5,M,10],[7,M,10],[10,M,100]])))
  node_list.append(Node(np.array([[3,M,10],[6,M,10],[11,M,10]])))
  node_list.append(Node(np.array([[4,M,10],[9,M,10],[12,M,10]])))
  node_list.append(Node(np.array([[5,M,10],[8,M,10],[10,M,10],[13,M,10]])))
  node_list.append(Node(np.array([[6,M,10],[9,M,10],[11,M,100],[14,M,10]])))
  node_list.append(Node(np.array([[7,M,10],[10,M,10],[15,M,100]])))
  node_list.append(Node(np.array([[8,M,10],[13,M,10]])))
  node_list.append(Node(np.array([[9,M,10],[12,M,10],[14,M,10]])))
  node_list.append(Node(np.array([[10,M,10],[13,M,10],[15,M,10]])))
  node_list.append(Node(np.array([[11,M,10],[14,M,10]])))

show_node_info(node_list)
edges = node2edge(node_list)
nodes = [i for i in range(15)]
print(edges)

g = nx.MultiDiGraph() #  グラフの種類

g.add_nodes_from(nodes) #  グラフにノード追加
g.add_edges_from(edges) #  グラフにエッジ追加

agraph = nx.nx_agraph.to_agraph(g)
agraph.node_attr["shape"] = "circle" #  表示方法変更
agraph.draw( "./sample.svg", prog="fdp", format="svg")
# SVGをPNGに変換
svg2png(url="./sample.svg", write_to="./sample.png")

str(round(line1[j]/sum_line1,2))
