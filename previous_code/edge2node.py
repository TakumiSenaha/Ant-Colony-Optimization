# ネットワークモデル作成手順
# 1. edge_listを生成
# 2. nodeの数だけnode_listにappend
# 3. edge_listの情報を基にnodeのconnection属性を生成
import math
import random
import numpy as np

M=10 # フェロモン最小値

node_list=[] # Nodeオブジェクト格納リスト
edge_list=[(0, 1), (1, 2), (1, 5),(2, 3), (2, 4), (2, 5)]

class Node():
  def __init__(self,connection=np.empty([0,3],dtype=int)):
    # 経路表として[接続先ノード,フェロモン,帯域幅]のセットを二次元配列で保持
    self.connection = connection


def edge2node(edge_list,node_list):
  # edge_listからnodeのconnection属性を構成
  for edge in edge_list:
    before_node = node_list[edge[0]]
    after_node = node_list[edge[1]]
    width = random.randint(1,100)
    before_node.connection = np.append(before_node.connection , np.array([[edge[1],M,width]]) , axis=0)
    after_node.connection = np.append(after_node.connection , np.array([[edge[0],M,width]]) , axis=0)

for _ in range(6):
  node_list.append(Node())

edge2node(edge_list, node_list)

print(node_list[0].connection)
print(node_list[1].connection)
