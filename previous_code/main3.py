# logの実装
# グラフの表示を実装
# テーブルに目的地があるかどうかの判定は未実装
import math
import random
import numpy as np
import matplotlib.pyplot as plt

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

class Ant():
  def __init__(self,current,destination,route,minwidth):
    self.current = current # 現在のノード
    self.destination = destination # 目的ノード(コンテンツ保持ノード)
    self.route = route # 辿ってきた経路の配列
    self.minwidth = minwidth # 辿ってきた経路の最小帯域

class Interest():
  def __init__(self,current,destination,route,minwidth):
    self.current = current # 現在のノード
    self.destination = destination # 目的ノード(コンテンツ保持ノード)
    self.route = route # 辿ってきた経路の配列
    self.minwidth = minwidth # 辿ってきた経路の最小帯域

#---------------------------------------------------

def create_ba(node_list,N):
  # baモデルを作成する関数
  # 事前準備　２つのノードからなる完全グラフを用意
  node_list.append(Node(np.array[[1,M,10]]))
  node_list.append(Node(np.array[[0,M,10]]))

  for _ in range(N):
    candidacy_list=[] # 接続先ノードの候補リスト

    for i in range(len(node_list)):
      # 次数(接続ノードの数)の分だけ接続先ノード候補リストにノード番号を追加
      tmp_list=[i]*len(node_list[i].connection)
      candidacy_list.extend(tmp_list)

    # 接続先ノードをランダムに選択
    select_node = random.choice(candidacy_list)
    # 帯域幅をランダムに決定
    width=random.randint(1,10)
    # 接続先ノードのconnection属性に新規ノード追加
    node_list[select_node].connection == np.append(node_list[select_node].connection, [len(node_list),M,width], axis=0)
    # 新規ノードをnode_listに追加
    node_list.append(Node(np.array[[select_node,M,width]]))

def volatilize(node_list,V):
  # node_listの全nodeのフェロモンをV倍する関数 フェロモンの揮発に相当
  for node in node_list:
    for j in range(len(node.connection)):
      new_pheronone=math.floor(node.connection[j][1]*0.9)
      # 最小フェロモン量を下回らないように
      if new_pheronone <= M:
        node.connection[j][1]=M
      else:
        node.connection[j][1]=new_pheronone

def update_pheromone(ant,node_list):
  # 目的ノードに到着したantによるフェロモンの付加(片側)
  for i in range(1,len(ant.route)):
    # ant.routeのi-1番目とi番目のノードを取得
    before_node = node_list[ant.route[i-1]]
    after_node = node_list[ant.route[i]]

    # ant.routeのi-1番目からi番目のedgeのフェロモン値を変更
    # bofore_nodeのconnectionの0列目を取得
    before_line0 = before_node.connection[:,0]
    # before_nodeのconnectionからafter_node番号の行を探索
    row = np.where(before_line0 == ant.route[i])[0][0]
    # i-1番ノードからi番ノードのフェロモン値を最小帯域を元に変更
    before_node.connection[row][1] += (F * ant.minwidth)


def ant_next_node(ant_list,node_list):
  # antの次のノードを決定
  # 繰り返し中にリストから削除を行うためreversed
  for ant in reversed(ant_list):
    candidacy_pheromones=[]
    # antが今いるノードの接続ノードとフェロモン値を取得
    line0 = node_list[ant.current].connection[:,0]
    line1 = node_list[ant.current].connection[:,1]
    # 接続ノードの内、antが辿っていないノード番号を取得
    and_set=set(ant.route) & set(line0)
    diff_list=list(set(line0) ^ and_set)

    # 候補先がないなら削除
    if diff_list==[]:
      ant_list.remove(ant)
      print("Can't Find Route " + str(ant.route))

    else:
      # 蟻が辿っていないノード番号のフェロモンを取得
      for i in diff_list:
        # diff_listの要素がline0の何行目か取得
        diff_row = np.where(line0 == i)[0][0]
        # diff_listの要素のフェロモン値をcandidacy_pheromonsにappend
        candidacy_pheromones.append(line1[diff_row])

      # 次のノード番号をl1の重みづけでl0からランダムに選択
      next_node=random.choices(diff_list,k=1,weights=candidacy_pheromones)[0]
      # 次のノード番号がconnect1列目の何行目か探索
      row = np.where(line0 == next_node)[0][0]

      # antの属性更新
      # もし現在ノードから次ノードの帯域幅が今までの最小帯域より小さかったら更新
      if node_list[ant.current].connection[row][2] < ant.minwidth:
        ant.minwidth = node_list[ant.current].connection[row][2]
      ant.current = node_list[ant.current].connection[row][0]
      ant.route.append(next_node)

    # 蟻が目的ノードならばノードにフェロモンの付加後ant_listから削除
      if ant.current==ant.destination:
        update_pheromone(ant,node_list)
        ant_list.remove(ant)
        print("Goal!")
        print(ant.route)
        print(ant.minwidth)

    # 蟻がTTLならばant_listから削除
      if (len(ant.route)==TTL):
        ant_list.remove(ant)
        print(ant.route)

def interest_next_node(interest_list,node_list,interest_log,loop):
  # interestの次のノードを決定
  # 繰り返し中にリストから削除を行うためreversed
  for interest in reversed(interest_list):
    candidacy_pheromones=[]
    # interestが今いるノードの接続ノードとフェロモン値を取得
    line0 = node_list[interest.current].connection[:,0]
    line1 = node_list[interest.current].connection[:,1]
    # 接続ノードの内、interestが辿っていないノード番号を取得
    and_set=set(interest.route) & set(line0)
    diff_list=list(set(line0) ^ and_set)

    # 候補先がないなら削除
    if diff_list==[]:
      interest_list.remove(interest)
      

    else:
      # interestが辿っていないノード番号のフェロモンを取得
      for i in diff_list:
        # diff_listの要素がline0の何行目か取得
        diff_row = np.where(line0 == i)[0][0]
        # diff_listの要素のフェロモン値をcandidacy_pheromonsにappend
        candidacy_pheromones.append(line1[diff_row])

      # 次のノード番号をl1の重みづけでl0からランダムに選択
      next_node=random.choices(diff_list,k=1,weights=candidacy_pheromones)[0]
      # 次のノード番号がconnect1列目の何行目か探索
      row = np.where(line0 == next_node)[0][0]

      # interestの属性更新
      # もし現在ノードから次ノードの帯域幅が今までの最小帯域より小さかったら更新
      if node_list[interest.current].connection[row][2] < interest.minwidth:
        interest.minwidth = node_list[interest.current].connection[row][2]
      interest.current = node_list[interest.current].connection[row][0]
      interest.route.append(next_node)

    # interestが目的ノードならばinterest_listから削除
      if interest.current==interest.destination:
        interest_list.remove(interest)
        interest_log[loop].append(interest.minwidth)

    # interestがTTLならばinterest_listから削除
      if (len(interest.route)==TTL):
        interest_list.remove(interest)
        

def show_node_info(node_list):
  for i in range(len(node_list)):
    print(node_list[i].connection)

def edge2node(edge_list,node_list):
  # edge_listからnodeのconnection属性を作成
  # 事前にnode_listに無垢のNodeインスタンスをNodeの数だけ準備しておく必要あり
  for edge in edge_list:
    before_node = node_list[edge[0]]
    after_node = node_list[edge[1]]
    # 帯域幅をランダムに作成
    width = 10
    # nodeのconnection属性に情報を追加
    before_node.connection = np.append(before_node.connection , np.array([[edge[1],M,width]]) , axis=0)
    after_node.connection = np.append(after_node.connection , np.array([[edge[0],M,width]]) , axis=0)


#---------------------------------------------------

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

  for loop in range(100):
    interest_log.append([])

    for _ in range(10):
      ant_list.append(Ant(0,15,[0],W))
      
    for _ in range(TTL):
      print("length of ant_list = "+str(len(ant_list)))
      ant_next_node(ant_list, node_list)

    for _ in range(50):
      interest_list.append(Interest(0,15,[0],W))
    
    for _ in range(20):
      interest_next_node(interest_list, node_list, interest_log, loop)

print()
print("----------------------End Loop------------------------------")
print()

summary_interest_log=[]
for i in range(len(interest_log)):
  tmp_list=[i,len(interest_log[i]),round(sum(interest_log[i])/len(interest_log[i]),1)]
  summary_interest_log.append(tmp_list)
# print(summary_interest_log)

show_node_info(node_list)

x=[]
reach_rate=[]
ave_wid=[]
for i in range(len(interest_log)):
  x.append(i*10)
  reach_rate.append((len(interest_log[i])/50)*100)
  ave_wid.append(round(sum(interest_log[i])/len(interest_log[i]),1))


plt.scatter(x,ave_wid)
plt.ylim(0,100)
plt.xlabel("The number of released ant")
plt.ylabel("Average maximum capacity path of reaching interest")
plt.show()

plt.scatter(x,reach_rate)
plt.ylim(0,100)
plt.xlabel("The number of released ant")
plt.ylabel("Probability of reaching interest [%]")
plt.show()
