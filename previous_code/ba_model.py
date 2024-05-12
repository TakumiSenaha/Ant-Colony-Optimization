import random
__TRAINING_COUNT__ = 100
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
