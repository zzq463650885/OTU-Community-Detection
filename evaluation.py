import numpy as np
import networkx as nx

# 功能：计算预测划分得到的图模块度
# 输入：有序的nx.graph, 有序的y_pred
# 输出: modularity
def modularity(G, y_pred):
    V = [ node for node in G.nodes()]
    m = G.size(weight='weight')  
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    Q = 0.0
    dist = { i:0 for i in range(8) }
    for i in range(n):
        dist[ y_pred[i] ] += 1       
    print('node community distribute:{}'.format(dist))
    for i in range(n):
        print('\rmodularity running node:{}'.format(i),end='')
        node_i = V[i]                     
        if i==233:
            print(' i:{},node_i:{}'.format(i, node_i))
        com_i = y_pred[int(node_i)]            # node->com_i
        degree_i = G.degree(node_i)
        for j in range(n):
            node_j = V[j]
            com_j = y_pred[int(node_j) ]
            if com_i != com_j:
                continue
            degree_j = G.degree(node_j)
            Q += A[i][j] - degree_i * degree_j/(2 * m)  
    print('\nnumber of edgs:{}, modularity result:{}'.format(G.number_of_edges(), Q))
    return Q/(2 * m)
