import numpy as np
import networkx as nx
from tqdm import tqdm

# 功能：利用矩阵对称性作下三角优化处理的图模块度计算函数
# 输入：有序的nx.graph, 有序的y_pred
# 输出: modularity
def symmetric_matrix_modularity(G, y_pred):
    V = [ node for node in G.nodes()]
    m = G.size(weight='weight')  
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    Q = 0.0
    print('number of edges:{}'.format(m))
    print('i:{},node_i:{}'.format(233, V[233] ))
    for i in tqdm(range(n)):
        # print('\rmodularity running node:{},now Q:{:8.8f}'.format(i,Q),end='')
        node_i = V[i]                     
        com_i = y_pred[int(node_i)]                         # node->com_i
        degree_i = G.degree(node_i)
        Q += ( 1.0 - degree_i * degree_i/(2 * m) ) / 2.0    # actually exits self-loops in dpwk graph
        for j in range(i):                                  # [0,i)
            node_j = V[j]
            com_j = y_pred[int(node_j) ]
            if com_i != com_j:
                continue
            degree_j = G.degree(node_j)
            Q += A[i][j] - degree_i * degree_j/(2 * m)  
    return Q / m

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
    print('number of edges:{}, node community distribute:{}'.format(m,dist))
    for i in range(n):
        print('\rmodularity running node:{},now 2*m*modularity:{:8.8f}'.format(i,Q),end='')
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
    return Q/(2 * m)
