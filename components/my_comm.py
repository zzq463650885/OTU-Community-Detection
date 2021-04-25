# -*- coding:utf-8 -*-

import community
import matplotlib.pyplot as plt
import networkx as nx

path="./mygraph/bio30_ps.adjlist"
Graph=nx.read_adjlist(path)

#输出图信息
print( Graph.graph)
#计算图或网络的传递性
print(nx.transitivity(Graph)) 
#节点邻居的个数
print(Graph.neighbors(1)) 
# 图划分
part = community.best_partition(Graph)
print(part)  
#计算模块度
mod = community.modularity(part,Graph)
print( mod) 

#绘图
values = [part.get(node) for node in Graph.nodes()]
nx.draw_spring(Graph, cmap=plt.get_cmap('jet'), node_color = values, node_size=25023, with_labels=False)
plt.show()

