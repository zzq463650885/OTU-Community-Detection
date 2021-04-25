# -*- coding:gb2312-*-

import numpy as np
import pandas as pd
from math import sqrt
import networkx as nx
import csv    # to write one line per time


def get_features():                                 

    otu_table = []                              
    otu_ids = []
    
    num_seqs_per_run = [56011,118070,87493,106001,92217,104742,94302,81851,
    73527,74166,92701,94428,110959,67172,88245,58277,84825,127247,121077,70241,153597,142899,111533,75197,
    60127,97292,93319,108536,104641,76003,53763,100829,68460,66566,100400,67121,141585,75578,134383,88533,
    66673,72298,115752,89371,108419,128528,87819,89488,114741,191312,131809,84484,126081,131136,165071,100248,
    68751,78840,107126,67463,68701,103327,155631,56433,94337,80266,75026,67105,81258,79195,109180,58489]
    
    run2col = {};                                   # 字典：ERR2031892  ->  列下标
    for i in range(72):
        pre = 'ERR2031'
        run_num = 892+i
        run_str = pre+str(run_num)
        run2col[run_str] = i
    
    file_name_to_read = 'final_otu_map_mc2.txt'     # 读、写文件名
    file_name_to_write = 'otu_table_25023.csv'
    
    
    with open(file_name_to_read, 'r') as f:
        
        while True:
            line = f.readline()
            if not line:
                break
            curr_strs = line.split()
            
            otu_id = curr_strs[0]                   # id
            curr_str_list = curr_strs[1:]           # otu的序列列表
            
            
            seqs_threshhold = 5           # 筛选地点
            if len( curr_str_list ) < seqs_threshhold :       
                continue
            curr_otu_line = np.zeros(72)            # one line in otu_table
            
            for seq in curr_str_list:           
                run_str = seq.split('.')[0] 
                curr_otu_line[ run2col[run_str] ] += 1
            otu_table.append( curr_otu_line )   
            otu_ids.append(otu_id)
    
    # get percentage: count->percentage
    line_num = len(otu_table)
    print(line_num)
    for line_index in range(line_num): 
        print(str(line_index) + ':percentage')      # line
        for col in range(72):
            otu_table[line_index][col] /= num_seqs_per_run[col]
        
    # save otu_table
    df = pd.DataFrame(otu_table,index=otu_ids, columns=np.arange(72))  
    df.to_csv(file_name_to_write)   
    print('write over')
    
    # save num->id of otu
    # df = pd.DataFrame(otu_ids)  
    # df.to_csv('D:\\ncbi\\my_total\\otuNum2Name.csv')
    

def get_pearson_network( in_table, out_file_name, num_filter, ids=[] ):
   
    m = len(in_table)                               # 行*列 m*n
    n = len(in_table[0])                           
    in_avgs = np.zeros(m)                           # 中间结果：均值的数组
    in_ssfm2 = np.zeros(m)                          # 中间结果：sqrt(sum((x-x_avg)^2))
    in_first_moment = []                            # 中间结果：一阶矩
    
    # x_avg of line
    for i in range(m):
        sum = 0
        for j in range(n):     
            sum += in_table[i][j]
        curr_avg = in_avgs[i] = sum/n 
        
        curr_arr = np.zeros(n)          # x_j - x_avg
        sum_first_moment_2 = 0
        for j in range(n):
            curr_arr[j] = in_table[i][j] - curr_avg
            sum_first_moment_2 += curr_arr[j] * curr_arr[j]
        in_ssfm2[i] = sqrt(sum_first_moment_2)
        in_first_moment.append( curr_arr )
        
    print('first_moment over')
    
    
    G = nx.Graph()                                  # 根据一阶矩计算相关系数,构建网络
    G.add_nodes_from(np.arange(m))                  # 自动避免离群点 
    
    # for i in range(30):                             # 测试
    for i in range(m):
        count_neighbours = 0
        for j in range(i+1,m):                      # 只计算上三角矩阵
            sum = 0
            for k in range(n):
                sum += in_first_moment[i][k] * in_first_moment[j][k]

            sum /= (in_ssfm2[i] * in_ssfm2[j]) 
            neg_filter = -1*num_filter
            if sum > num_filter or sum < neg_filter :
                G.add_edge(i,j)
                count_neighbours += 1
        print('node i:{}, neighbours: {}'.format(i,count_neighbours))                     # i ok
        
    nx.write_adjlist(G, out_file_name)              # 理应 4万多行(不去除离群点45738行) * 若干列
    print('network saved')
    return G
    


def order1graph():
    in_file_name = 'otu_table_25023.csv'
    df = pd.read_csv(in_file_name, header= 0, index_col= 0)
    in_table = np.array(df).tolist()
    print(len(in_table), len(in_table[0]))
    
    get_pearson_network(in_table, 'order1graph.adjlist', 0.5 )
    return 
    
    

def remove_outliers():

    G = nx.read_adjlist('graph.adjlist')
    to_rm = []
    for i in G.nodes:
        if G.degree(i) == 0 :
            to_rm.append(i)
    G.remove_nodes_from(to_rm)
    nx.write_adjlist(G,'order1graph_notls.adjlist')
    
    print(G.number_of_nodes())                      # 一阶网络(节点都有边)：45210结点数    5423476边数
    print(G.number_of_edges())



def order2graph():
    in_prefix = '../features/'
    out_prefix = './graphs/'
    embds = ['dpwk','line','n2v','lle']
    in_file_names = [ in_prefix + i + '.txt' for i in embds  ]
    out_file_names = [ out_prefix + i + '.adjlist' for i in embds  ]
    
    left_idxs = [1,2]
    for i in left_idxs: 
    # for i in range(len(in_file_names)):           # 4个二阶网络,i是文件索引
    # for i in range(1):                              # 测试
        file_name = in_file_names[i]
        print('{} running...'.format(file_name))
        with open(file_name, 'r') as f:             # 读取embeddings文件，放到容器里
            m, n = 25023, 128
            print('file:{} rows:{} cols:{}'.format(file_name,m,n))
            in_table = []                           # 读入向量
            nums = []                               # 节点序号
            
            for j in range(m):                      # j是行数索引
            # for j in range(15):
                line = f.readline()
                curr_str_list = line.split()        # 去除第一个otu名
                numbers = [ float(x) for x in curr_str_list[1:] ]
                in_table.append(numbers)
            # print(in_table)
        
        
        G = get_pearson_network(in_table, out_file_names[i], 0.7 )
            
        print(G.number_of_nodes())          # 
        print(G.number_of_edges())
        
    return
    
    
    
if __name__ == "__main__":
    
    # get_features()
    
    # order1graph()
    
    # remove_outliers()
    
    order2graph()
    
    pass
