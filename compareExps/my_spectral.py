import numpy as np
from  sklearn import datasets
from sklearn.cluster import SpectralClustering
import pandas as pd
import csv
import networkx as nx
# from networkx.algorithms import community
import time
import argparse

#   from utils import load_graph
from myutils import load_ordered_graph              # nodes ordered

# added in 20210314
def modularity(G, community):
    """
    Compute modularity of communities of network

    Parameters
    --------
    G : networkx.Graph
        an undirected graph
    community : dict
        the communities result of community detection algorithms
    """
    V = [node for node in G.nodes()]
    m = G.size(weight='weight')  # if weighted
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    Q = 0
    count = 0
    com_cnt = {}
    for i in range(8):
        com_cnt[i] = 0
    for i in range(n):
        print('\r{} node modularity running.......'.format(i),end='')
        node_i = V[i]               # i : index
        com_i = community[i]
        degree_i = G.degree(node_i)
        com_cnt[com_i] += 1
        for j in range(n):
            node_j = V[j]
            com_j = community[j]
            if com_i != com_j:
                continue
            degree_j = G.degree(node_j)
            Q += A[i][j] - degree_i * degree_j/(2 * m)
            count += 1
    print('\ncommunity count:{}'.format(com_cnt))
    print('com_i==com_j count:{}'.format(count))
    print('number of edges:{}'.format(G.number_of_edges()))
    print('now 2mQ:{}'.format(Q))
    return Q/(2 * m)
    

def my_GN( G ):
     # nx GN community detection algorithm    fucking slow!!!
    com_gen = community.girvan_newman(G)
    for i in range(100):
        print('{} epoch gn community detection running...'.format(i))
        com_ret_i = next(com_gen)
        if i == 99:
            print(com_ret_i)



'''
my_spectral usage:  $ python my_spectral.py --objtype features --features_path ../mydata/embd_dpwk.txt 
                    input:  features and features;
                    output: spectral clustering community detection modularity metric and community distribution shown on screen;
'''
if __name__ == '__main__':
    
    # args parse
    parser = argparse.ArgumentParser( description='spectral cluster',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--objtype',type=str,default='matrix')
    parser.add_argument('--graph_path',type=str,default='../mygraph/bio30_ps.adjlist')   # graph path
    parser.add_argument('--features_path',type=str,default='None')                    # when objtype features needed
    parser.add_argument('--n_clusters',type=int,default=8)                          # n_cluster
    args = parser.parse_args()

    print('begin time:{}'.format(time.asctime(time.localtime(time.time()))))
    print('method:spectral cluster')
    print( 'args:{}'.format(args) )
    
    
 
    # predict                               # cluster(data.x)
    #  y_pred = SpectralClustering( n_clusters = 8).fit_predict(x)  

    # graph_fname = './mygraph/bio30_ps.adjlist'     # graph adjlist format
    G = nx.read_adjlist( args.graph_path )  
    print('graph:{} read over'.format( args.graph_path ) )
       
    # adjacent matrix   # pred matrix
    print('predict running...')
    y_pred = []
    A = nx.to_numpy_array(G)
    if args.objtype == 'matrix': # matrix precomputed already, only arg:matrix, matrix got by graph, idx very ok
        y_pred = SpectralClustering( args.n_clusters , affinity='precomputed').fit_predict( A )  
    else:                       # data.x matrix computed by spectral cluster
        if args.features_path == r'../mydata/bio72.csv':
            df = pd.read_csv( args.features_path , header=0, index_col=0)
            x = np.array(df).tolist() 
            # print( x[:3] )
        else:
            total = np.loadtxt( args.features_path, skiprows=1  )   # embds  first line skip
            print('features:{} read over'.format( args.features_path ) )  # features read over
            total_sorted = sorted(total, key = lambda line:line[0])     # sorted by node
            # x, ids  = total_sorted[ : , 1:], total_sorted[ : ,0 ]                                  # embd value 128 cols, embd id
            x, ids = [] , []            # front and back cols
            for line in total_sorted:
                x.append( list(line[1:] ) )
                ids.append( line[0] )
            # print( ids[:100] )  # watch
            # print( x[:3] )
        y_pred = SpectralClustering( args.n_clusters ).fit_predict( x )             # id->y_pred not ok
        G = load_ordered_graph( args.graph_path  )          # special process
    # print(y_pred)
     
    # test direct/undirect  
    # for i in range(len(A)):
    #    if A[0][i] != A[i][0]:
    #       print('it\'s not a undirected graph ')
    #       exit()
   
   
    print('modularity evaluating...')
    community = {}                    # idx2com          
    for i,com_i in enumerate(y_pred):
        community[i] = com_i
    modu_val = modularity(G, community)  # eval
    print('modularity value:{}'.format(modu_val))
    print('end time:{}'.format(time.asctime(time.localtime(time.time()))))
