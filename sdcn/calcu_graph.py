import pandas as pd
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize

import csv 
from math import sqrt
import networkx as nx


topk = 30

# def construct_graph(features, label, method='heat'):
def construct_graph(features, method='heat'):
    fname = 'bio30_graph.txt'
    # num = len(label)  
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        # features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)
    
    # bio graph
    adj_name = 'bio30.adjlist'
    G = nx.Graph() 
    G.add_nodes_from(np.arange(25023))
    
    f = open(fname, 'w')
    # counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                # if label[vv] != label[i]:
                #     counter += 1
                f.write('{} {}\n'.format(i, vv))
                
                G.add_edge(i, vv)
                
    f.close()
    nx.write_adjlist(G, adj_name)
    
    # print('error rate: {}'.format(counter / (num * topk)))

'''
f = h5py.File('data/usps.h5', 'r')
train = f.get('train')
test = f.get('test')
X_tr = train.get('data')[:]
y_tr = train.get('target')[:]
X_te = test.get('data')[:]
y_te = test.get('target')[:]
f.close()
usps = np.concatenate((X_tr, X_te)).astype(np.float32)
label = np.concatenate((y_tr, y_te)).astype(np.int32)
'''

'''
hhar = np.loadtxt('data/hhar.txt', dtype=float)
label = np.loadtxt('data/hhar_label.txt', dtype=int)
'''

# otu_table_25023.csv

in_file_name = 'otu_table_25023.csv'
df = pd.read_csv(in_file_name, header= 0, index_col= 0)
in_table = np.array(df).tolist()

# reut = np.loadtxt('data\reut.txt', dtype=float)
# label = np.loadtxt('data\reut_label.txt', dtype=int)

construct_graph(in_table, 'ncos')
