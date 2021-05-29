import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset

import pandas as pd
import csv
import networkx as nx

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def load_graph( name ):
    '''
    path = ''
    if name =='bio72':
        path = './mygraph/bio30_ps.txt'   # my mix graph
    elif name in ['dpwk','line','lle','n2v']:
        path = './mygraph/' + name + '_edgelist.txt'

    '''    
    path = './mygraph/' + name + '_edgelist.txt'
    # print('training graph({}) reading...'.format(path))
    n = 25023           # number of nodes 

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)

    adj_dense = adj.todense()
    adj_csr = adj.tocsr()
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj,adj_csr, adj_dense

# watch and check adj graph
# adj = load_graph(' ',0)
# print(adj[17909])

class load_data(Dataset):
    def __init__(self, name):
        # self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        # self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)
        
        print('reading {}.txt, please wait for about a thousand years....'.format(name))
        if name == 'bio72':
            in_fname = './mydata/bio72.csv'                      # bio72
            df = pd.read_csv(in_fname, header= 0, index_col= 0)
            self.x = np.array(df).tolist()
        elif name == 'biomat':
            in_fname = './mydata/biomat.txt'
            self.x = np.loadtxt( in_fname, dtype=int)
        elif name in ['dpwk','line','lle','n2v']:
            in_fname = './mydata/ordered_'+name+'.txt'
            self.x = np.loadtxt( in_fname, dtype=float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(),\
            torch.from_numpy(np.array(idx))
        #       torch.from_numpy(np.array(self.y[idx])),\
               


def mytest():
    
    fname = './mygraph/bio30_ps.adjlist'
    G = nx.read_adjlist(fname)
    l = list(G.nodes())
    print( l[ :10] )
    print( 'node type:{}'.format(type(l[0]))  )

if __name__ == '__main__':
    mytest()
