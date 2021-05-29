import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from otucd.utils import to_sparse_tensor

__all__ = [
    'GCN',
    'GraphConvolution',
]


def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.cuda.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)


class GraphConvolution(nn.Module):
    """Graph convolution layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        return adj @ (x @ self.weight) + self.bias


class GCN(nn.Module):
    """Graph convolution network.

    References:
        "Semi-superivsed learning with graph convolutional networks",
        Kipf and Welling, ICLR 2017
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)  # 128-> | 512->17 | 
        self.layers = nn.ModuleList([GraphConvolution(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(GraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None
            
        # features auto_encoder decoder part
        x_decoder_hiddens = np.flip(hidden_dims.copy())
        x_decoder_dims = np.concatenate([x_decoder_hiddens, [input_dim]]).astype(np.int32) #  | 512,128 |
        self.dec_layers = nn.ModuleList([GraphConvolution(output_dim, x_decoder_dims[0])]) # 17->512
        for idx in range(len(x_decoder_dims) - 1):                                         # 17->512->128
            self.dec_layers.append(GraphConvolution(x_decoder_dims[idx], x_decoder_dims[idx + 1])) 
        if batch_norm:
            self.dec_batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in x_decoder_hiddens
            ]
        else:
            self.dec_batch_norm = None
        
        
    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1) # 0.7274
            #     adj.setdiag(10) 
            
            # normalized diag( sum^2 )
            #     s0, s1 = adj.sum(0), adj.sum(1)
            #     print( 's0 shape:{}, s1 shape:{}'.format(s0.shape, s1.shape) )
            #     squa_sum =  np.array( s0.dot( s1 ) ) [0][0]   # sum of diag square
            #     squa_list = s0. dot( sp.diags(  np.array(s0) , [0]  ).toarray()  )  # square list of diag
            #     adj.setdiag( 25023 * np.array(squa_list) / squa_sum ) # low loss 0.216 but low modul 0.717,0.726  ``` 0.7343 ``` 
            
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)

    def forward(self, x, adj):
        for idx, gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = gcn(x, adj)
            if idx != len(self.layers) - 1:
                x = F.relu(x)  # leaky_relu:nan celu:nan bad rrelu:nan prelu:miss w selu:nan
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        
        # feature decoder part
        x_rec = F.relu(x)
        for idx, gae in enumerate(self.dec_layers):
            if self.dropout != 0:
                x_rec = sparse_or_dense_dropout(x_rec, p=self.dropout, training=self.training)
            x_rec = gae(x_rec, adj)
            if idx != len(self.dec_layers) - 1:
                x_rec = F.relu(x_rec)  # leaky_relu:nan celu:nan bad rrelu:nan prelu:miss w selu:nan
                if self.dec_batch_norm is not None:
                    x_rec = self.dec_batch_norm[idx](x_rec)
        
        return x, x_rec

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
