import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
# from evaluation import eva

import time
import pandas as pd
import csv

#torch.cuda.set_device(3)
class AE(nn.Module):

    # initially n_input,500,500,2000,10,2000,500,500,n_input
    def __init__(self, n_input, n_1, n_2, n_3, n_z, n_d3, n_d2, n_d1 ):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_1)
        self.enc_2 = Linear(n_1, n_2)
        self.enc_3 = Linear(n_2, n_3)
        self.z_layer = Linear(n_3, n_z)

        self.dec_3 = Linear(n_z, n_d3)
        self.dec_2 = Linear(n_d3, n_d2)
        self.dec_1 = Linear(n_d2, n_d1)
        self.x_bar_layer = Linear(n_d1, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)
        
        dec_h3 = F.relu(self.dec_3(z))  
        dec_h2 = F.relu(self.dec_2(dec_h3))                                                                  
        dec_h1 = F.relu(self.dec_1(dec_h2))
        x_bar = self.x_bar_layer(dec_h1)
        
        return x_bar, enc_h1, enc_h2, enc_h3, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def pretrain_ae(model, dataset, y):
def pretrain_ae(epochs, model, dataset, out_fname, _lr ):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=_lr )   # lr
    for epoch in range( epochs ):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _h1, _h2, _h3, _hz = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar,_h1, _h2, _h3, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))           
    
        if epoch == epochs-1:
            print('writing...')
            torch.save(model.state_dict(), out_fname)


def dopretrain( name, epochs, _n_1, _n_2, _n_3, _n_z, _n_d3, _n_d2, _n_d1, _lr = 1e-3):
    print( 'file reading...please wait and wait, without complaint, because it may not change anyway.so sad.')
    # x = np.loadtxt('dblp.txt', dtype=float)                     # dblp
    # y = np.loadtxt('dblp_label.txt', dtype=int)
    if name == 'bio72':
        in_fname = './mydata/bio72.csv'
        out_fname = './pretrain/bio72.pkl'
        df = pd.read_csv(in_fname, header= 0, index_col= 0)         # bio72
        x = np.array(df).tolist()
    elif name == 'biomat':
        in_fname = './mydata/biomat.txt'
        out_fname = './pretrain/biomat.pkl'
        x = np.loadtxt( in_fname, dtype=float)                        
    
    _n_input = len(x[0])
    dataset = LoadDataset(x)
    model = AE( 
        n_input = _n_input, n_1=_n_1,  n_2=_n_2,  n_3=_n_3,
        n_z = _n_z,  n_d3=_n_d3,  n_d2=_n_d2, n_d1=_n_d1 ).cuda()
                                                    
    print( 'pretraining...' )
    pretrain_ae(epochs, model, dataset, out_fname, _lr)
     


if __name__ == '__main__':
    print('begin time:{}'.format(time.asctime(time.localtime(time.time()))))    # time
    # epochs  n_1 n_2 n_3 n_z n_d3, n_d2, n_d1, learing rate
    dopretrain('bio72',50, 500,500,2000,10,2000,500,500,1e-3)
    print('end time:{}'.format(time.asctime(time.localtime(time.time()))))    # time
