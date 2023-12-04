# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:42:38 2023

@author: ha242089
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io

import mctorch.nn as mnn


class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)

        return out


class Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def Laplacian_Connectivity(A: torch.Tensor, lmax = 1, C_type = 'E') -> torch.Tensor:
    
    A = torch.sigmoid(A)
    N = len(A)
    
    if C_type == 'E':
        A = A + A.T # Make it to be Connectivity matrix, symmetry property
        
    if C_type == 'R':
        A = A + A.T# It's already postive definite, no further action is needed    
    
    #T_Diag2zero = (torch.ones(N,N)-torch.eye(N,N)).to('cuda')
    T_Diag2zero = (torch.ones(N,N)-torch.eye(N,N))     
    A = A*T_Diag2zero # Make the diag to be 0   
    
    
    d = torch.sum(A, 1)
    d_ihalf = 1 / torch.sqrt((d + 1e-10))
    D_ihalf = torch.diag_embed(d_ihalf)
    L = torch.eye(N,N) - torch.matmul(torch.matmul(D_ihalf, A), D_ihalf)
    
    vals = torch.linalg.eigvals(L)
    vals = torch.view_as_real(vals)[:,0]
    lmax = torch.max(vals)
    
    Lnorm=(2*L/lmax) - torch.eye(N,N)
    return Lnorm


def generate_cheby_adj(L: torch.Tensor, num_chebys: int) -> torch.Tensor:
    support = []
    for i in range(num_chebys):
        if i == 0:
            support.append(torch.eye(L.shape[-1]))
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1]) - support[-2]
            support.append(temp)
    return support


class Chebynet(nn.Module):
    def __init__(self, in_channels: int, num_chebys: int, out_channels: int):
        super(Chebynet, self).__init__()
        self.num_chebys = num_chebys
        self.gc1 = nn.ModuleList()
        
        for i in range(num_chebys):
            self.gc1.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x: torch.Tensor, Lnorm: torch.Tensor) -> torch.Tensor:
        adj = generate_cheby_adj(Lnorm, self.num_chebys)
        
        for i in range(self.num_chebys):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = result.relu()
        return result


class DGSST(nn.Module):
    def __init__(self, in_channels: int = 41, num_electrodes: int = 32, num_chebys: int = 2, hid_channels: int = 16, num_classes: int = 1, net_type = 'C'):
        super(DGSST, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_chebys = num_chebys
        self.num_classes = num_classes
        self.net_type = net_type

        self.layer1 = Chebynet(in_channels, num_chebys, hid_channels)
        self.BN1 = nn.BatchNorm1d(hid_channels)
        self.fc1 = Linear(num_electrodes * hid_channels, 64)
        self.fc2 = Linear(64, num_classes)
        
        # B_1. Euclidean  EEG distance connectivity
        # filename_1 = 'C:/Users/ha242089/Dropbox/A. Journal/2024 GNN/Data/DEAP/EEG/Connectivity_distance_normalized.mat'
        # mat_content = scipy.io.loadmat(filename_1)
        # orig_adjacency = mat_content['EEG_Connectivity_Matrix']   
        # self.Connectivity = nn.Parameter(torch.FloatTensor(orig_adjacency))
        
        # B_2. Euclidean  EEG random connectivity
        # self.Connectivity = nn.Parameter(torch.FloatTensor(self.num_electrodes, self.num_electrodes))
        # nn.init.xavier_normal_(self.Connectivity)
        
        # B_3. Riemannian EEG Coherence connectivity, we will implement it later
        # filename_2 = 'C:/Users/ha242089/Dropbox/A. Journal/2024 GNN/Data/DEAP/EEG_Coherence/Connectivity_coherence_S_' + str(Index_subject+1) + '.mat'
        # mat_content = scipy.io.loadmat(filename_2)
        # orig_adjacency = mat_content['BNA_matrix_binary_234']   
        # orig_adjacency = orig_adjacency - np.eye(self.num_nodes)
        # self.Connectivity = mnn.Parameter(data = orig_adjacency, manifold = mnn.PositiveDefinite(self.num_electrodes) )
        
        # B_4. Riemannian EEG random connectivity
        self.Connectivity = mnn.Parameter(data = None, manifold = mnn.PositiveDefinite(self.num_electrodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
               
        #print(x.shape)
        #print(x.shape)
        L = Laplacian_Connectivity(self.Connectivity, C_type= self.net_type)
        x = self.layer1(x, L)
        
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2) # keep nodes as the same, only do batchnorm for batch*features
        x = x.reshape(x.shape[0], -1)  # flatten here batch*[nodes*features]
        x = self.fc1(x).relu()
        nn.Dropout(0.5)
        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        
        return x      
