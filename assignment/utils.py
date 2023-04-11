'''
Tailored for DSAA6000B assignment
'''

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, n_feature=10, n_class=2, n_layer=2):
        super().__init__()
        torch.manual_seed(123)
        self.n_layer = n_layer
        self.conv0 = GCNConv(n_feature, n_class)
        self.conv1 = GCNConv(n_feature, hidden_channels)
        self.conv_mid = GCNMidLayer(GCNlayer(hidden_channels), n_layer-2)
        self.convN = GCNConv(hidden_channels, n_class)
    
    def forward(self, x, edge_index, dropout=0.1):
        if self.n_layer==1:
            x = self.conv0(x, edge_index)
        elif self.n_layer==2:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=dropout)
            x = self.convN(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=dropout)
            x = self.conv_mid(x, edge_index)
            x = self.convN(x, edge_index)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GCNlayer(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, dropout=0.1):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=dropout)
        return x
    
class GCNMidLayer(torch.nn.Module):
    def __init__(self, layer, n_layer):
        super().__init__()
        self.layers = _get_clones(layer, n_layer)

    def forward(self, x, edge_index, dropout=0.1):
        for layer in self.layers:
            x = layer(x, edge_index, dropout=dropout)
            
        return x

def load_object(path:str):
    from pickle import load
    with open(path, 'rb') as ff:
        s = load(ff)
    return s

def dump_object(obj, path:str, protocol=4):
    from pickle import dump
    with open(path, 'wb') as handle:
        if(protocol==0):
            dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            dump(obj, handle, protocol=protocol)
