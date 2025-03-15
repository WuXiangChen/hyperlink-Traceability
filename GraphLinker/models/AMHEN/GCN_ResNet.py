# 在本节中设计GCN-Res的基本结构
import torch.nn as nn
from torch_geometric.nn import GCNConv,global_mean_pool
import torch.nn.functional as F

class GCN_Res(nn.Module):
    def __init__(self, num_layers, hidden_dim, step_layer=3):
        super(GCN_Res, self).__init__()
        self.step_layer = step_layer
        self.num_layers = num_layers
        self.gconv_layers = nn.ModuleList([
            GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
            for _ in range(num_layers)
        ])
        self.batchnorm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index):
        for i, gconv in enumerate(self.gconv_layers):
            if i % self.step_layer == 0:
                x_0 = x
            x_in = x
            x = gconv(x, edge_index)
            x = F.relu(x)
            if i != 0 and i != len(self.gconv_layers) - 1 and (i + 1) % self.step_layer != 0:
                x = x + x_in
                x = F.relu(x)
            elif (i!=0 and (i+1)%self.step_layer==0) or i==len(self.gconv_layers)-1:
                x = x + x_0
                x = F.relu(x)
            x = self.batchnorm_layers[i](x)
        return x