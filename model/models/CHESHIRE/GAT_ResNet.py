
# 在本节中设计GAT-Res的基本结构
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv,global_mean_pool
import torch.nn.functional as F

class GAT_Res(nn.Module):
  def __init__(self, num_layers, hidden_dim, step_layer=3):
    super(GAT_Res, self).__init__()
    # 使用 ModuleList 创建多个 GATConv 层
    self.step_layer = step_layer
    self.num_layers = num_layers
    self.gconv_layers = nn.ModuleList([
        GATConv(in_channels=hidden_dim * 8, out_channels=hidden_dim, heads=8)
        for _ in range(num_layers)  # num_layers 是你想要的层数
    ])

    self.batchnorm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim*8)
            for _ in range(num_layers)  # num_layers 是你想要的层数
        ])

  def forward(self, x, edge_index):
    for i, gconv in enumerate(self.gconv_layers):
            if i % self.step_layer ==0:
                x_0 = x
            x_in = x
            x = gconv(x, edge_index)
            x = F.relu(x)
            if i!=0 and i!=len(self.gconv_layers)-1 and (i+1)%self.step_layer!=0:
                x = x + x_in
                x = F.relu(x)
            elif (i!=0 and (i+1)%self.step_layer==0) or i==len(self.gconv_layers)-1:
                x = x + x_0
                x = F.relu(x)
            x = self.batchnorm_layers[i](x)
    return x