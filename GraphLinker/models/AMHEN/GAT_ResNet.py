
# 在本节中设计GAT-Res的基本结构
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv,global_mean_pool
import torch.nn.functional as F

class GAT_Res(nn.Module):
  def __init__(self, num_layers, input_dim, hidden_dim=128, step_layer=3, out_dim=128, heads=4):
    super(GAT_Res, self).__init__()
    # 使用 ModuleList 创建多个 GATConv 层
    self.step_layer = step_layer
    self.num_layers = num_layers
    self.heads = heads
    self.head_gconv = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=heads)
    self.gconv_layers = nn.ModuleList([
        GATConv(in_channels=hidden_dim * heads, out_channels=hidden_dim, heads=heads)
        for _ in range(num_layers)  # num_layers 是你想要的层数
    ])

    self.batchnorm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim*heads)
            for _ in range(num_layers)  # num_layers 是你想要的层数
        ])
    self.outlayer = nn.Linear(hidden_dim * heads, out_dim)

  def forward(self, x, edge_index, batch):
    x = self.head_gconv(x, edge_index)
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
    x = global_mean_pool(x, batch)  # 这里的 batch 表示节点所属图的标识
    out = self.outlayer(x)
    return out