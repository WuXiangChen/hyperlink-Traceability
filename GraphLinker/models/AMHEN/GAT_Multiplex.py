
'''导包区'''
from turtle import forward
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv,global_mean_pool
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter
from GraphLinker.models.AMHEN.GAT_ResNet import GAT_Res
from GraphLinker.models.AMHEN.GCN_ResNet import GCN_Res

class MultiGaT(nn.Module):
  def __init__(self, hp_hiddenD = [145,21,1212,5], layer_per_hf=3, combineType = "Attention",  out_dim=128):
    super().__init__()  # 确保调用父类的初始化
    self.hp_hD = hp_hiddenD
    self.hD = 128
    # self.hp_GC = nn.ModuleList(
    #     [GAT_Res(num_layers=layer_per_hf, input_dim=hfD, hidden_dim=self.hD, step_layer=2) for hfD in hp_hiddenD]
    # )
    self.hp_GC = nn.ModuleList(
        [gnn.ChebConv(in_channels=hfD, out_channels=self.hD, K=layer_per_hf) for hfD in hp_hiddenD]
    )
    
    self.hp_GN = nn.ModuleList(
        [gnn.GraphNorm(self.hD) for hfD in hp_hiddenD]
    )

    if combineType=="Attention":
      self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hD*1, num_heads=1)  # 假设输入维度和头数
    self.max_pool = gnn.global_max_pool
    self.mean_pool = gnn.global_mean_pool
    self.dropout = nn.AlphaDropout(p=0.3)
    self.tanh = nn.Hardtanh()
    self.outLayer = nn.Linear(self.hD*1*1, out_dim)

  @staticmethod
  def norm_pool(x, batch):
        size = int(batch.max().item() + 1)
        counts = torch.unique(batch, sorted=True, return_counts=True)[1]
        return torch.sqrt(scatter(x ** 2, batch, dim=0, dim_size=size, reduce='sum') / counts.view(-1, 1))
  
  @staticmethod
  def min_pool(x, batch):
      size = int(batch.max().item() + 1)
      return scatter(x, batch, dim=0, dim_size=size, reduce='min')
  
  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    splitI1, splitI2, splitI3 = self.hp_hD[0], self.hp_hD[0]+self.hp_hD[1], self.hp_hD[0]+self.hp_hD[1]+self.hp_hD[2]
    cur_device = self.outLayer.weight.device
    edge_index = edge_index.to(device=cur_device)
    batch = x[:, -1].to(device=cur_device)
    # 转换特征为 torch.float 并移动到相同的设备
    ab_features = x[:, :splitI1].to(dtype=torch.float, device=cur_device)
    x1 = self.tanh(self.hp_GC[0](ab_features, edge_index, batch=batch))
    x1_N = self.hp_GN[0](x1)
    x1 = self.dropout(x1_N)
    
    # lb_features = x[:, splitI1:splitI2].to(dtype=torch.float, device=cur_device)
    # x2 = self.tanh(self.hp_GC[1](lb_features, edge_index, batch=batch))
    # x2_N = self.hp_GN[1](x2)
    # x2 = self.dropout(x2_N)

    # fb_features = x[:, splitI2:splitI3].to(dtype=torch.float, device=cur_device)
    # x3 = self.tanh(self.hp_GC[2](fb_features, edge_index, batch=batch))
    # x3_N = self.hp_GN[2](x3)
    # x3 = self.dropout(x3_N)

    # atb_features = x[:, splitI3:-1].to(dtype=torch.float, device=cur_device)
    # x4 = self.hp_GC[3](atb_features, edge_index, batch=batch)
    # AllFea = [x1, x2, x3]
    
    AllFea = [x1]
    # 这里做selfAttention是否会更加合理一些？
    x = torch.cat(AllFea, dim=1)
    x = self.max_pool(x, batch)
    output = self.outLayer(x)
    return output