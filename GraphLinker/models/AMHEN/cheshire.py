import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter

class CHESHIRE(nn.Module):
    def __init__(self, input_dim, emb_dim, conv_dim, k=3, p=0.1):
        super(CHESHIRE, self).__init__()
        self.linear_encoder = nn.Linear(input_dim, emb_dim)
        self.tanh = nn.Hardtanh()
        self.norm_emb = gnn.GraphNorm(emb_dim)
        self.dropout = nn.AlphaDropout(p)
        self.graph_conv = gnn.ChebConv(emb_dim, conv_dim, k)
        self.max_pool = gnn.global_max_pool
        self.linear = nn.Linear(2 * conv_dim, 2 * conv_dim)

    def forward(self, feature, edge_index, batch):
        feature = feature.float()
        x = self.tanh(self.linear_encoder(feature))
        x = self.dropout(self.norm_emb(x, batch))
        x = self.tanh(self.graph_conv(x, edge_index))
        y_maxmin = self.max_pool(x, batch) - self.min_pool(x, batch)
        y_norm = self.norm_pool(x, batch)
        y = torch.cat((y_maxmin, y_norm), dim=1)
        return self.linear(y)
    
    def min_pool(self, x, batch):
        size = int(batch.max().item() + 1)
        return scatter(x, batch, dim=0, dim_size=size, reduce='min')
    
    def norm_pool(self, x, batch):
        size = int(batch.max().item() + 1)
        counts = torch.unique(batch, sorted=True, return_counts=True)[1]
        return torch.sqrt(scatter(x ** 2, batch, dim=0, dim_size=size, reduce='sum') / counts.view(-1, 1))