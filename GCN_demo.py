import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, MaxPooling

class GCN_dgl(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout=0.5, training=True):
        super(GCN_dgl, self).__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.pool = MaxPooling()
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout
        self.training = training
    def forward(self, g, x):
        x = x.to(self.gc1.weight.device)
        g = g.to(self.gc1.weight.device)
        x = F.relu(self.gc1(g, x))  # g 是图，x 是节点特征
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)
        x = F.relu(x)
        x = self.pool(g, x)
        x = self.linear(x)
        return x