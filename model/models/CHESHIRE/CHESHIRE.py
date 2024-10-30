import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter

from model.models.resNet import ResidualBlock
from utils import Utils

class CHESHIRE(nn.Module):
    def __init__(self, pos_set, emb_dim, artifacts,node_semantic_feature, conv_dim, k, p, running_type):
        super(CHESHIRE, self).__init__()
        self.node_semantic_feature = node_semantic_feature
        self.artifacts = artifacts
        # 加一层卷积结构，使得最后的输出维度为1
        # 定义卷积层序列
        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=1024, out_channels=512),
            ResidualBlock(in_channels=512, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=1)
        )
        self.linear_encoder = nn.Linear(pos_set.shape[1], emb_dim)
        self.tanh = nn.Hardtanh()
        self.norm_emb = gnn.GraphNorm(emb_dim)
        self.dropout = nn.AlphaDropout(p)
        self.graph_conv = gnn.ChebConv(emb_dim, conv_dim, k)
        self.max_pool = gnn.global_max_pool
        self.linear = nn.Linear(2 * conv_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.feature = torch.Tensor(pos_set)
        self.running_type = running_type
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, incidence_matrix, labels=None):
        torch.autograd.set_detect_anomaly(True)
        incidence_matrix = incidence_matrix.T
        feature = self.feature.to(self.linear_encoder.weight.device)
        if self.running_type == "structure":
            x = self.tanh(self.linear_encoder(feature))
        elif self.running_type == "semantic":
            x_ = self.node_semantic_feature.to(self.linear_encoder.weight.device)
            x = self.conv_layers(x_).squeeze()
            # del x_
        else:
            x = self.tanh(self.linear_encoder(feature))
            x_ = self.node_semantic_feature.to(self.linear_encoder.weight.device)
            x_ = self.conv_layers(x_).squeeze()
            x = x + x_
            del x_
        x, hyperedge_index = self.partition(x, incidence_matrix)
        edge_index, batch = self.expansion(hyperedge_index)
        x = self.dropout(self.norm_emb(x, batch))
        x = self.tanh(self.graph_conv(x, edge_index))
        y_maxmin = self.max_pool(x, batch) - self.min_pool(x, batch)
        y_norm = self.norm_pool(x, batch)
        y = torch.cat((y_maxmin, y_norm), dim=1)
        y_hat = self.linear(y)
        output = self.sigmoid(y_hat).squeeze()
        # Loss calculation
        if labels is not None:
            # Assuming binary classification, you can use BCE loss
            if len(labels) == 0 or len(output) == 0:
                print("labels:", labels)
                print("outputs:", output)
            loss = self.loss_fn(output, labels.float())  # Ensure labels are float for BCE
            return [loss, output]
        else:
            raise ValueError("Labels are required for training")

    @staticmethod
    def norm_pool(x, batch):
        size = int(batch.max().item() + 1)
        counts = torch.unique(batch, sorted=True, return_counts=True)[1]
        return torch.sqrt(scatter(x ** 2, batch, dim=0, dim_size=size, reduce='sum') / counts.view(-1, 1))

    @staticmethod
    def min_pool(x, batch):
        size = int(batch.max().item() + 1)
        return scatter(x, batch, dim=0, dim_size=size, reduce='min')

    def expansion(self, hyperedge_index):
        node_set = hyperedge_index[0]
        index = hyperedge_index[1].int()
        edge_index = torch.empty((2, 0), dtype=torch.int64).to(self.linear_encoder.weight.device)
        batch = torch.empty(len(node_set), dtype=torch.int64).to(self.linear_encoder.weight.device)
        for i in range(index[-1] + 1):
            nodes = node_set[index == i]
            batch[nodes.long()] = i
            num_nodes = len(nodes)
            adj_matrix = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            row, col = torch.where(adj_matrix)
            row, col = nodes[row], nodes[col]
            edge = torch.cat((row.view(1, -1), col.view(1, -1)), dim=0)
            edge_index = torch.cat((edge_index, edge), dim=1)
        return edge_index, batch

    def partition(self, x, incidence_matrix):
        hyperedge_index = Utils.create_hyperedge_index(incidence_matrix)
        node_set, sort_index = torch.sort(hyperedge_index[0])
        hyperedge_index[1] = hyperedge_index[1][sort_index]
        x = x[node_set.long(), :]
        hyperedge_index[0] = torch.arange(0, len(hyperedge_index[0]))
        index_set, sort_index = torch.sort(hyperedge_index[1])
        hyperedge_index[1] = index_set
        hyperedge_index[0] = hyperedge_index[0][sort_index]
        return x, hyperedge_index
