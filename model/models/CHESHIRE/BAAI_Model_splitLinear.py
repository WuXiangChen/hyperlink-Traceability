import dgl
import torch
import torch.nn as nn
from utils import Utils
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv, MaxPooling,GraphConv
import networkx as nx
from torch_geometric.nn import GATConv,global_mean_pool
from torch_geometric.data import Data,Batch
from torch_geometric.utils import from_networkx
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=3):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert (
            input_dim % num_heads == 0
        ), "input_dim must be divisible by num_heads"

        # 定义线性层用于生成查询、键和值
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        
        # 输出线性层
        self.out_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x 的形状是 (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # 生成查询、键和值
        queries = self.query_linear(x)  # (batch_size, seq_len, input_dim)
        keys = self.key_linear(x)        # (batch_size, seq_len, input_dim)
        values = self.value_linear(x)    # (batch_size, seq_len, input_dim)

        # 将它们重塑为多头形式
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)          # (batch_size, num_heads, seq_len, head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)      # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # 在 seq_len 上做 softmax

        # 计算上下文向量
        context = torch.bmm(attention_weights.reshape(batch_size * self.num_heads, seq_len, seq_len), values.reshape(batch_size * self.num_heads, seq_len, self.head_dim))  # (batch_size * num_heads, seq_len, head_dim)

        # 重塑回原来的形状
        context = context.view(batch_size, self.num_heads, seq_len, self.head_dim).transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        context = context.contiguous().view(batch_size, seq_len, -1)  # (batch_size, seq_len, input_dim)

        # 通过输出线性层
        output = self.out_linear(context)  # (batch_size, seq_len, input_dim)
        return output

class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(input_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x):
        # x 的形状是 (batch_size, seq_len, input_dim)
        scores = torch.matmul(x, self.attention_weights)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(scores, dim=1)  # 在 seq_len 上做 softmax
        context = torch.bmm(attention_weights.transpose(1, 2), x)  # (batch_size, 1, input_dim)
        return context

class BAAI_model(nn.Module):
    def __init__(self, artifacts_dict, artifacts, tokenizer, model):
        super(BAAI_model, self).__init__()
        self.model = model
        # for param in self.model.parameters():
        #    param.requires_grad = False
        self.artifacts = artifacts
        self.tokenizer = tokenizer
        self.artifacts_dict = {value: key for key, value in artifacts_dict.items()}
        in_dim = 384
        hidden_dim =256
        self.linear = nn.Sequential(
            nn.Linear(in_dim*2, hidden_dim),
            nn.ReLU(),
            #  nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU()
        )
        self.gconv_layers = nn.ModuleList([
            GATConv(in_channels=in_dim, out_channels=hidden_dim, heads=2),
            # GATConv(in_channels=hidden_dim*2, out_channels=hidden_dim, heads=3),
            GATConv(in_channels=hidden_dim*2, out_channels=hidden_dim // 2, heads=1)
        ])
        self.batchnorm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * 2),
            # nn.BatchNorm1d(hidden_dim * 3),
            nn.BatchNorm1d(hidden_dim // 2)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classify1 = nn.Linear(hidden_dim, 1)
        self.loss_fn = torch.nn.BCELoss()

    def graph_to_torch_geometric_data(self, nx_graph):
        # Get the edge list from the NetworkX graph
        edge_list = list(nx_graph.edges())
        # Create the edge_index tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def forward(self, edges, labels=None):
        # 输出每一列中的非0元素的索引
        node_sentence_list, nodes_list = Utils.process_edges_data(self.artifacts, edges)
        inputs = []
        attention_masks = []
        for NL_PLs in node_sentence_list:
            # NL_PL = "".join(NL_PL)
            for NL_PL in NL_PLs:
                encoded = self.tokenizer.encode_plus(
                    NL_PL,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_attention_mask=True
                )
                inputs.append(encoded['input_ids'])
                attention_masks.append(encoded['attention_mask'])

        inputs = torch.cat(inputs, dim=0).to(self.model.device)
        attention_masks = torch.cat(attention_masks, dim=0).to(self.model.device)
        x = self.model(inputs,attention_masks).pooler_output
        #x = self.linear(x)
        # 这里接graph 
        all_connected_graph = Utils.get_fully_network(edges)
        # 接着 为这里每个graph赋值
        outputs = []
        for c_graph in all_connected_graph:
            fea = []
            for node in list(c_graph.nodes())[0:2]:
                index = nodes_list.index(node)
                #ft = x[index,:].detach()
                #ft = self.linear(ft)
                #c_graph.nodes[node]["ft"] = ft
                fea.append(x[index,:].detach().unsqueeze(0))
            fea_ = torch.cat(fea, dim=1).to(self.model.device)
            # fea_ = fea_.permute(1, 0)
            # output = self.pool(fea_)
            # output = output.permute(1, 0)
            outputs.append(fea_)
        # outputs = torch.stack(outputs,dim=1)
        outputs = torch.cat(outputs)
        # attention_layer = SimpleAttention(input_dim=384).to(self.model.device)
        # outputs = attention_layer(outputs).squeeze(1)  # 形状将变为 (25, 1, 384)
        outputs = self.linear(outputs)
        # outputs = self.linear(outputs)
        # outputs = self.classify1(outputs)
        # # 创建一个列表来存储转换后的 DGL 图
        # dgl_graphs = []

        # # 将每个 nx.Graph 转换为 DGL 图
        # for c_graph in all_connected_graph:
        #     dgl_graph = dgl.from_networkx(c_graph, node_attrs=["ft"])
        #     dgl_graphs.append(dgl_graph)
        # # 使用 dgl.batch() 合并所有 DGL 图
        # all_graph = dgl.batch(dgl_graphs)
        # all_graph = all_graph.to(self.linear[0].weight.device)

        # torch_geometric_graphs = []
        # for c_graph in all_connected_graph:
        #     torch_geometric_graph = from_networkx(c_graph, group_node_attrs=["ft"])
        #     torch_geometric_graphs.append(torch_geometric_graph)
        # data = Batch.from_data_list(torch_geometric_graphs).to(self.linear[0].weight.device)
        # # data = Data(x=x, edge_index=edge_index).to(self.linear[0].weight.device)
        # x, edge_index = data.x, data.edge_index
        # for i, gconv in enumerate(self.gconv_layers):
        #     x = gconv(x, edge_index)
        #     x = F.relu(x)
        #     x = self.batchnorm_layers[i](x)
        
        # # Global pooling
        # x = global_mean_pool(x, data.batch)  # Use global mean pooling
        outputs = self.classify1(outputs)
        outputs = nn.Sigmoid()(outputs).squeeze(1)
        # 计算loss
        if labels is not None:
            loss = self.loss_fn(outputs, labels)
            print("training:", loss)
            return [loss, outputs]


