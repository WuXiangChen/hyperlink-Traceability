import dgl
import torch
import torch.nn as nn
from model.models.CHESHIRE.GAT_ResNet import GAT_Res
from model.models.CHESHIRE.GCN_ResNet import GCN_Res
from utils import Utils
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv, MaxPooling,GraphConv
import networkx as nx
from torch_geometric.nn import GATConv,global_mean_pool, GCNConv
from torch_geometric.data import Data,Batch
from torch_geometric.utils import from_networkx
import torch
import torch.nn as nn

class BAAI_model(nn.Module):
    def __init__(self, artifacts_dict, artifacts, tokenizer, model, in_dim, freeze, with_knowledge, gat):
        super(BAAI_model, self).__init__()
        self.model = model
        self.freeze = freeze
        self.with_knowledge = with_knowledge

        if freeze:
            for param in self.model.parameters():
               param.requires_grad = False
        
        # 这里待测试
        if not self.with_knowledge:
            for module_ in self.model.named_modules(): 
                if module_[0].startswith("encoder") and hasattr(module_[1], "weight"):
                    module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                elif module_[0].startswith("encoder") and hasattr(module_[1], "weight"):
                    module_[1].bias.data.zero_()
                

        self.artifacts = artifacts
        self.tokenizer = tokenizer
        self.artifacts_dict = {value: key for key, value in artifacts_dict.items()}
        self.in_dim = in_dim
        hidden_dim = 512
        self.linear = nn.Sequential(
            nn.Linear(in_dim*2, hidden_dim),
            nn.ReLU(),
        )
        num_layers = 10 # step_layer的倍数+1

        if gat:
            self.head_gconv_layers = GATConv(in_channels=in_dim, out_channels=hidden_dim, heads=8)
            self.norm1 = nn.BatchNorm1d(hidden_dim*8)
            self.neck_gconv_layers = GAT_Res(num_layers, hidden_dim)
            self.tail_gconv_layers = GATConv(in_channels=hidden_dim*8, out_channels=hidden_dim, heads=1)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
            self.classify1 = nn.Linear(hidden_dim, 1)
            self.loss_fn = torch.nn.BCELoss()
        else:
            self.head_gconv_layers = GCNConv(in_channels=in_dim, out_channels=hidden_dim)
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.neck_gconv_layers = GCN_Res(num_layers, hidden_dim)
            self.tail_gconv_layers = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
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
            for NL_PL in NL_PLs:
                encoded = self.tokenizer.encode_plus(
                    NL_PL,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=128,
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
        torch_geometric_graphs = []
        for c_graph in all_connected_graph:
            for node in list(c_graph.nodes()):
                index = nodes_list.index(node)
                ft = x[index,:].detach()
                c_graph.nodes[node]["ft"] = ft
            torch_geometric_graph = from_networkx(c_graph, group_node_attrs=["ft"])
            torch_geometric_graphs.append(torch_geometric_graph)

        data = Batch.from_data_list(torch_geometric_graphs).to(self.linear[0].weight.device)
        x, edge_index = data.x, data.edge_index
        x = self.head_gconv_layers(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.neck_gconv_layers(x, edge_index)
        x = self.tail_gconv_layers(x, edge_index)
        x = self.norm2(x)
        
        # # Global pooling
        outputs = global_mean_pool(x, data.batch)  # Use global mean pooling
        outputs = self.classify1(outputs)
        outputs = nn.Sigmoid()(outputs).squeeze(1)
        # 计算loss
        if labels is not None:
            loss = self.loss_fn(outputs, labels)
            print("training:", loss)
            return [loss, outputs]