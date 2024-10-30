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

# 这里加一个LSTM结构用以从word embedding中提取sentence embedding

class SentenceEmbeddingLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=5, bidirectional=False, in_dim=512):
        super(SentenceEmbeddingLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

    def forward(self, x):
        # x: (batch_size, seq_length, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch_size, hidden_dim)
        if self.bidirectional:
            h_n = h_n.view(-1, x.size(0), 2, self.hidden_dim)
            h_n = h_n[:, :, -1, :].mean(dim=2)  # 取双向最后一个隐藏状态的平均
        else:
            h_n = h_n[-1]  # 取最后一层的隐藏状态
        return h_n.unsqueeze(0)  # 返回句子嵌入

class BAAI_model(nn.Module):
    def __init__(self, artifacts_dict, artifacts, tokenizer, model, in_dim, freeze, with_knowledge):
        super(BAAI_model, self).__init__()
        hidden_dim =256
        self.model = model
        self.freeze = freeze
        self.with_knowledge = with_knowledge
        self.senEmbLSTM = SentenceEmbeddingLSTM(embedding_dim=in_dim, hidden_dim=hidden_dim)

        if freeze:
            for param in self.model.parameters():
               param.requires_grad = False
        
        # 这里待测试
        if not self.with_knowledge:
            for layer in self.model.children():
                if hasattr(layer, 'weight'):
                    torch.nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        
        self.artifacts = artifacts
        self.tokenizer = tokenizer
        self.artifacts_dict = {value: key for key, value in artifacts_dict.items()}

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

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
            inputs_ = []
            atts_ = []
            for NL_PL in NL_PLs:
                encoded = self.tokenizer.encode_plus(
                    NL_PL,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_attention_mask=True
                )
                inputs_.append(encoded['input_ids'])
                atts_.append(encoded['attention_mask'])
            inputs.append(inputs_)
            attention_masks.append(atts_)
        # inputs = torch.cat(inputs, dim=0).to(self.model.device)
        # 这里需要根据输入内容进行合并
        sen_emb = []
        for k, input_ in enumerate(inputs):
            att_ = torch.cat(attention_masks[k]).to(self.model.device)
            input_ = torch.cat(input_).to(self.model.device)
            x = self.model(input_ , att_).pooler_output
            sen_emb.append(self.senEmbLSTM(x))
        sen_emb_ = torch.cat(sen_emb, dim=0)
        outputs = self.linear(sen_emb_)
        outputs = self.classify1(outputs)
        outputs = nn.Sigmoid()(outputs).squeeze(1)
        # 计算loss
        if labels is not None:
            loss = self.loss_fn(outputs, labels)
            print("training:", loss)
            return [loss, outputs]


