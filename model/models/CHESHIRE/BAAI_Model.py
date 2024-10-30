import torch
import torch.nn as nn
from utils import Utils
import torch.nn.functional as F

class BAAI_model(nn.Module):
    def __init__(self, artifacts_dict, artifacts, tokenizer, model, freeze, with_knowledge, in_dim):
        super(BAAI_model, self).__init__()
        self.model = model
        self.freeze = freeze
        self.with_knowledge = with_knowledge

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
            # nn.Linear(384, 384),
            # nn.ReLU(),
            nn.Linear(in_dim, 1)
        )
        # self.linear = nn.Linear(1024, 1)
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, edges, labels=None):
        # 输出每一列中的非0元素的索引
        node_sentence_list, nodes_list = Utils.process_edges_data(self.artifacts, edges)
        inputs = []
        attention_masks = []
        for NL_PL in node_sentence_list:
            NL_PL = "".join(NL_PL)
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
        outputs = self.model(inputs,attention_masks).pooler_output
        outputs = self.linear(outputs)
        outputs = nn.Sigmoid()(outputs).squeeze(1)
        # 计算loss
        if labels is not None:
            loss = self.loss_fn(outputs, labels)
            return [loss, outputs]