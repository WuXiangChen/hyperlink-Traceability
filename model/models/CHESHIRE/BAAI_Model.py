import torch
import torch.nn as nn
from model.models.CHESHIRE.BAAI_Decoder import TransformerDecoder
from utils import Utils
import torch.nn.functional as F

class BAAI_model(nn.Module):
    def __init__(self, artifacts, tokenizer, model, freeze, with_knowledge, in_dim, latent_dim=1024):
        super(BAAI_model, self).__init__()
        self.model = model
        self.freeze = freeze
        self.with_knowledge = with_knowledge
        self.latent_dim = latent_dim
        if not self.with_knowledge:
            for layer in self.model.children():
                if hasattr(layer, 'weight'):
                    torch.nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
                    
        self.artifacts = artifacts
        self.tokenizer = tokenizer
        self.artifacts_dict = []
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_var = nn.Linear(in_dim, latent_dim)
        #self.fc_decode = nn.Linear(latent_dim, in_dim)
        
        self.linear = nn.Sequential(
            nn.Linear(in_dim, 1)
        )
        self.trans_decoder = TransformerDecoder(input_dim=in_dim, output_dim=in_dim)
        self.loss_fn = torch.nn.BCELoss()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, edges, labels=None):
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
        model_output = self.model(inputs, attention_masks, output_hidden_states=True, return_dict=True)
        output = model_output.pooler_output

        # VAE part
        mu = self.fc_mu(output)
        log_var = self.fc_var(output)
        z = self.reparameterize(mu, log_var)
        outputs = output + z

        outputs_ = outputs.detach()
        emb = model_output.hidden_states[0].detach()  # 第0层是嵌入层输出
        reconstructed_embedding = self.trans_decoder(emb, outputs_)
        loss_rec = self.trans_decoder.compute_loss(reconstructed_embedding, emb)

        outputs = self.linear(outputs)
        outputs = nn.Sigmoid()(outputs).squeeze(1)
        
        # 计算loss
        if labels is not None:
            loss = self.loss_fn(outputs, labels)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss_all = loss + loss_rec + kl_loss
            return [loss_all, outputs]
        return outputs