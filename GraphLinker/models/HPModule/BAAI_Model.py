import torch
import torch.nn as nn
from GraphLinker.models.HPModule.BAAI_Decoder import TransformerDecoder
from utils import Utils
import torch.nn.functional as F
from GraphLinker.generateHyperLinkFeature import generateHyperLinkFeature
from GraphLinker.models.AMHEN.GAT_Multiplex import MultiGaT

class BAAI_model(nn.Module):
    def __init__(self, artifacts, tokenizer, model,  in_dim, latent_dim=1024, training_type=1, max_length = None, test_k_index=0, hp_hiddenDList=[], hp=False):
        super(BAAI_model, self).__init__()
        self.model = model
        self.latent_dim = latent_dim
        self.gHPFea = generateHyperLinkFeature()
        self.test_k_index = test_k_index
        self.hp = hp
                    
        self.artifacts = artifacts
        # 这里还有一个需求，就是找出artifacts中的max_length
        max_len = max_length
        self.stru_max_len = max_len
        if training_type>1 and self.test_k_index!=0:
            self.gatMuti = MultiGaT(hp_hiddenD = hp_hiddenDList)

        self.tokenizer = tokenizer
        self.training_type = training_type
        if test_k_index==0 or  training_type==1:
            final_out_dim = latent_dim
        elif training_type==2:
            final_out_dim = 128
        else:
            final_out_dim = latent_dim + 128
        
        self.linear = nn.Sequential(nn.Linear(final_out_dim, 1))
        self.trans_decoder = TransformerDecoder(input_dim=in_dim, seq_length=128, output_dim=in_dim)
        self.loss_fn = torch.nn.BCELoss()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=final_out_dim, num_heads=8)

    def setP2PInfo(self, p2pModelTrain, P2Pdatasets, P2PLabels, reponame=None, k=None):
        self.p2pModelTrain = p2pModelTrain
        self.P2Pdatasets = P2Pdatasets
        self.P2PLabels = P2PLabels
        self.reponame = reponame
        self.k = k
    
    def getArt_adj(self):
        return self.art_adj

    def setArt_adj(self, art_adj):
        self.art_adj = art_adj

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, edges, labels=None):
        node_sentence_list, _ = Utils.process_edges_data(self.artifacts, edges)
        sta_outputs = None
        stru_outputs = None
        semantic_outputs = None

        if self.training_type>1 and self.test_k_index!=0:
            # 这里已经的得到所有的结构特征，接下来是通过这里的结构特征建图+多级图卷积
            data_graph = self.gHPFea.generateHyperLinkFeature(edges, self.artifacts) #  这里统计特征也就算是加入了
            stru_outputs = self.gatMuti(data_graph).to(self.model.device)

        if self.training_type!=2 and self.test_k_index!=0:
            inputs = []
            attention_masks = []
            for NL_PL in node_sentence_list:
                NL_PL = "".join(NL_PL)
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
            model_output = self.model(inputs, attention_masks, output_hidden_states=True, return_dict=True)
            semantic_outputs = model_output.pooler_output

            outputs_ = semantic_outputs.detach()
            emb = model_output.hidden_states[0].detach()  # 第0层是嵌入层输出
            reconstructed_embedding = self.trans_decoder(emb, outputs_)
            loss_rec = self.trans_decoder.compute_loss(reconstructed_embedding, emb)

        tensors_to_concat = []
        if semantic_outputs is not None:
            tensors_to_concat.append(semantic_outputs)
        if stru_outputs is not None:
            tensors_to_concat.append(stru_outputs)
        # 检查 sta_outputs 是否存在
        if sta_outputs is not None:
            tensors_to_concat.append(sta_outputs)
        if len(tensors_to_concat)==0:
            raise "The Input Feature is None, Please set at least one feature."

        # 合并所有存在的张量
        all_output = torch.cat(tensors_to_concat, dim=1)
        # all_output,_ = self.multihead_attn(all_output,all_output,all_output)
        outputs = self.linear(all_output)
        outputs = nn.Sigmoid()(outputs).squeeze(1)
        

        # 进行P2P的预测
        p2p_loss = None
        if not self.hp:
            art_adj, p2p_loss = art_adj = self.p2pModelTrain(self.P2Pdatasets, self.P2PLabels, reponame=self.repoName, k=self.k)
            self.setArt_adj(art_adj)

        # 计算loss
        if labels is not None:
            loss_all = self.loss_fn(outputs, labels)
            # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            if self.training_type!=2:
                loss_all = loss_all + loss_rec
                if p2p_loss != None:
                    loss_all = 0.5*loss_all + 0.5*p2p_loss
            return [loss_all, outputs]
        return outputs