import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.conv import SAGEConv
import numpy as np
import random

seed = 41
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class MAGNN_metapath_specific(nn.Module):
    def __init__(self,
                 etypes,
                 out_dim,
                 num_heads,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.3,
                 alpha=0.01,
                 use_minibatch=False,
                 attn_switch=True,
                 num_layers=10, #3
                 deepSchema=False):

        super(MAGNN_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.etypes = etypes
        self.r_vec = r_vec
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch
        
        if self.attn_switch:
            self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
            nn.init.xavier_normal_(self.attn2, gain=1.441)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(0.3)
        else:
            self.attn_drop = lambda x: x

        self.num_layers = num_layers
        self.deepSchema = deepSchema
        # 定义 SAGEConv 层
        if self.deepSchema:
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(out_dim, out_dim//2, 'mean'))
            for _ in range(1, num_layers):
                self.convs.append(SAGEConv(out_dim//2, out_dim//2, 'mean'))
            self.fc_sage = nn.Linear(out_dim//2, out_dim)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # print("attention: ", attention[2][:3])
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, inputs):
        g, features, type_mask, edge_metapath_indices, target_idx = inputs
        edata = F.embedding(edge_metapath_indices, features)
        if self.rnn_type == 'RotatE0' or self.rnn_type == 'RotatE1':
            # print("self.r_vec: ", self.r_vec)
            r_vec = F.normalize(self.r_vec, p=2, dim=2)
            # print("original_r_vec: ", r_vec)
            if self.rnn_type == 'RotatE0':
                r_vec = torch.stack((r_vec, r_vec), dim=1)
                r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            # print("original_edata: ", edata[2][:3])
            final_r_vec = torch.zeros([edata.shape[1], self.out_dim // 2, 2], device=edata.device)
            final_r_vec[-1, :, 0] = 1 # 这里不应该是-3
            # print(final_r_vec.shape)
            # print("r_vec: ", r_vec)
            for i in range(final_r_vec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] -  final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
                else:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
            # print("final_r_vec: ", final_r_vec)
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] +edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2 # 这里就表明 第一维的数据经过指定旋转，变成第二维
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            # print("edata: ", edata[2][:3])
            hidden = torch.mean(edata, dim=1) # 如果旋转是对的，那么这里的数据均值 将与最后一维的数据相同，差距越小说明这个旋转越对，也就表示转换关系学习的越好
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim
        if self.attn_switch:
            center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # E x out_dim
            a1 = self.attn1(center_node_feat)  # E x num_heads
            a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
            a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1
        else:
            a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)

        # 这里图g 和 前文计算是如何关联的，这里的g是怎么和前文的g关联的？
        g = g.to(a.device) # 这里不要加这个，否则会报错，先禁用gpu
        g.edata.update({'eft': eft, 'a': a})
        self.edge_softmax(g)
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ## 从实验结果可以看出来，损失下不去。我初步判断是这里的卷积过程太浅了，需要加深一点
        ### 加深的过程可以从图本身出发 先尝试增加多个图卷积层
        h = g.ndata["ft"]
        # E x num_heads x out_dim
        # print("h: ", h[2][:3])
        if self.deepSchema:
            for i in range(self.num_layers):
                h_in = h  # 保存输入以便进行残差连接
                h = self.convs[i](g, h)  # 进行卷积操作
                h = F.relu(h)  # 激活函数
                if i > 0:  # 从第二层开始添加残差连接
                    h = torch.add(h,h_in)  # 残差连接
                    h = F.relu(h)  # 激活函数
            h = self.fc_sage(h)
        ret = g.ndata['ft'] = h  # E x num_heads x out_dim
        if self.use_minibatch:
            return ret[target_idx]
        else:
            return ret


class MAGNN_ctr_ntype_specific(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 use_minibatch=False,
                 deepSchema=False):
        super(MAGNN_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(MAGNN_metapath_specific(etypes_list[i],
                                                                out_dim,
                                                                num_heads,
                                                                rnn_type,
                                                                r_vec,
                                                                attn_drop=attn_drop,
                                                                use_minibatch=use_minibatch,
                                                                deepSchema=deepSchema))

        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs
            metapath_outs = []
            for g, edge_metapath_indices, target_idx, metapath_layer in zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers):
                output = metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,self.num_heads * self.out_dim)
                metapath_outs.append(F.elu(output))

        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h
