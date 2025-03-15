import copy
import torch
import torch.nn as nn
import numpy as np
import random
from .base_MAGNN import MAGNN_ctr_ntype_specific

seed = 42
# torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class MAGNN_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.5,
                 deepSchema=False):
        super(MAGNN_lp, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.rnn_type = rnn_type

        # Feature transformation layers
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x

        # etype-specific parameters
        self.r_vec = None
        if rnn_type == 'RotatE0':
            self.r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, hidden_dim // 2, 2)))

        # Initialize r_vec if it's used
        if self.r_vec is not None:
            nn.init.constant_(self.r_vec, 1)

        # MAGNN layer for user and item
        self.user_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[0],
                                                   etypes_lists[0],
                                                   hidden_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   self.r_vec,
                                                   dropout_rate,
                                                   use_minibatch=True,
                                                   deepSchema=deepSchema)
        
        self.item_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[1],
                                                   etypes_lists[1],
                                                   hidden_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   self.r_vec,
                                                   dropout_rate,
                                                   use_minibatch=True,
                                                   deepSchema=deepSchema)

        # Fully connected layers for user and item logits
        self.fc_user = nn.Linear(hidden_dim * num_heads, out_dim, bias=True)
        self.fc_item = nn.Linear(hidden_dim * num_heads, out_dim, bias=True)

        # Merged layer for link prediction
        self.merged_pre = nn.Linear(out_dim * 2, 1, bias=True)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)

        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])

        transformed_features = self.feat_drop(transformed_features)

        # User and item layer outputs
        h_user = self.user_layer((g_lists[0], transformed_features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        logits_user = self.fc_user(h_user)
        
        h_item = self.item_layer((g_lists[1], transformed_features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))
        logits_item = self.fc_item(h_item)

        # Merge user and item logits for link prediction
        merged_logits = torch.cat((logits_user, logits_item), dim=-1)
        pre_LinkORNot = self.merged_pre(merged_logits)

        return pre_LinkORNot