import os.path
import pickle
import shutil
import warnings
import random
from torch import nn

from utils import Utils
from .utils.utils import gen_global_identifier, transferArtIdIntoIndex
from .utils.Gen_datasets import prepare_datasets

warnings.filterwarnings("ignore")
import time
import argparse
import pandas as pd
import torch,json
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from .utils.pytorchtools import EarlyStopping
from .utils.data import load_LastFM_data
from .utils.tools import index_generator, parse_minibatch_LastFM, initialize_weights, split_train_validation
from .magnn_model.MAGNN_lp import MAGNN_lp

# Params
num_ntype = 4
dropout_rate = 0.3
weight_decay = 0.005
etypes_lists = [[[0], [1, 1], []],
                [[4, 4], [4,4], []]]
os.chdir(os.getcwd())

def test_process(net, art_adj, test_pos, test_neg, 
                 adjlists_ip, edge_metapath_indices_list_ip, 
                 features_list, type_mask, neighbor_samples, 
                 device, offsets, valFlag = True):
    net.eval()
    with torch.no_grad():
        test_pos_batch = test_pos.tolist()
        test_neg_batch = test_neg.tolist()

        test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
            art_adj, adjlists_ip, edge_metapath_indices_list_ip, test_pos_batch, device, neighbor_samples, offsets)
        test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
            art_adj, adjlists_ip, edge_metapath_indices_list_ip, test_neg_batch, device, neighbor_samples, offsets)
        
        Logits_pos_linkORNot = net((test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
        Logits_neg_linkORNot = net((test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
        pos_out = torch.sigmoid(Logits_pos_linkORNot)
        neg_out = torch.sigmoid(Logits_neg_linkORNot)

        # 合并正负样本的预测结果
        all_pre_label = torch.cat([pos_out, neg_out])
        if valFlag:
            all_true_label = torch.cat([torch.ones(pos_out.shape).to(device), torch.zeros(neg_out.shape).to(device)])
            val_loss = nn.BCELoss()(all_pre_label, all_true_label)
            val_loss = torch.mean(torch.tensor(val_loss))
            return val_loss
        else:
            y_proba_test = all_pre_label.cpu().numpy().reshape(-1)
            return y_proba_test
    
def train_process(net, art_adj, train_pos, train_neg, adjlists_ip, edge_metapath_indices_list_ip,
                       features_list, type_mask, neighbor_samples,  device, train_pos_idx_generator, offsets):
    net.train()
    train_pos_idx_batch = train_pos_idx_generator.next()
    train_pos_idx_batch.sort()
    train_pos_batch = train_pos[train_pos_idx_batch].tolist()

    train_neg_idx_batch = np.random.choice(len(train_neg), len(train_pos_idx_batch))
    train_neg_idx_batch.sort()
    train_neg_batch = train_neg[train_neg_idx_batch].tolist()
    train_pos_g_lists, train_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(art_adj, adjlists_ip, edge_metapath_indices_list_ip,
                                                                                    train_pos_batch, device, neighbor_samples, offsets) 

    train_neg_g_lists, train_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(art_adj, adjlists_ip, edge_metapath_indices_list_ip,
                                                                                    train_neg_batch, device, neighbor_samples, offsets)
    
    Logits_neg_linkORNot = net((train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, val_neg_idx_batch_mapped_lists))
    Logits_pos_linkORNot = net((train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, val_pos_idx_batch_mapped_lists))
    
    pos_out = F.sigmoid(Logits_pos_linkORNot)
    neg_out = F.sigmoid(Logits_neg_linkORNot)
    all_pre_label = torch.cat([pos_out, neg_out])
    all_true_label = torch.cat([torch.ones(pos_out.shape).to(device), torch.zeros(neg_out.shape).to(device)])
    loss_A = nn.BCELoss()(all_pre_label, all_true_label)
    train_loss = torch.mean(loss_A)
    return train_loss

def loadTrainedModel(netModelPath, name=None, i=1):
    net = None
    if os.path.exists(netModelPath) and os.listdir(netModelPath)!=[]:
        if name is not None:
            modelPath_ = name
        else:
            files = [f for f in os.listdir(netModelPath) if os.path.isfile(os.path.join(netModelPath, f))]
            sorted_files = sorted(files)
            modelPath_ = os.path.join(netModelPath, sorted_files[-i])
        net = torch.load(modelPath_)
    return net, modelPath_

def loadTrainedSpecificModel(netModelPath, name=None, i=1):
    net = None
    if os.path.exists(netModelPath):
        net = torch.load(netModelPath)
    else:
        raise "ERROR IN LOADING MODEL"
    return net

def run_model(repoName, feats_type, hidden_dim, out_dim, num_heads,
                     attn_vec_dim, rnn_type, batch_size, 
                     neighbor_samples, save_postfix, cudaN, deepSchema=False, 
                     num_users=None, num_repos=None, num_issues=None, num_prs=None, lr=0.03,
                     dataset=None, TestFlag=False, art_adj=None, k=1):
    # 制备数据集过程
    (adjlists_ip, edge_metapath_indices_list_ip, features_list, type_mask) = load_LastFM_data(repoName, num_issues)
    device = torch.device(cudaN if torch.cuda.is_available() else 'cpu')
    print("deviced used for training:", str(device))
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    # 这一句话就是将feature转译成floattensor的形式，并将其转移到device上

    in_dims = []
    if feats_type == 0: in_dims = [features.shape[1] for features in features_list]

    partial_offsets = [num_users + num_repos, num_users + num_repos + num_issues]
    pos_ = np.array(dataset["pos"])
    neg_ = np.array(dataset["neg"])

    if deepSchema:
        netModelPath = 'checkpoint/deepSchema/{}/{}/{}'.format(rnn_type,repoName,k)
    else:
        netModelPath = 'checkpoint/normalSchema/{}/{}/{}'.format(rnn_type,repoName,k)
    
    # 建立节点的全局标识，以及其对应的partially observable的邻接信息
    if not TestFlag:
        offsets = [num_users, num_repos, num_issues, num_prs]
        art_adj = gen_global_identifier(offsets, pos_)
        train_pos, train_neg, valid_pos, valid_neg = split_train_validation(pos_, neg_)
        net = MAGNN_lp([3,3],  6, etypes_lists, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate, deepSchema)
        initialize_weights(net)
        net.train()

        if not os.path.exists(netModelPath):
            os.makedirs(netModelPath)
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos))

        print(train_pos_idx_generator.num_iterations())
        net.to(device)
        train_loss = train_process(net, art_adj, train_pos, train_neg, adjlists_ip, edge_metapath_indices_list_ip,
                      features_list, type_mask, neighbor_samples,  device, train_pos_idx_generator, partial_offsets)
        return art_adj, train_loss
    else:
        try:
            # loaded_net = loadTrainedSpecificModel(netModelPath=locate_)
            loaded_net, _ = loadTrainedModel(netModelPath=netModelPath, i=-1)
        except Exception as e:
            print(e)
            return [[0, 0], [0, 0]]
        if loaded_net is not None:
            net = loaded_net

        y_proba_test = test_process(net, art_adj, pos_, neg_, adjlists_ip, edge_metapath_indices_list_ip, features_list, type_mask, neighbor_samples, device, partial_offsets, valFlag=False)
        y_true_test = torch.cat([torch.ones(len(pos_)).to(device), torch.zeros(len(neg_)).to(device)]).cpu().numpy().reshape(-1)
        y_proba_test = np.where(y_proba_test >= 0.5, 1, 0).reshape(-1)
        cm = confusion_matrix(y_true_test, y_proba_test)

        return cm

from types import SimpleNamespace

def parse_args():
    """
    Return the arguments as a SimpleNamespace object.
    """
    # Define default values using a dictionary
    args_dict = {
        'feats_type': 0,
        'r': 'saltstack',
        'hidden_dim': 2,
        'cudaN': 'cuda:0',
        'out_dim': 2,
        'num_heads': 2,
        'attn_vec_dim': 64,
        'rnn_type': 'RotatE0',
        'samples': 10,
        'repeat': 1,
        'save_postfix': 'checkpoint',
        'deepSchema': True
    }

    # Convert dictionary to SimpleNamespace
    args = SimpleNamespace(**args_dict)
    
    return args

def run_process_for_p2pMoudle(P2Pdatasets, P2PLabels, testFlag=False, reponame="symfony", art_adj=None, k=1):
    """
    Method to perform testing for the recommendation dataset as specified in the arguments.
    """
    args = parse_args()
    # Set the basic repository path
    basic_repo = f'../dataset/P2PDataSupport/{reponame}/'

    # Load the dataset files
    mapping = pd.read_json(basic_repo + 'mapping.json').T
    issues = mapping[mapping.iloc[:, 1].str.contains('issue', case=True, na=False)]
    prs = mapping[mapping.iloc[:, 1].str.contains('pr', case=True, na=False)]
    users = pd.read_csv(basic_repo + '/Index/user_index.txt', sep='\t', header=None, names=['user_id', 'user'], keep_default_na=False, encoding='utf-8')
    repos = [1]

    # Get the number of users, repos, issues, and PRs
    num_users = len(users)
    num_repos = len(repos)
    num_issues = len(issues)
    num_prs = len(prs)
    lr = 0.008
    if not isinstance(P2Pdatasets, list):
        P2Pdatasets = P2Pdatasets.values()
    ## 这里需要插入一个方法 将原始的ArtId转换为index 再转换为adjM中的相应序位
    dataset = transferArtIdIntoIndex(P2Pdatasets, P2PLabels, mapping)
    
    rootPath = f'middleP2PDatasets/{reponame}/{k}/'
    file_path = rootPath + f'dataset_Testing_{str(testFlag)}.pkl'
    if not os.path.exists(rootPath):
        os.makedirs(rootPath)
        
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)

    # Call the model running function with the arguments
    return run_model(
        reponame, args.feats_type, args.hidden_dim, args.out_dim, args.num_heads,
        args.attn_vec_dim, args.rnn_type, args.samples, args.save_postfix, args.cudaN, args.deepSchema, num_users, 
        num_repos, num_issues, num_prs, lr, dataset, TestFlag=testFlag,art_adj=art_adj, k=k
    )
