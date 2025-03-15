from unittest import result
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import scipy.sparse
import pickle
import pandas as pd
from .tools import read_pickle_file, read_adjlist_file

# FIXME: 本节随机性已经得到控制

def extendAdjMIndex(resultLs, offset):
    resultDict = {}
    for idx, re in enumerate(resultLs):
        re = list(map(int, re.split(' ')))
        re = list(map(lambda x: x + offset, re))
        idx = idx+offset
        resultDict[idx] = re
    return resultDict


def load_LastFM_data(repoName, num_issues):
    prefix = f'../dataset/P2PDataSupport/{repoName}/processed'
    basic_repo = f'../dataset/P2PDataSupport/{repoName}/'
    users = pd.read_csv(basic_repo + '/Index/user_index.txt', sep='\t', header=None, names=['user_id', 'user'], keep_default_na=False, encoding='utf-8')
    num_users = len(users)
    num_repo = 1
    offsets = [num_users+num_repo,num_users+num_repo+num_issues]

    result = {}

    result['adjlist01'] = read_adjlist_file(f"{prefix}/2/2-0-2.adjlist")
    result['adjlist08'] = read_adjlist_file(f"{prefix}/3/3-0-3.adjlist")

    result['idx01'] = read_pickle_file(f"{prefix}/2/2-0-2_idx.pickle")
    result['idx05'] = read_pickle_file(f"{prefix}/3/3-0-3_idx.pickle")

    ##### 传入节点特征
    # user的信息是通过随机数值表示，剩下三个应该就是分别是repo、issue和PR的特征表示了
    # 这里面所加载的feature都已经是降至50维的信息
    features_0 = np.random.rand(num_users, 10)
    features_1 = np.load(prefix + '/feature_1.npy')
    features_2 = np.load(prefix + '/feature_2.npy')
    features_3 = np.load(prefix + '/feature_3.npy')
    
    type_mask = np.load(prefix + '/node_types.npy')
    return [result['adjlist01'],
            result['adjlist08']],\
           [result['idx01'],
            result['idx05']], \
           [features_0, features_1, features_2, features_3], \
            type_mask