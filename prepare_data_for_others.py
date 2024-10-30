# 本节的主要目的是将已有工作中的数据准备好，保持相同的数据格式，以备后期使用
## 主要的数据类型包括，关联矩阵数据，邻接矩阵数据
## 还需要包括其中的假阳性数据
import itertools
import os
from typing import List
import glob

import networkx as nx
import numpy as np
import pandas as pd
from wandb.cli.cli import artifact
from DataStructure._1_HyperlinkGenerate import generateHyperLinkDataset
from scipy.sparse import csr_matrix, save_npz
from utils import Utils
from scipy.io import savemat
class preDataForOtherMethods:
    def __init__(self):
        self.data_path = 'dataset/other_repo_connect_info/'
        self.global_nodes = 0

    # 共需要准备三种数据类型，分别是： 1.关联矩阵数据  2.邻接矩阵数据
    def generateHyperLinkIncidenceDataset(self, filteredJson:List[List], repoName):
        G = nx.Graph()
        artifact_set = set()
        for source, target in filteredJson:
            artifact_set.add(source)
            artifact_set.add(target)
            G.add_edge(source, target)
        # 将self.G转成无向图
        G = G.to_undirected()
        hyperlinks = [list(x) for x in nx.connected_components(G)]
        self.global_nodes += len(artifact_set)
        print(f'Number of artifacts: {len(artifact_set)}')
        artifact_dict = {artifact:idx for idx, artifact in enumerate(artifact_set)}

        ghl = generateHyperLinkDataset(artifact_set, hyperlinks)
        posHyperlink, negHyperlink, negative_adjacents, full_posHyperlink = ghl.generateHyperLink(1, artifact_dict,0.1, 0.2)
        return posHyperlink, negHyperlink, negative_adjacents, artifact_dict, full_posHyperlink

    def generateAdjacencyMatrixDataset(self, pair_wise:List[List], negative_adjacents, artifact_dict):
        n = len(artifact_dict)
        adj_matrix = np.zeros((n, n))
        for pair in pair_wise:
            adj_matrix[artifact_dict[pair[0]], artifact_dict[pair[1]]] = 1
            adj_matrix[artifact_dict[pair[1]], artifact_dict[pair[0]]] = 1
        positive_adjacents = adj_matrix
        negative_adjacents = negative_adjacents
        return positive_adjacents, negative_adjacents

if __name__ == '__main__':
    preOtherM = preDataForOtherMethods()
    data_path_pattern = preOtherM.data_path+'/graph_*'
    projectName = glob.glob(data_path_pattern)
    for project in projectName:
        repoName = project.split('_')[-1].split('-')[0]
        repoDataJson = Utils.loadJson(f'{project}')["links"]
        filteredJson = [[data["source"],data["target"]] for data in repoDataJson]
        posHyperlink, negHyperlink, negative_adjacents, artifact_dict, full_posHyperlink \
            = preOtherM.generateHyperLinkIncidenceDataset(filteredJson, repoName)
        positive_adjacents, negative_adjacents = preOtherM.generateAdjacencyMatrixDataset(filteredJson, negative_adjacents, artifact_dict)
        ### 保存成mat文件
        # Model = {
        #     'S': posHyperlink,
        #     'U': negHyperlink
        # }
        # mat_fileName = os.path.join("dataset/all_used_dataset/mat/", repoName+".mat")
        # savemat(mat_fileName, {'Model': Model})

        # 保存成关联矩阵文件
        inci_fileName = os.path.join("dataset/all_used_dataset/hyperlink_npz/", repoName)
        # np.savez(inci_fileName, positive=posHyperlink, negative=negHyperlink, full_posHyperlink=full_posHyperlink, artifact_dict = artifact_dict)
        # print(f"Saved incidence matrix for {repoName}")

        # 保存成邻接矩阵文件
        adg_fileName = os.path.join("dataset/all_used_dataset/adjacent_npz/", repoName)
        # 改成稀疏矩阵的形式再保存
        positive_sparse = csr_matrix(positive_adjacents)
        negative_sparse = csr_matrix(negative_adjacents)

        save_npz(adg_fileName, positive_sparse, negative_sparse)
        # print(f"Saved adjacency matrix for {repoName}")
        print("All nodes: ", preOtherM.global_nodes)