# 本节的主要目的是完成Unlabelled Artifact的预选择过程：
## 1. 根据给定的Unlabelled Artifact的特征，计算其与已有的Labeled Artifact Graph的“相似度”：这里的相似度的定义还未明确
## 2. 通过Random Ｗalk的方式，生成Labeled Graph的embedding，也即是Global Node的建立方式

'''
    导包区
'''
import networkx as nx
import random
import numpy as np
import os
from .models.base_MAGNN import MAGNN_metapath_specific
import torch
from numpy import argmax
from ._1_2LabeledArtifactGraphList import ArtifactGraphs
from typing import List
from ._0_Artifact import Artifact
import torch.nn as nn
from ._1_1generateGlobalNode import DeepWalk
from utils import Utils
from .models.GraphSAGEModel import GraphSAGE
from .models.RandomForestClassifierModel import RandomForestClassifierModel

class Preselector(nn.Module):
    def __init__(self,embedding_size: int = 300):
        super(Preselector, self).__init__()
        #magnn = MAGNN_metapath_specific(out_dim=embedding_size, num_heads=1, in_dim=2, attn_drop=0.3, alpha=0.01, num_layers=3, deepSchema=True)
        randomForest = RandomForestClassifierModel()
        self.model = randomForest

    def cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> float:
        dot_product = np.dot(A, B)  # 计算点积
        norm_A = np.linalg.norm(A)  # 计算向量 A 的范数
        norm_B = np.linalg.norm(B)  # 计算向量 B 的范数
        if norm_A == 0 or norm_B == 0:
            return 0.0  # 防止除以零
        return dot_product / (norm_A * norm_B)

    def generateGraphEmd(self):
        self.artifactGraphs.generate_graphs_embedding(self.artifactGraphs.graph_list)

    # 计算两个向量的相似度，来确定Artifact应该归属于哪一个图
    def forward(self, unlabelled_artifact_Tensor: torch.Tensor) -> (float, int):
        """
        Compute the similarity between the unlabelled artifact and the labeled artifact graph
        :param unlabelled_artifact: Unlabelled Artifact
        :return: similarity, index of the most similar graph
        """
        similaritys = self.model.fit(unlabelled_artifact_Tensor)
        return similaritys



