# 本节的主要目的是实现一个基于不确定图的训练框架，该框架可以用于训练任何基于图的模型，包括GCN、GAT等。

'''
    导包区
'''
from utils import Utils
import torch
import torch.nn as nn
from typing import List
import networkx as nx
import torch.nn.functional as F
from model.resource._0_Artifact import Artifact
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from model.resource._2_1CompleteGraphDefine import CompleteGraph

class UncertainGraphTraining(nn.Module):
    # groundTruth是一个跟随实验结果动态生成的图模型
    def __init__(self, feature_dim: int, in_channels: int = 300, hidden_channels: int = 526, out_channels: int = 10, num_heads: int =3):
        super(UncertainGraphTraining, self).__init__()

        self.feature_dim = feature_dim
        self.cg = CompleteGraph()
        self.csg = self.cg.standard_graph
        self.groundTruth = None
        self.query_layer = nn.Linear(feature_dim, feature_dim)
        self.key_layer = nn.Linear(feature_dim, feature_dim)
        self.value_layer = nn.Linear(feature_dim, feature_dim)
        # 融合后的线性层
        self.fusion_layer = nn.Linear(feature_dim, feature_dim)

        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1)

    # 生成groundTruth
    def set_groundTruth(self, labeled_artifacts: nx.Graph):
        self.groundTruth = labeled_artifacts

    def getCorrespondFeature(self, node: int or tuple, graph: nx.Graph):
        # 从graph中获取节点node的特征
        return graph.nodes[node]
    def graphMapping(self, subGraphs: List[nx.Graph],unlabeled_artifacts: List[Artifact]):
        # 将subGraphs映射到标准图上
        mapping_lists = []
        for subGraph in subGraphs:
            mapping_lists.append(self.cg.get_mapping_list(subGraph))

        # 重置已用节点
        self.cg.initialized_used_nodes()

        # 这里的mapping_lists是一个列表，列表中存储的是每个需要进行特征替换的节点映射表
        ## 每个需要被映射的特征集合，都需要经过GAT进行特征融合
        total_pairs = sum(len(mpG) for mpG in mapping_lists)  # 计算总的节点对数量
        feature_dim = self.feature_dim  # 特征维度，根据您的具体情况定义
        # 初始化张量
        feature_sub = torch.empty((total_pairs, feature_dim))  # (N, feature_dim)
        feature_com = torch.empty((total_pairs, feature_dim))

        index = 0  # 用于跟踪当前插入的位置
        for i, mpG in enumerate(mapping_lists):
            for node_pair in mpG:
                # 获取对应的特征
                subNode = node_pair[0]
                comNode = node_pair[1]
                # 使用 GAT 进行特征融合
                subNode_feature = self.getCorrespondFeature(subNode, subGraphs[i])
                comNode_feature = self.getCorrespondFeature(comNode, self.csg)

                # 将特征赋值到张量中
                feature_sub[index] = subNode_feature["ft"]
                feature_com[index] = comNode_feature["ft"]
                index += 1  # 更新索引

        unlabeledArt_mapping_list = {}
        for i, mappingPair in enumerate(mapping_lists):
            for artifact in unlabeled_artifacts:
                for artifactMap in mappingPair:
                    if artifact.getArtifactId() == artifactMap[0]:
                        unlabeledArt_mapping_list[(i, artifact.getArtifactId())] = artifactMap[1]
                        break

        return feature_sub, feature_com, mapping_lists, unlabeledArt_mapping_list

    def compute_attention(self, A_flat, B_flat, batch_size):
        # 计算注意力权重
        query = self.query_layer(A_flat)  # (batch_size, 90000)
        key = self.key_layer(B_flat)  # (batch_size, 90000)
        value = self.value_layer(B_flat)  # (batch_size, 90000)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(1, 0))  # (batch_size, 90000) x (90000, batch_size)
        attention_scores = torch.softmax(attention_scores / (self.feature_dim ** 0.5), dim=-1)  # 缩放并归一化

        # 计算加权值
        attention_output = torch.matmul(attention_scores, value)  # (batch_size, 90000)

        # 融合特征
        fused_output = A_flat + attention_output  # (batch_size, 90000)

        # 通过融合层
        fused_output = self.fusion_layer(fused_output)

        # 将输出恢复为 (batch_size, 300, 300)
        fused_output = fused_output.view(batch_size, self.feature_dim)

        return fused_output

    def get_connected_node_pairs(self, unlabeled_art_mapping_list):
        node_pairs = []
        for artifactMap in unlabeled_art_mapping_list:
            cg_node = unlabeled_art_mapping_list[artifactMap]
            connected_nodes = list(self.csg.neighbors(cg_node))
            for node in connected_nodes:
                node_pairs.append((cg_node, node))
        return node_pairs

    def linkTrain(self, data: Data):
        x = self.gat1(data.x, data.edge_index)
        x = F.elu(x)
        x = self.gat2(x, data.edge_index)
        return x

    def compute_link_score(self, embeddings, node_pair):
        node_a = node_pair[0]
        node_b = node_pair[1]
        emb_a = embeddings[node_a]
        emb_b = embeddings[node_b]
        score = torch.dot(emb_a, emb_b)
        return score

    def getTargetMapping(self, mapping_lists, sourceDict):
        targetMappedIndex = {}
        sourceIndexvalues = list(sourceDict.values())

        for mappingPair in mapping_lists:
            for node_pair in mappingPair:
                if node_pair[0] != -1 and node_pair[1] not in sourceIndexvalues:
                    targetMappedIndex[node_pair[1]] = node_pair[0]

        return targetMappedIndex

    def forward(self, subGraphs: List[nx.Graph], unlabeled_artifacts: List[Artifact]):
        # 生成groundTruth
        feature_sub, feature_com, mapping_lists, unlabeledArt_mapping_list = self.graphMapping(subGraphs, unlabeled_artifacts)
        # 计算batch_size
        batch_size = feature_sub.size(0)
        feature_com_flat = feature_com.view(batch_size, -1)  # (batch_size, 90000)
        feature_sub_flat = feature_sub.view(batch_size, -1)  # (batch_size, 90000)
        new_feature = self.compute_attention(feature_sub_flat, feature_com_flat, batch_size)
        # 然后将new_feature赋予standardGraph的对应节点中
        self.cg.change_node_feature(mapping_lists, new_feature)
        data = Utils.transNxGraphToTorch(self.csg)

        unlabeled_node_pairs = self.get_connected_node_pairs(unlabeledArt_mapping_list)
        # 计算链接得分
        embeddings = self.linkTrain(data)

        targetMappedIndex = self.getTargetMapping(mapping_lists, unlabeledArt_mapping_list)
        targetMappedKeys = list(targetMappedIndex.keys())
        scoresMapped = {}
        scores = []
        for node_pair in unlabeled_node_pairs:
            score = self.compute_link_score(embeddings, node_pair)
            targetNode = node_pair[1]
            if targetNode in targetMappedKeys:
                scoresMapped[targetMappedIndex[targetNode]] = score
            scores.append(score)

        return scoresMapped, scores












