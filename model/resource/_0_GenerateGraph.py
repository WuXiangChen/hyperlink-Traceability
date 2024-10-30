# 本节的主要目的是依据现有的artifact信息和linkingParis信息，构建一个图结构，用于后续的训练。

'''
    导包区
'''

import random
from utils import Utils
import networkx as nx
import numpy as np


class generateGraph:
    def __init__(self, artifacts: list, repoName: str):
        self.artifacts = artifacts
        self.repoName = repoName
        self.Full_G = nx.Graph()
        self.Train_G = nx.Graph()
        self.Test_G = nx.Graph()
        self.utils = Utils(repoName)
        self.linkpairs, self.train_linkingpairs = self.getLinkingPairs()
        self.artDict = self.getArtifactDict(self.artifacts)
        self.train_artifacts = {}
        self.test_artDict = {}

    def getLinkingPairs(self):
        filePath = "dataset/"+self.repoName + "/originalDataSupport/linkingPairs.json"
        linkingpairs = self.utils.load_json(filePath)

        filePath = "dataset/"+self.repoName + "/processed/train.json"
        train_linkingpairs = self.utils.load_json(filePath)
        return linkingpairs, train_linkingpairs

    def getArtifactDict(self, artifacts: list):
        artDict = {}
        for art in artifacts:
            artId = art.getArtifactId()
            artDict[artId] = art
        return artDict

    def generateTrainGraph(self, train_artifacts: list):
        self.train_artifacts = self.getArtifactDict(train_artifacts)
        for linkingpair in self.linkpairs:
            source =list(linkingpair.keys())[0]
            target = list(linkingpair.values())[0]
            if int(source) not in self.train_artifacts.keys() or int(target) not in self.train_artifacts.keys():
                continue
            source = int(source)
            target = int(target)
            source_features = self.getNodeFeature(source)
            target_features = self.getNodeFeature(target)
            self.Train_G.add_node(source, **source_features)
            self.Train_G.add_node(target,**target_features)
            self.Train_G.add_edge(source, target)
        #isolated_nodes = [node for node in self.Train_G.nodes() if self.Train_G.degree(node) == 0]
        return self.Train_G

    def generateTestGraph(self, test_artifacts: list):
        if self.Train_G.number_of_nodes() == 0:
            raise ValueError("The Train_G has not been generated yet!")
        self.test_artDict = self.getArtifactDict(test_artifacts)
        for art in self.test_artDict.items():
            art_id = art[0]
            art_value = art[1]
            if art_id in self.Train_G.nodes():
                continue
            artFeatures = self.getNodeFeature(art_id)
            self.Test_G.add_node(art_id, **artFeatures)
        # 合并Train_G和Test_G
        self.Test_G = nx.disjoint_union(self.Train_G, self.Test_G)
        return self.Test_G

    def generateFullGraph(self):
        for linkingpair in self.linkpairs:
            source = int(list(linkingpair.keys())[0])
            target = int(list(linkingpair.values())[0])
            if source not in self.artDict.keys() or target not in self.artDict.keys():
                continue
            source_features = self.getNodeFeature(source)
            target_features = self.getNodeFeature(target)
            self.Full_G.add_node(source, **source_features)
            self.Full_G.add_node(target, **target_features)
            self.Full_G.add_edge(source, target)
        return self.Full_G

    def getNodeFeature(self, node_id):
        features = {
            'ft': self.artDict[node_id].getArtifactFeature("ft"),
            'createdAt': self.artDict[node_id].getArtifactFeature("createdAt"),
            'fileName': self.artDict[node_id].getArtifactFeature("fileName")
        }
        return features

    def getComponents(self, G: nx.Graph):
        # 获取连接的组件
        components = nx.connected_components(G)
        # 返回每个组件的子图
        # split_components = self.split_components()
        subgraphs = {i : G.subgraph(component).copy() for i, component in enumerate(components)}
        return subgraphs

    def split_components(self, target_groups=2):
        # 获取所有连通分量
        components = list(nx.connected_components(self.G))
        # 如果组件数量小于或等于目标组数，直接返回
        if len(components) <= target_groups:
            return components
        # 计算每个分量的大小
        component_sizes = np.array([len(comp) for comp in components])
        # 初始化结果
        result = [set() for _ in range(target_groups)]
        current_sums = np.zeros(target_groups)  # 每组的当前总大小
        # 对组件按照大小从大到小排序
        sorted_indices = np.argsort(-component_sizes)
        # 贪心地将组件分配到每组
        for idx in sorted_indices:
            # 找到当前总和最小的组
            min_group_index = np.argmin(current_sums)
            # 将组件添加到该组
            result[min_group_index] = result[min_group_index].union(components[idx])
            current_sums[min_group_index] += component_sizes[idx]
        return result

class generateNodePair:
    def __init__(self, train_artifacts: list, test_artifacts: list, repoName: str):
        self.repoName = repoName
        self.utils = Utils(repoName)
        self.train_artDict = self.getArtifactDict(train_artifacts)
        self.test_artDict = self.getArtifactDict(test_artifacts)
        self.train_linkingpairs = self.getLinkingPairs(self.train_artDict)
        self.test_linkingpairs = self.getLinkingPairs(self.test_artDict)

    def getLinkingPairs(self, artDict):
        filePath = "dataset/"+self.repoName + "/originalDataSupport/linkingPairs.json"
        con = self.utils.load_json(filePath)
        linkingpairs = []
        for i in range(len(con)):
            node1 = int(list(con[i].values())[0])
            node2 = int(list(con[i].keys())[0])
            if node1 in artDict.keys() and node2 in artDict.keys():
                linkingpairs.append([node1, node2])
        return linkingpairs

    def getArtifactDict(self,artifacts: list):
        artDict = {}
        for art in artifacts:
            artId = art.getArtifactId()
            artDict[artId] = art
        return artDict

    def generateNodePairs(self, ratio=1):
        positive_pairs = self.train_linkingpairs
        negative_pairs = []
        artifacts = list(self.train_artDict.keys())
        ran1 = random.choices(artifacts, k=int(ratio * len(positive_pairs)))
        ran2 = random.choices(artifacts, k=int(ratio * len(positive_pairs)))
        for i in range(len(ran1)):
            source = ran1[i]
            target = ran2[i]
            if source == target:
                continue
            if [source, target] in positive_pairs or [target, source] in positive_pairs:
                continue
            negative_pairs.append([source, target])

        return positive_pairs, negative_pairs

    def generateTrainingNodePairs(self):
        # 1 生成训练集
        positive_pairs, negative_pairs = self.generateNodePairs()
        # 2. 生成标签
        pos_labels = [1] * len(positive_pairs)
        neg_labels = [0] * len(negative_pairs)
        pos_emb = []
        # 3. 生成数据集
        for pos_index in range(len(positive_pairs)):
            node1_index = positive_pairs[pos_index][0]
            node2_index = positive_pairs[pos_index][1]
            pos_emb.append([self.train_artDict[node1_index], self.train_artDict[node2_index]])
        neg_emb = []
        for neg_index in range(len(negative_pairs)):
            node1_index = negative_pairs[neg_index][0]
            node2_index = negative_pairs[neg_index][1]
            neg_emb.append([self.train_artDict[node1_index], self.train_artDict[node2_index]])
        all_emb = pos_emb + neg_emb
        all_labels = pos_labels + neg_labels
        return all_emb, all_labels

    def generateTestingNodePairs(self):
        all_artifactDict = {}
        all_artifactDict.update(self.train_artDict)
        all_artifactDict.update(self.test_artDict)
        all_artifactId = list(all_artifactDict.keys())
        test_artifactId = list(self.test_artDict.keys())
        pos_pair = []
        neg_pair = []
        for i in  test_artifactId:
            for j in all_artifactId:
                if i == j:
                    continue
                if [i,j] in self.test_linkingpairs or [j,i] in self.test_linkingpairs:
                    node1_emb = all_artifactDict[i]
                    node2_emb = all_artifactDict[j]
                    pos_pair.append([node1_emb, node2_emb])
                else:
                    node1_emb = all_artifactDict[i]
                    node2_emb = all_artifactDict[j]
                    neg_pair.append([node1_emb, node2_emb])
        pos_labels = [1] * len(pos_pair)
        neg_labels = [0] * len(neg_pair)
        all_emb = pos_pair + neg_pair
        all_labels = pos_labels + neg_labels
        return all_emb, all_labels

class generateNode_Graph_Pair:
    def __init__(self, train_artifacts: list, test_artifacts: list, train_graphs_emb: list, Trian_GT:np.ndarray, Test_GT:np.ndarray, repoName: str):
        self.repoName = repoName
        self.utils = Utils(repoName)
        self.train_artDict = self.getArtifactDict(train_artifacts)
        self.test_artDict = self.getArtifactDict(test_artifacts)
        self.train_graphs_emb = train_graphs_emb
        self.train_linkingpairs = Trian_GT
        self.test_linkingpairs = Test_GT

    def getArtifactDict(self,artifacts: list):
        artDict = {}
        for art in artifacts:
            artId = art.getArtifactId()
            artDict[artId] = art
        return artDict

    def generateNodePairs(self, ratio=1.1):
        positive_pairs = self.train_linkingpairs
        negative_pairs = []
        artifacts = list(self.train_artDict.keys())
        k = int(ratio * len(positive_pairs))
        ran1 = np.random.choice(artifacts, size=k, replace=True)
        ran2 = np.random.choice(len(self.train_graphs_emb), size=k, replace=True)
        for i in range(len(ran1)):
            source = ran1[i]
            target = ran2[i]
            if [source, target] in positive_pairs or [target, source] in positive_pairs:
                continue
            negative_pairs.append([source, target])
        return positive_pairs, negative_pairs

    def getNodeFt(self, node_index, artifactDict):
        node_emb = artifactDict[node_index].getArtifactFeature("ft")
        node_input_ids = node_emb['input_ids']
        node_token_type_ids = node_emb['token_type_ids']
        node_attention_mask = node_emb['attention_mask']
        return node_input_ids,node_attention_mask, node_token_type_ids

    def getFeatureFromNodePair(self, node1_index,node2_index):
        node1_ft, node1_mask, node_token_type_ids = self.getNodeFt(node1_index, self.train_artDict)
        node2_ft = list(map(int, self.train_graphs_emb[node2_index].tolist()))
        node_ft = node1_ft + node2_ft
        node_mask = node1_mask + [1] * len(node2_ft)
        node_token_type_ids = node_token_type_ids + [1] * len(node_mask)
        fea = {"input_ids": node_ft, "attention_mask": node_mask, "token_type_ids": node_token_type_ids}
        return fea

    def getFeatureFromNodeGraph(self, node1_index, fullGraphs, j):
        node2_emb = list(map(int, fullGraphs[j]))
        node1_ft, node1_mask, node_token_type_ids = self.getNodeFt(node1_index, self.test_artDict)
        node_mask = node1_mask + [1] * len(node2_emb)
        node_ft = node1_ft + node2_emb
        node_token_type_ids = node_token_type_ids + [1] * len(node_mask)
        fea = {"input_ids": node_ft, "attention_mask": node_mask, "token_type_ids": node_token_type_ids}
        return fea

    def generateTrainingNodePairs(self):
        # 1 生成训练集
        positive_pairs, negative_pairs = self.generateNodePairs()
        positive_pairs = positive_pairs
        negative_pairs = negative_pairs
        # 2. 生成标签
        pos_labels = [1] * len(positive_pairs)
        neg_labels = [0] * len(negative_pairs)
        pos_emb = []
        # 3. 生成数据集
        for pos_index in range(len(positive_pairs)):
            node1_index = positive_pairs[pos_index][0]
            node2_index = positive_pairs[pos_index][1]
            fea = self.getFeatureFromNodePair(node1_index, node2_index)
            pos_emb.append(fea)
        neg_emb = []
        for neg_index in range(len(negative_pairs)):
            node1_index = negative_pairs[neg_index][0]
            node2_index = negative_pairs[neg_index][1]
            fea = self.getFeatureFromNodePair(node1_index, node2_index)
            neg_emb.append(fea)

        all_emb = pos_emb + neg_emb
        all_labels = pos_labels + neg_labels
        return all_emb, all_labels

    def generateTestingNodePairs(self, fullGraphs: list):
        full_graph_ids = list(range(len(fullGraphs)))
        test_artifactId = list(self.test_artDict.keys())[:50]
        pos_pair = []
        neg_pair = []
        for k, i in enumerate(test_artifactId):
            if k%100 == 0:
                print(k)
            for j in full_graph_ids:
                if [i,j] in self.test_linkingpairs or [j,i] in self.test_linkingpairs:
                    node1_index = i
                    fea = self.getFeatureFromNodeGraph(node1_index, fullGraphs, j)
                    pos_pair.append(fea)

                else:
                    node1_index = i
                    fea = self.getFeatureFromNodeGraph(node1_index, fullGraphs, j)
                    neg_pair.append(fea)

        pos_labels = [1] * len(pos_pair)
        neg_labels = [0] * len(neg_pair)
        all_emb = pos_pair + neg_pair
        all_labels = pos_labels + neg_labels
        return all_emb, all_labels