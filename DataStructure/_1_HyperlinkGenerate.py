# 本节的主要目的是生成用于训练和测试的所有hyperlink，以及对应的label
# 生成的数据格式为：(source, target, label)

'''
    导包区
'''
import math
import random
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm

from ._0_DataStructure import ArtifactHyperLink

class generateHyperLinkDataset:
    def __init__(self, artifactIdList, linkingpairs_set, utils=None):
        self.artifactIdList = set(artifactIdList)
        self.positive_set = linkingpairs_set
        self.probabilities = self.default_probabilities()
        self.utils = utils
        # self.utils.visualizeGraph(26)

    def generateHyperLink(self, ratio, artifact_index,  vali_ratio=0.1, test_ratio=0.2):
        # 这里生成离散的正样本
        partition_pos_set = []
        for pos_set in self.positive_set:
            if len(pos_set) < 4:
                partition_pos_set.append(pos_set)
            else:
                # 随机生成若干个不相交的子集，子集的数量皆大于2
                max_num = math.floor(len(pos_set) / 2)
                nums = random.randint(2, max_num, seed=43)
                for i in range(nums):
                    left_unalloc = 2 * (nums - i - 1)
                    alloc_num = len(pos_set) - left_unalloc
                    if alloc_num < 2:
                        break
                    else:
                        set_ = random.sample(pos_set, alloc_num, seed=43)
                        pos_set = list(set(pos_set) - set(set_))
                        partition_pos_set.append(set_)
        # 这样可以衍生出一系列不完整的正样本集合
        positive_set = partition_pos_set
        full_pos_set = [pos_set for pos_set in positive_set if len(pos_set) > 4]
        negative_set, negative_adjacents = self.generate_negative_samples(positive_set, num_samples=5)
        pos_artiHyper = ArtifactHyperLink(self.artifactIdList, positive_set, artifact_index)
        neg_artiHyper = ArtifactHyperLink(self.artifactIdList, negative_set, artifact_index)
        full_pos_artiHyper = ArtifactHyperLink(self.artifactIdList, full_pos_set, artifact_index)

        posHyperlink = pos_artiHyper.create_incidence_matrix()
        negHyperlink = neg_artiHyper.create_incidence_matrix()
        full_posHyperlink = full_pos_artiHyper.create_incidence_matrix()

        return posHyperlink, negHyperlink, negative_adjacents, full_posHyperlink

    def default_probabilities(self):
        probabilities = {
            2: 0.6,
            3: 0.1,
        }
        for size in range(4, 101):
            probabilities[size] = probabilities[size - 1] * 0.5
        total_prob = sum(probabilities.values())
        normalized_probs = {k: v / total_prob for k, v in probabilities.items()}
        return normalized_probs

    def get_sample_size(self):
        # 根据概率选择样本大小
        sample_sizes = list(self.probabilities.keys())
        weights = list(self.probabilities.values())
        return random.choices(sample_sizes, weights=weights, seed=43)[0]

    def generate_negative_samples(self, positive_set, num_samples=5, num_folds=5):
        negative_set = []
        neg_graph = self.initialize_negative_graph()
        # kf = KFold(n_splits=num_folds, shuffle=True, random_state=43)
        for pos_set in positive_set:
            flag = True
            num_samples_ = num_samples
            while flag:
                sample_size = math.ceil(len(pos_set)/2)
                former_r = pos_set[:sample_size]
                artifactIdList = self.artifactIdList - set(former_r)
                indices = random.sample(artifactIdList, k=sample_size, seed=43)
                sample = former_r + indices
                if sample not in positive_set and sample not in negative_set:
                    negative_set.append(sample)
                    neg_graph = self.generate_connected_graph_with_given_nodes(sample, neg_graph)
                    if num_samples_ <= 0:
                        flag = False
                    num_samples_ -= 1
        negative_adjacents = nx.adjacency_matrix(neg_graph).toarray()
        return negative_set,negative_adjacents

    def generate_specific_negative_samples(self, positive_set, num_samples=5, selected_nodes=None):
        negative_set = []
        neg_graph = self.initialize_negative_graph()
        # kf = KFold(n_splits=num_folds, shuffle=True, random_state=43)
        for pos_set in positive_set:
            flag = True
            num_samples_ = num_samples
            while flag:
                sample_size = math.ceil(len(pos_set)/2)
                former_r = pos_set[:sample_size]
                artifactIdList = selected_nodes - set(former_r)
                indices = random.sample(artifactIdList, k=sample_size)
                sample = former_r + indices
                if sample not in positive_set and sample not in negative_set:
                    negative_set.append(sample)
                    neg_graph = self.generate_connected_graph_with_given_nodes(sample, neg_graph)
                    if num_samples_ <= 0:
                        flag = False
                    num_samples_ -= 1
        negative_adjacents = nx.adjacency_matrix(neg_graph).toarray()
        return negative_set,negative_adjacents

    # 初始化负样本图的所有节点
    def initialize_negative_graph(self):
        neg_graph = nx.Graph()
        for node in self.artifactIdList:
            neg_graph.add_node(node)
        return neg_graph

    def generate_connected_graph_with_given_nodes(self, nodes, graph):
        graph.add_nodes_from(nodes)
        # 随机排列节点以生成一个生成树
        shuffled_nodes = nodes[:]
        random.shuffle(shuffled_nodes)
        for i in range(len(shuffled_nodes) - 1):
            node1 = shuffled_nodes[i]
            node2 = shuffled_nodes[i + 1]
        return graph