
'''
    导包区
'''
import itertools
import numpy as np
import networkx as nx
from typing import List, Dict, Any
from matplotlib import pyplot as plt

from utils import Utils

class GroundTruthGraph():
    def __init__(self, repoDataJson, trusted_project=True):
        self.linkingpairs = self.getLinkingPairs(repoDataJson, trusted_project)
        self.G = nx.Graph()
        self.build_groundtruth_graph()

    def getLinkingPairs(self, repoDataJson, trusted_projec=True):
        pair_wise = []
        if not trusted_projec:
            for link in repoDataJson["links"]:
                sou = link["source"]
                tar = link["target"]
                pair_wise.append([int(sou), int(tar)])
        else:
            for link in repoDataJson:
                for ids in link.items():
                    sou, tar = ids[0], ids[1]
                    pair_wise.append([int(sou), int(tar)])
                    break
        return pair_wise

    def build_groundtruth_graph(self):
        for pair in self.linkingpairs:
            self.G.add_edge(pair[0], pair[1])
        return self.G

    def getComponents(self):
        return list(nx.connected_components(self.G))

    def isConnect(self, pair):
        return nx.has_path(self.G, pair[0], pair[1])

    def getEdges(self):
        return list(self.G.edges)

    def getNodes(self):
        return list(self.G.nodes)

    # 本节的主要目的是根据已经建立的groundTruth图，生成训练集合中的Pair2Pair的groundTruth
    def generateTrainingP2PDataset(self, train_hyperlink:Dict, testFlag=False):
        P2PDatasets = {}
        P2PLabels = []
        HPvalues = list(train_hyperlink.values())
        edges = self.getEdges()
        P2PDatasets, P2PLabels = Utils.P2P_Expanded(HPvalues, edges, addTwoComORNot=True, testFlag=testFlag)
        return P2PDatasets, P2PLabels

    def plot(self):
        """绘制给定的 NetworkX 图"""
        plt.figure(figsize=(20, 6))# 设置节点位置
        nx.draw(self.G, with_labels=True, node_color='lightblue', node_size=700, font_size=6, font_color='black')
        plt.title("Standard Graph Structure")
        plt.show()
