# 本节的主要目的是通过给定的连接集合，构建一个groundtruth图，用于后续的测试验证过程

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
    def __init__(self, repoDataJson):
        self.linkingpairs = self.getLinkingPairs(repoDataJson)
        self.G = nx.Graph()

    def getLinkingPairs(self, repoDataJson):
        pair_wise = []
        for link in repoDataJson["links"]:
            sou = link["source"]
            tar = link["target"]
            pair_wise.append([int(sou), int(tar)])
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

    def plot(self):
        """绘制给定的 NetworkX 图"""
        plt.figure(figsize=(20, 6))# 设置节点位置
        nx.draw(self.G, with_labels=True, node_color='lightblue', node_size=700, font_size=6, font_color='black')
        plt.title("Standard Graph Structure")
        plt.show()
