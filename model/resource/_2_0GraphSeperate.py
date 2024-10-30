# 本节的主要目的是进行图分解

import torch
from typing import List
import networkx as nx
from model.resource._0_Artifact import Artifact
class GraphSeperate(nx.Graph):
    def __init__(self, threshold: int=4):
        super(GraphSeperate).__init__()
        self.threshold = threshold

    def sort_graph_by_degree(self, G, ascending=False):
        # 计算每个节点的度
        degree_dict = dict(G.degree())
        # 按照度排序
        sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=not ascending)
        return sorted_nodes

    def decompose_graph(self, G: nx.Graph, art:Artifact)->List[nx.Graph]:
        """
        将图分解为子图
        :return:
        """
        # 生成子图
        subgraphs = []
        # sort_node = self.sort_graph_by_degree(G)
        G_ = G.copy()
        while len(G_.nodes) >= self.threshold-1:
            # 选择一个连通组件
            components = list(nx.connected_components(G_))
            component = components.pop()

            # 判断这个component的节点数是否小于阈值-1
            if len(component) < self.threshold - 1:
                # 如果小于阈值-1，则直接将这个连通组件加入到子图中
                subgraph = self.get_sub_graph(component, G, art)
                subgraphs.append(subgraph)
                G_.remove_nodes_from(component)
                continue

            # 选择一条边
            edge = list(G_.edges(component)).pop()
            nodes = edge[0], edge[1]
            subgraph = self.get_sub_graph(nodes, G, art)
            subgraphs.append(subgraph)

            # 从图G_中删除这条边的两端节点
            G_.remove_nodes_from(nodes)

        if len(G_.nodes) > 0:
            com_subGraph = self.get_sub_graph(G_.nodes, G, art)
            subgraphs.append(com_subGraph)

        return subgraphs

    def get_sub_graph(self, component: nx.connected_components, G: nx.Graph, art:Artifact):
        """
        获取映射列表
        :param subGraph:
        :return:
        """
        G_ = G.copy()
        new_G = nx.Graph()
        nodes = list(component)
        G_.remove_nodes_from(nodes)
        feature_dim = len(list(G.nodes(data=True))[0][1]['ft'])
        tensor_ = torch.empty(size=(len(G_.nodes), feature_dim), dtype=torch.float32)
        for i, node in enumerate(G_.nodes):
            tensor_[i] = G_.nodes[node]['ft']
        maxFeature = tensor_.max(dim=0).values

        new_G.add_node(-1, ft=maxFeature)
        new_G.add_node(art.getArtifactId(), ft=art.getArtifactFeature('ft'))

        for node in nodes:
            new_G.add_node(node, ft=G.nodes[node]['ft'])

        com_subGraph = self.create_complete_graph(new_G)

        return com_subGraph

    def create_complete_graph(self, G):
        # 获取原图的所有节点
        nodes = G.nodes()

        # 创建全连接无向图
        complete_graph = nx.complete_graph(nodes)

        # 将原有特征复制到完全图中
        for node in nodes:
            complete_graph.nodes[node]['ft'] = G.nodes[node]['ft']

        return complete_graph



