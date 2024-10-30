# 本节的主要目的是将已经Labeled Artifact Graph形成表示，以便后续的预选择过程：


'''
    导包区
'''
import concurrent
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import torch
from typing import Dict
from .models.GraphSAGEModel import GraphSAGEModel
from torch_geometric.utils import from_networkx


class ArtifactGraphs(nx.Graph):
    def __init__(self, graph_list: Dict[int,nx.Graph], embedding_size: int = 300):
        super(ArtifactGraphs, self).__init__()
        self.graph_list = graph_list
        # 在初始化的时候，加入所有的节点信息
        self.embsize = embedding_size
        self.graph_embedding = torch.empty((len(graph_list), embedding_size), requires_grad=False)
        self.GraphSAGEModel = GraphSAGEModel(in_channels=embedding_size, hidden_channels= 512, out_channels=embedding_size)
        self.GraphSAGEModel.to(torch.device("cuda:0"))

    def asyngenerate_graphs_embedding(self, graph_list: Dict[int,nx.Graph], max_workers=1):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.generate_graph_embedding_from_GraphSAGE, graph): i for i, graph in graph_list.items()}
            for future in concurrent.futures.as_completed(futures):
                graph_embedding = future.result()
                i = futures[future]
                self.insert_graph_embedding(graph_embedding, i)
                if i % 100 == 0:
                    print(i)

    # 生成图的embedding
    def generate_graph_embedding_from_randomWalk(self, graph: nx.Graph):
        if graph.number_of_nodes()<2:
            graph_embedding = graph.nodes[0]['ft']
        else:
             graph_embedding = self.deep_walk.get_walks(graph)
        return graph_embedding

        # 生成图的embedding
    def generate_graph_embedding_from_GraphSAGE(self, graph: nx.Graph):
        data = from_networkx(graph)
        ls_ft = []
        for ft in data.ft:
            input_ids = ft["input_ids"]
            ls_ft.append(input_ids)
        ls_ft = torch.tensor(ls_ft, dtype=torch.float32)
        graph_embedding = ls_ft.mean(dim=0)
        # data.x = ls_ft
        # self.GraphSAGEModel.train_(data)
        # graph_embedding = self.GraphSAGEModel.get_graph_embedding(data)
        #print(graph.number_of_nodes())

        return graph_embedding

    # 在指定位置插入图的embedding
    def insert_graph_embedding(self,graph_embedding, index: int = -1):
        self.graph_embedding[index] = graph_embedding

    # 获取指定位置的图的embedding
    def get_graph_embedding(self, index: int):
        return self.graph_embedding[index]

    def get_graph_list(self):
        return self.graph_list

    # def get_graph_embedding_list(self):
    #     return self.graph_embedding
    #
    # # 增加
    # def add_graph(self, graph: nx.Graph):
    #     self.graph_list.append(graph)
    #
    # # 获取
    # def get_graph(self, index: int) -> nx.Graph:
    #     return self.graph_list[index]
    #
    # # 删除
    # def remove_graph(self, index: int):
    #     self.graph_list.pop(index)
    #
    # # 获取所有的节点
    # def get_all_nodes(self) -> List[str]:
    #     nodes = []
    #     for graph in self.graph_list:
    #         nodes += list(graph.nodes)
    #     return nodes
    #
    # # 获取所有的边
    # def get_all_edges(self) -> List[str]:
    #     edges = []
    #     for graph in self.graph_list:
    #         edges += list(graph.edges)
    #     return edges
    #
    # # 获取指定节点的邻居节点
    # def get_neighbors(self, node: str) -> List[str]:
    #     neighbors = []
    #     for graph in self.graph_list:
    #         if graph.has_node(node):
    #             neighbors += list(graph.neighbors(node))
    #     return neighbors
    #
    # # 获取指定节点的度
    # def get_degree(self, node: str) -> int:
    #     degree = 0
    #     for graph in self.graph_list:
    #         if graph.has_node(node):
    #             degree += graph.degree(node)
    #     return degree
    #
    # # 获取指定节点的度中心性
    # def get_degree_centrality(self, node: str) -> float:
    #     degree_centrality = 0
    #     for graph in self.graph_list:
    #         if graph.has_node(node):
    #             degree_centrality += nx.degree_centrality(graph)[node]
    #     return degree_centrality