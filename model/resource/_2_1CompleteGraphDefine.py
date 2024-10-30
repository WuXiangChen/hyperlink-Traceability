# 在这里我定义一个标准的图模型，这个图模型不必是一张完全图，它是连通的，但不必是完全的
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

class CompleteGraph:

    def __init__(self, row_count=2, col_count=20, feature_dim=(300,), pos=None):
        self.row_count = row_count
        self.col_count = col_count
        self.feature_dim = feature_dim
        self.standard_graph = self.create_standard_graph()
        self.pos = pos
        self.used_nodes = set()  # 用于存储已经使用的节点

    # 子图映射, 获取映射过程的标签列表
    def get_mapping_list(self, subgraph: nx.Graph) -> list:
        """将子图映射到标准图上"""
        # 分成几步来完成，首先从第一排以均匀采样的方式选取一组相邻节点，然后
        ## 子图的映射锚点就已经确定了
        num_samples = math.ceil(subgraph.number_of_nodes()/self.row_count)
        anchor_nodes = []

        # 保证已用节点与锚点节点的交集为空
        while anchor_nodes==[] or set(anchor_nodes).intersection(self.used_nodes)!=set():
            anchor_nodes = self.sample_continuous_nodes_from_first_row(num_samples)

        self.update_used_nodes(set(anchor_nodes))

        subNodes = list(subgraph.nodes())
        len_anchor_nodes = len(anchor_nodes)
        replaced_nodePairs = []
        for i in range(math.ceil(len(subNodes)/len_anchor_nodes)):
            for j, anchor_index in enumerate(anchor_nodes):
                index = i*len_anchor_nodes + j
                anchor_index = anchor_index + i*self.col_count
                if i*num_samples + j >= subgraph.number_of_nodes():
                    break
                replaced_nodePairs.append((subNodes[index], anchor_index))
        return replaced_nodePairs

    # 绘制图
    def draw_graph(self, csg, pos):
        """绘制给定的 NetworkX 图"""
        plt.figure(figsize=(20, 5))
        nx.draw(csg, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=12, font_color='black')
        plt.title("Standard Graph Structure")
        plt.show()

    def sample_continuous_nodes_from_first_row(self, num_samples)->list:
        """从标准图的第一排均匀采样给定数量的连续节点"""
        if num_samples > self.col_count:
            raise ValueError("采样数量不能大于列数")
        # 随机选择起始位置
        start_index = random.randint(0, self.col_count - num_samples)
        sampled_nodes = [col for col in range(start_index, start_index + num_samples)]
        return sampled_nodes

    def create_standard_graph(self):
        """创建一个标准图结构，分成多排，每排的节点两两相连，并与对面节点相连"""
        G = nx.Graph()
        row_count = self.row_count
        col_count = self.col_count
        feature_dim = self.feature_dim

        # 添加节点及其位置
        # 添加节点及其位置
        pos = {}  # 用于存储节点位置
        for i in range(col_count * row_count):
            random_feature = torch.empty(feature_dim)  # 创建一个空张量
            normalized_feature = nn.init.normal_(random_feature)  # 使用正态分布进行初始化
            G.add_node(i, ft=normalized_feature)

            # 计算行和列
            row = i // col_count
            col = i % col_count

            # 设置节点位置
            pos[i] = (col, -row)  # x 坐标为列，y 坐标为行的负值

        self.pos = pos  # 存储位置

        # 添加每排的节点之间的边
        for row in range(row_count):
            for col in range(col_count - 1):
                G.add_edge(row * col_count + col, row * col_count + col + 1)  # 连接同一排的节点

        # 添加对面节点之间的边
        for row in range(row_count - 1):
            for col in range(col_count):
                G.add_edge(row * col_count + col, (row + 1) * col_count + col)  # 连接上下排同一列的节点
                if col > 0:
                    G.add_edge(row * col_count + col, (row + 1) * col_count + col - 1)  # 连接左侧节点
                if col < col_count - 1:
                    G.add_edge(row * col_count + col, (row + 1) * col_count + col + 1)  # 连接右侧节点

        return G

    def change_node_feature(self, mapping_list: list, feature_com: torch.Tensor):
        """将子图的特征替换到标准图上"""
        for i, mapG in enumerate(mapping_list):
            for node_pair in mapG:
                comNode = node_pair[1]
                self.standard_graph.nodes[comNode]['ft'] = feature_com[i]

    def initialized_used_nodes(self):
        self.used_nodes = set()

    def update_used_nodes(self, nodes: set):
        self.used_nodes = self.used_nodes.union(nodes)

# 测试
# if __name__ == "__main__":
#     cg = CompleteGraph()
#     edges = cg.standard_graph.edges()
#     cg.draw_graph(cg.standard_graph, cg.pos)