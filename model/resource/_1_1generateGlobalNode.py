import networkx as nx
import random
import numpy as np
from typing import List

import torch
from torch import dtype
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec

class DeepWalk:

    def __init__(self, window_size: int, embedding_size: int, walk_length: int, walks_per_node: int):
        """
        :param window_size: window size for the Word2Vec model
        :param embedding_size: size of the final embedding
        :param walk_length: length of the walk
        :param walks_per_node: number of walks per node
        """
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_length = walk_length
        self.walk_per_node = walks_per_node

    def random_walk(self, g: nx.Graph, start: str, use_probabilities: bool = False, walk_length: int =5) -> List[str]:
        """
        Generate a random walk starting on start
        :param g: Graph
        :param start: starting node for the random walk
        :param use_probabilities: if True take into account the weights assigned to each edge to select the next candidate
        :return:
        """
        walk_feature = [g.nodes[start]["ft"]]
        walk = [start]
        for i in range(walk_length):
            neighbours = g.neighbors(walk[i])
            neighs = list(neighbours)
            if len(neighs) == 0:
                break
            if use_probabilities:
                probabilities = [g.get_edge_data(walk[i], neig)["ft"] for neig in neighs]
                sum_probabilities = sum(probabilities)
                probabilities = list(map(lambda t: t / sum_probabilities, probabilities))
                p = np.random.choice(neighs, p=probabilities)
                ft = g[p]["ft"]
            else:
                p = random.choice(neighs)
                ft = g.nodes[p]["ft"]
            walk.append(p)
            walk_feature.append(ft)
        return walk_feature

    def get_walks(self, g: nx.Graph, use_probabilities: bool = False) -> List[List[str]]:
        """
        Generate all the random walks
        :param g: Graph
        :param use_probabilities:
        :return:
        """
        random_walks = []
        walk_per_node = min(self.walk_per_node, len(g.nodes))
        walk_length = min(self.walk_length, len(g.nodes))
        for _ in range(walk_per_node):
            random_nodes = list(g.nodes)
            random.shuffle(random_nodes)
            for node in random_nodes:
                random_walks.extend(self.random_walk(g=g, start=node, use_probabilities=use_probabilities,
                                                     walk_length=walk_length))

       # random_walks转成tensor
        tensor_ = torch.empty(len(random_walks), len(random_walks[0]))
        for i in range(len(random_walks)):
            tensor_[i] = torch.from_numpy(random_walks[i])

        tensor_random_walks_mean = tensor_.mean(dim=0)
        return tensor_random_walks_mean

    def compute_embeddings(self, walks: List[List[str]]):
        """
        Compute the node embeddings for the generated walks
        :param walks: List of walks
        :return:
        """
        model = Word2Vec(sentences=walks, window=self.window_size, vector_size=self.embedding_size)
        # model.train(walks, total_examples=len(walks), epochs=10)
        # 第一个问题 这里的训练结构是怎么回事？
        # 第二个问题 这里的vectors是怎么回事
        wv = model.wv
        toVector = wv.vectors.mean(axis=0)
        return toVector