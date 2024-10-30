# 本节主要是一个工具，用以将预定义的各种特征类型 集合在某一个具体的向量中，并将此向量作为模型的输入内容
## 按照LLM的设计，向量输入的结构应当是固定的，因此这里的输出形式应当是一个固定长度的向量

# 一个最简单的策略是只使用这里的ft特征，暂时不去考虑其他统计特征

'''
    导包区
'''
from typing import Dict

import torch
import networkx as nx
from tqdm import tqdm
from utils import Utils
from model.resource._0_Artifact import Artifact

class convertDataToArtAndLabel:
    def __init__(self, Data: dict, PL_Embed_name:str, full_artifacts: list, dimension: int = 512):
        # keys = list(Data.keys())[:1000]
        # self.Data = {key:Data[key] for key in keys}
        self.Data = Data
        self.dataAndLabel = {"art1":[],"art2":[], "labels": [], "art1_index":[], "art2_index":[]}
        self.full_artifacts = Utils.getArtifactDict(full_artifacts)
        self.dimension = dimension
        self.count_graph = 0

    def convert(self, type="test"):
        for key, value in tqdm(self.Data.items()):
            art:Artifact = self.full_artifacts[key]
            pos_graph = value[1]
            neg_graph = value[2]
            artFt = art.getArtifactFeature("ft")
            if len(pos_graph) != 0:
                self.eachGraphConvert(artFt, pos_graph, 1, art1_index=key)
            if len(neg_graph) != 0:
                self.eachGraphConvert(artFt, neg_graph, 0, len_limit=len(pos_graph),art1_index=key)

        return self.dataAndLabel

    def eachGraphConvert(self, artFt: torch.tensor, graphs: Dict[int, nx.Graph], label: int, len_limit:int = 128, art1_index:int = 0):
        input_ids = artFt.to(torch.int16)
        for index in list(graphs.keys()):
            graph = graphs[index]
            self.dataAndLabel["art1"].append(input_ids)
            self.dataAndLabel["art2"].append(graph)
            self.dataAndLabel["labels"].append(label)
            self.dataAndLabel["art1_index"].append(art1_index)
            self.count_graph += 1
            self.dataAndLabel["art2_index"].append(self.count_graph)