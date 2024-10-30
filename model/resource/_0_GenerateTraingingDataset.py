# 本节的主要目的是生成合适的训练集与测试集，并动态的支持所有可能的约束设计

'''
    导包区
'''
from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import tqdm

from model.resource._0_Artifact import Artifact
from utils import Utils
from model.resource._0_TimeLine import TimeLine_Artifact
from model.resource._0_RestrictionConstructed import restrictsRegister
np.random.seed(42)

class generateNode_Graph_Pair:
    def __init__(self, artifacts: list, fullArtifacts: list,  TL: TimeLine_Artifact, restrictMap: restrictsRegister,
                 repoName: str, type: str = "train"):
        self.TL = TL
        self.type = type
        self.restrictMap = restrictMap
        self.repoName = repoName
        self.utils = Utils(repoName)
        self.artDict = self.getArtifactDict(artifacts)
        self.fullArtifacts = self.getArtifactDict(fullArtifacts)
        self.linkingpairs = self.getLinkingPairs()
        self.temp_G = nx.Graph()

    def getLinkingPairs(self):
        filePath = "dataset/" + self.repoName + "/processed/test.json"
        testlinkingpairs = self.utils.load_json(filePath)

        filePath = "dataset/" + self.repoName + "/processed/train.json"
        trainlinkingpairs = self.utils.load_json(filePath)

        # 使用集合来存储无序的对
        if self.type.__eq__("train"):
            pairs = trainlinkingpairs
        else:
            pairs = list(testlinkingpairs)+list(trainlinkingpairs)
        # 创建一个集合来存储无序的对
        linkingpairs_set = {frozenset(pair) for pair in pairs}
        return linkingpairs_set

    def changeState(self, type, artifacts:list):
        self.type = type
        self.artDict = self.getArtifactDict(artifacts)
        self.linkingpairs = self.getLinkingPairs()

    def getArtifactDict(self, artifacts: list):
        artDict = {}
        for art in artifacts:
            artId = art.getArtifactId()
            artDict[artId] = art
        # 按照artId进行排序
        sorted_artDict = dict(sorted(artDict.items()))
        return sorted_artDict

    def getNodeFeature(self, node_id):
        # 增加 lenCode, filepath,  participants, assignees, event_timeline, event_state, lenCommits 的特征
        features = {
            'ft': self.fullArtifacts[node_id].getArtifactFeature("ft"),
            'createdAt': self.fullArtifacts[node_id].getArtifactFeature("createdAt"),
            'lenCode': self.fullArtifacts[node_id].getArtifactFeature("lenCode"),
            'filepath': self.fullArtifacts[node_id].getArtifactFeature("filepath"),
            'participants': self.fullArtifacts[node_id].getArtifactFeature("participants"),
            'assignees': self.fullArtifacts[node_id].getArtifactFeature("assignees"),
            'event_timeline': self.fullArtifacts[node_id].getArtifactFeature("event_timeline"),
            'event_state': self.fullArtifacts[node_id].getArtifactFeature("event_state"),
        }
        return features

    def syncGetDataPair(self):
        DataPair = defaultdict(list)
        allRes = self.restrictMap.getAllRestricts()
        last_artId = set()
        keys = list(self.artDict.keys())[:3000]
        for k, artId in tqdm.tqdm(enumerate(keys), total=len(keys)):
            art = self.artDict[artId]
            filteredSet = {}
            for resName in allRes.keys():
                dfArts = allRes[resName](art, self.type)
                if dfArts is not None:
                    filteredSet[resName] = dfArts
            dfIntersect = pd.concat(filteredSet.values(), axis=1, join='inner')
            artsDict = set(dfIntersect.to_dict(orient='index').keys())
            add_artIds = []
            if len(artsDict) > 0:
                add_artIds = list(artsDict.difference(last_artId))
                last_artId = last_artId.union(artsDict)
            temp_G = self.generateGraphAndNodeCollection(add_artIds)
            temp_G_components = list(nx.connected_components(temp_G))
            pos_pairs = self.generatePosPairs(temp_G_components, art)
            if self.type == "train":
                neg_pairs = self.generateNegPairs(temp_G_components,  pos_pairs)
            else:
                neg_pairs = self.generateNegPairs(temp_G_components,  pos_pairs, 0)
            pos_G = {i: temp_G.subgraph(component).copy() for i, component in enumerate(pos_pairs)}
            neg_G = {i: temp_G.subgraph(component).copy() for i, component in enumerate(neg_pairs)}
            DataPair[artId].append(artId)
            DataPair[artId].append(pos_G)
            DataPair[artId].append(neg_G)
        return DataPair

    def generateGraphAndNodeCollection(self, artIDs: List[str]):
        existed_artIds = list(self.temp_G.nodes)
        for artId in artIDs:
            self.temp_G.add_node(artId, **self.getNodeFeature(artId))
        for e_artId in existed_artIds:
            for artId in artIDs:
                if Utils.is_linked(e_artId, artId, self.linkingpairs):
                    self.temp_G.add_edge(e_artId, artId)
        if len(artIDs) < 2:
            return self.temp_G
        for artId1 in artIDs:
            for artId2 in artIDs:
                if artId1 == artId2:
                    continue
                if Utils.is_linked(artId1, artId2, self.linkingpairs):
                    self.temp_G.add_edge(artId1, artId2)
        return self.temp_G

    def generatePosPairs(self, temp_G_components: List[str], art:Artifact):
        pos_pairs = []
        target = art.getArtifactId()
        for component in temp_G_components:
            for source in component:
                if Utils.is_linked(source, target, self.linkingpairs):
                    pos_pairs.append(component)
                    break
        return pos_pairs

    def generateNegPairs(self, temp_G_components: List[str], pos_pairs: List, ratio=1.1):
        pos_len = len(pos_pairs)
        neg_G_components = []
        if pos_len == 0:
            neg_G_components = temp_G_components
        else:
            for component in temp_G_components:
                if component not in pos_pairs:
                    neg_G_components.append(component)
        if ratio != 0:
            if len(neg_G_components) < int(pos_len*ratio)+1:
                return neg_G_components
            # 计算权重
            weights = [2 if len(item) > 1 else 1 for item in neg_G_components]
            # 归一化权重
            weights = np.array(weights) / np.sum(weights)
            neg_Trainging_pairs = np.random.choice(neg_G_components, size=int(pos_len * ratio)+1, replace=False, p=weights).tolist()
            return neg_Trainging_pairs
        else:
            neg_Testing_pairs = neg_G_components
            return neg_Testing_pairs

