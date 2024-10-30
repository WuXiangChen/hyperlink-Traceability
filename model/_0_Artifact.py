# 本节的主要目的是 为每一个项目的节点注册其对应的特征

'''
    导包区
'''
from typing import List, Dict, Any
import networkx as nx
import numpy as np
from transformers import AutoTokenizer
from model.tokenizationText import encode_NL_PL
from utils import Utils

class Artifacts:
    def __init__(self, NL_Model_Path):
        self.artifacts = None
        self.artifact_dict = None
        self.tokenizer_NL = AutoTokenizer.from_pretrained(NL_Model_Path)

    def registerArtifacts(self, artifact_list: List[Dict[str, Dict]]):
        self.artifact_dict = {int(key):value for con in artifact_list for key, value in con.items()}
        self.artifacts = set(self.artifact_dict.keys())
        # 为每个节点注册向量特征
        for artifact_id in self.artifacts:
            artifact = self.artifact_dict[artifact_id]
            if "desc" not in artifact.keys() or artifact["desc"]==None:
                desc = ""
            else:
                desc = artifact["desc"]
            if "title" not in artifact.keys() or artifact["title"]==None:
                title = ""
            else:
                title = artifact["title"]
            # NL = title + " " + desc
            NL = title
            embed = encode_NL_PL(self.tokenizer_NL, NL)
            self.setArtifactFeature(artifact_id, "ft", embed)

    def getArtifactFeature(self, artifact_id: str, feature:str = "ft") -> Any:
        return self.artifact_dict[artifact_id][feature]

    def setArtifactFeature(self, artifact_id: int, feature: str, value: Any):
        self.artifact_dict[artifact_id][feature] = value

