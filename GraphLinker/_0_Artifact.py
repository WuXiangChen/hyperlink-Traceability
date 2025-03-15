# 本节的主要目的是 为每一个项目的节点注册其对应的特征

'''
    导包区
'''
from typing import List, Dict, Any
import networkx as nx
import numpy as np
from transformers import AutoTokenizer
from utils import Utils

class Artifacts:
    def __init__(self,
                 NL_Model_Path,
                 repo_sta_commitAndFiles:Dict[str,int]= None,
                 repo_sta_artifactInfo:Dict[str,List]= None,
                 indexToArtId:Dict[int,int]=None):
        self.artifacts = None
        self.artifact_dict = None
        self.tokenizer_NL = AutoTokenizer.from_pretrained(NL_Model_Path)
        # 维护一个index与Id之间的对应关系
        self.indexToArtId = indexToArtId
        # 这是个字典，以Artifact Id为键，以commit和fileName为值
        self.repo_sta_commitAndFiles = repo_sta_commitAndFiles
        # 这是个字典，以Artifact Id为键，以统计结果为值
        self.repo_sta_artifactInfo = repo_sta_artifactInfo
        self.globalDict = {}
        self.hp_HiddenDList = []

    # 补充：这里还维护一个信息，将 allAuthors, allLabel, allFileName, allArtifactType等信息注入到每一个artifacts中
    def registerArtifacts(self, artifact_list: List[Dict[str, Dict]], repoName: str):
        if repoName in ['ansible','angular','elasticsearch','symfony', 'elastic', 'dotnet', 'saltstack']:
            self.artifact_dict = {int(key):value[0] for key, value in artifact_list.items()}
        else:
            self.artifact_dict = {int(key):value for con in artifact_list for key, value in con.items()}

        self.artifacts = set(self.artifact_dict.keys())
        allAuthors = set()
        allLabels = set()
        allFileNames = set()
        allArtifactTypes = {"pr", "issue", "created", "updated", "closed"}
        for artId in self.artifacts:
            if artId in self.repo_sta_artifactInfo.keys():
                staCon = self.repo_sta_artifactInfo[artId]["issue_data"]
                contributors = set()
                self.setArtifactFeature(artId, "open_time", staCon["created_at"])
                self.setArtifactFeature(artId, "updata_time", staCon["updated_at"])
                self.setArtifactFeature(artId, "close_time", staCon["closed_at"])
                author = staCon["user"]["login"]
                self.setArtifactFeature(artId, "author", author)
                contributors.add(author)
                if staCon["closed_by"]!=None:
                    closed_by = staCon["closed_by"]["login"]
                    contributors.add(author)
                else:
                    closed_by = None
                comments = self.repo_sta_artifactInfo[artId]["comments_data"]
                reviewer = []
                if isinstance(comments, list):
                    reviewer = [con['user']['login'] for con in comments]
                contributors.update(set(reviewer))
                self.setArtifactFeature(artId, "reviewer", reviewer)
                self.setArtifactFeature(artId, "closed_by", closed_by)
                self.setArtifactFeature(artId, "author_association", staCon["author_association"])
      
                labels = staCon["labels"]
                if len(labels)!=0:
                    labels = [con["name"] for con in labels]
                self.setArtifactFeature(artId, "labels", labels)

                is_pr = staCon.get("pull_request") is not None

                if is_pr:
                    if artId not in self.repo_sta_commitAndFiles.keys():
                        self.setArtifactFeature(artId, "commits", [])
                        self.setArtifactFeature(artId, "files", [])
                    else:
                        re = self.repo_sta_commitAndFiles[artId]
                        if "commits" in re.keys():
                            commit_ = re["commits"]
                            self.setArtifactFeature(artId, "commits", commit_)
                        else:
                            self.setArtifactFeature(artId, "commits", [])

                        if "files" in re.keys():
                            files_ = re["files"]
                            self.setArtifactFeature(artId, "files", files_)
                        else:
                            files_ = []
                            self.setArtifactFeature(artId, "files", files_)
                        allFileNames.update(files_)
                    self.setArtifactFeature(artId, "is_pr", 1)
                else:
                    self.setArtifactFeature(artId, "commits", [])
                    self.setArtifactFeature(artId, "files", [])
                    self.setArtifactFeature(artId, "is_pr", 0)
                
                allAuthors.update(contributors)
                allLabels.update(set(labels))

            else:       
                self.setArtifactFeature(artId, "open_time", None)
                self.setArtifactFeature(artId, "updata_time", None)
                self.setArtifactFeature(artId, "close_time", None)
                self.setArtifactFeature(artId, "author", None)
                self.setArtifactFeature(artId, "reviewer", None)
                self.setArtifactFeature(artId, "closed_by", None)
                self.setArtifactFeature(artId, "is_pr", None)
                self.setArtifactFeature(artId, "files", None)
                self.setArtifactFeature(artId, "commits", None)
                self.setArtifactFeature(artId, "labels", None)
                self.setArtifactFeature(artId, "author_association", None)

        self.globalDict["allAuthors"] = allAuthors
        self.globalDict["allLabels"] = allLabels
        self.globalDict["allFileNames"] = allFileNames
        self.globalDict["allArtifactTypes"] = allArtifactTypes
        self.hp_HiddenDList.extend([len(allAuthors), len(allLabels), len(allFileNames), len(allArtifactTypes)])


    ## 需要一个判断逻辑，来判断是否存在指定名称的特征？

    def getArtifactFeature(self, artifact_id: str, feature:str = "ft") -> Any:
        return self.artifact_dict[artifact_id][feature]

    def setArtifactFeature(self, artifact_id: int, feature: str, value: Any):
        self.artifact_dict[artifact_id][feature] = value

