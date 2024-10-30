# 本节是用于定义描述Issue和PR的类，用于存储Issue和PR的信息
## 所有Issue和PR应当先注册成Artifact，然后再进行后续的处理

'''
    导包区
'''
import torch
from transformers import AutoTokenizer
from utils import Utils
from ._0_generateSTA import staFeature
from model.tokenizationText import encode_text_with_glove, encode_with_specific_tokenizer

class Artifact:
    def __init__(self, artifact_id: int, artifact_type: str, artifact_feature: dict, tokenizer_NL=None, tokenizer_PL=None):
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self.artifact_feature = {}
        self.generateTextualFeature(artifact_feature, tokenizer_NL, tokenizer_PL)

    def regestrateArtifactFeature(self, artifact_feature:dict):
        dictFeature = {}
        dictFeature["title"] = artifact_feature["title"]
        dictFeature["desc"] = artifact_feature["desc"]
        dictFeature["comment"] = artifact_feature["comment"]
        try:
            dictFeature["NL"] = " ".join([dictFeature["title"], dictFeature["desc"], dictFeature["comment"]])
        except Exception as e:
            print(f"regestrateArtifactFeature Error: {e}")

        dictFeature["PL"] = " ".join(artifact_feature["PL"])
        dictFeature["createdAt"] = artifact_feature["createdAt"]

        # 这里去除停用词
        for key in dictFeature.keys():
            if isinstance(dictFeature[key],str) and key != "createdAt":
                dictFeature[key] = Utils.remove_stopwords(dictFeature[key])
            elif isinstance(dictFeature[key], list) and key != "createdAt":
                dictFeature[key] = [Utils.remove_stopwords(item) for item in dictFeature[key]]
        return dictFeature

    def generateTextualFeature(self, ori_artifact_fea, tokenizerNL, tokenizer_PL):
        artifact_feature = self.regestrateArtifactFeature(ori_artifact_fea)
        PL = " ".join(artifact_feature["PL"])
        NL = artifact_feature["title"]
        ft = encode_with_specific_tokenizer(tokenizerNL, tokenizer_PL, NL, PL)
        del tokenizerNL, tokenizer_PL
        self.setFeature("ft", ft)
        self.setFeature("createdAt", artifact_feature["createdAt"])
        self.setFeature("title", artifact_feature["title"])
        self.setFeature("desc", artifact_feature["desc"])
        self.setFeature("lenCode", Utils.sumBeginWithPlus(artifact_feature["PL"]))
        self.setFeature("filepath", ori_artifact_fea['file_paths'])
        self.setFeature("participants", ori_artifact_fea['participants'])
        self.setFeature('assignees', ori_artifact_fea['assignees'])
        self.setFeature('event_timeline', ori_artifact_fea['event_timeline']["created_at"].tolist())
        self.setFeature('event_state', ori_artifact_fea['event_timeline']["event"].tolist())
        self.setFeature('lenCommits', len(ori_artifact_fea['commit_shas']))

    def getArtifactId(self):
        return int(self.artifact_id)

    def getArtifactType(self):
        return self.artifact_type

    def getArtifactFeature(self, feature_name: str):
        # 存在性检查
        if feature_name not in self.artifact_feature.keys():
            raise Exception(f"{self.artifact_id} Feature Name {feature_name} Not Existed!")
        return self.artifact_feature[feature_name]

    def setFeature(self, feature_name: str, feature_value):
        self.artifact_feature[feature_name] = feature_value

import concurrent.futures
from typing import List, Dict
from memory_profiler import profile
from transformers import AutoModel
class ArtifactRegister:
    def __init__(self, artifactContent: Dict[str, List[dict]], plINFOContent: Dict[str, List[dict]], dataType: str, repoName: str, StaDataRootPath:str, staClass:staFeature, device):
        self.artifactContent = artifactContent
        self.plINFOContent =plINFOContent
        self.dataType = dataType
        self.repoName = repoName
        self.utils = Utils(repoName)
        self.staClass = staClass
        NL_Model_Path = "model/models/BAAI_bge-m3_small/"
        PL_Model_Path = "model/models/codet5p-110m-embedding/"
        tokenizer_NL = AutoTokenizer.from_pretrained(NL_Model_Path)
        tokenizer_PL = AutoTokenizer.from_pretrained(PL_Model_Path)

        model = AutoModel.from_pretrained(NL_Model_Path, trust_remote_code=True)
        NL_embeddings = model.get_input_embeddings()

        model = AutoModel.from_pretrained(PL_Model_Path, trust_remote_code=True)
        PL_embeddings = model.get_input_embeddings()

        self.tokenizer_NL = (tokenizer_NL, NL_embeddings)
        self.tokenizer_PL = (tokenizer_PL, PL_embeddings)

        # StaDataRootPath 是统计数据的根目录，其中包括了Issue和PR的统计数据，我们现在主要使用其中的create_time和pr的fileName信息
        self.IssueCreatedAt = self.utils.load_json(StaDataRootPath+"/issueCreatedTime.json")
        self.PrCreatedAt = self.utils.load_json(StaDataRootPath+"/prCreatedTime.json")
        self.PRFileName = self.utils.load_json(StaDataRootPath+"/prFileAndCommit.json")

    def process_artifact(self, artifactId: str) -> Artifact or None:
        artContent = self.artifactContent[artifactId]
        artifactType = self.utils.judgePRORIssue(artifactId)
        if len(artContent) == 0 or artifactType == "None":
            return None
        title = artContent[0]['title']
        desc = artContent[0]["issueBody"] or ""
        comments = "\n".join(artContent[i]['comment'] for i in range(1, len(artContent)))
        artifactFeature = {"title": title, "desc": desc, "comment": comments}
        if artifactType == "PR":
            artifactFeature["createdAt"] = self.PrCreatedAt[artifactId]
            #artifactFeature["fileName"] = self.PRFileName[artifactId]["files"]
            if artifactId in self.plINFOContent.keys():
                Patch = self.plINFOContent[artifactId]
                artifactFeature["PL"] = Patch
            else:
                print(f"PR {artifactId} has no Patch Info.")
                artifactFeature["PL"] = []
        elif artifactType == "Issue":
            artifactFeature["createdAt"] = self.IssueCreatedAt[artifactId]
            artifactFeature["PL"] = []
            #artifactFeature["fileName"] = []
        artifactFeature.update(self.staClass.getSTAInfoByEventId(int(artifactId), artifactType))
        return Artifact(int(artifactId), artifactType, artifactFeature, self.tokenizer_NL, self.tokenizer_PL)

    def register_artifacts(self, max_workers, save_path) -> List[Artifact]:
        unlabeled_artifacts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers= max_workers) as executor:
            futures = {executor.submit(self.process_artifact, artifactId): artifactId for artifactId in self.artifactContent}
            i = 0
            for future in concurrent.futures.as_completed(futures):
                artifact = future.result()
                if artifact is not None:
                    unlabeled_artifacts.append(artifact)
                    if i % 100 == 0 and i!=0:
                        print(i)
                        self.utils.save_artifact_pickle(unlabeled_artifacts, save_path)
                        unlabeled_artifacts = []
                    i += 1
                else:
                    del artifact

        self.utils.save_artifact_pickle(unlabeled_artifacts, save_path)
        print(f"Artifact {self.dataType} has been registered.")
        return unlabeled_artifacts

# Example usage:
# registrar = ArtifactRegister(artifactContent, dataType, repoName)
# artifacts = registrar.register_artifacts()