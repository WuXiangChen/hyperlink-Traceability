# 本节的主要目的是动态的注册所有的约束设计

'''
    导包区
'''
import pandas as pd

from model.resource._0_Artifact import Artifact
from model.resource._0_TimeLine import TimeLine_Artifact
from utils import Utils


class restrictsRegister:
    def __init__(self, repoName: str):
        self.restricts = {}
        self.utils = Utils(repoName=repoName)

    def register(self, restrict):
        restrictName = restrict.__class__.__name__
        self.restricts[restrictName] = restrict.performFilter

    def lsAllRestricts(self):
        return self.restricts.keys()

    def getRestrict(self, restrictName):
        return self.restricts[restrictName]

    def getAllRestricts(self):
        return self.restricts


class timelineRestrict:
    def __init__(self, repoName: str, TL: TimeLine_Artifact):
        self.repoName = repoName
        self.utils = Utils(repoName=repoName)
        self.dfTL = TL.getTimeline()

    def performFilter(self, art: Artifact, type="train"):
        createdAt = pd.to_datetime(art.getArtifactFeature("createdAt"))
        two_years_constraint = createdAt - pd.DateOffset(years=2)
        filteredDF = self.dfTL[
            (self.dfTL["createdAt"] < createdAt) &
            (self.dfTL["last_event_time"] > two_years_constraint)
            ]
        if type == "train":
            filteredDF = filteredDF[filteredDF["type"] == type]
        filteredArts = filteredDF[["artifactID","art"]]
        filteredArts = filteredArts.drop_duplicates(subset=['artifactID'])
        dfArts = filteredArts.set_index('artifactID')
        return dfArts

class fileNameRestrict:
    def __init__(self, repoName: str, TL: TimeLine_Artifact):
        self.repoName = repoName
        self.utils = Utils(repoName=repoName)
        self.dfTL = TL.getTimeline()

    def performFilter(self, art: Artifact, type="train"):
        artType = art.getArtifactType()
        if artType == "Issue":
            return None
        fileName = set(art.getArtifactFeature("fileName"))
        filteredDF = self.dfTL[self.dfTL["fileName"].apply(lambda x: bool(set(x).intersection(fileName)))]
        if type == "train":
            filteredDF = filteredDF[filteredDF["type"] == type]
        filteredArts = filteredDF[["artifactID", "art"]]
        filteredArts = filteredArts.drop_duplicates(subset=['artifactID'])
        dfArts = filteredArts.set_index('artifactID')
        return dfArts
