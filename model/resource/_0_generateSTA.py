# 本节的主要目的是生成STA Feature，以一种依赖注入的方式进行合作，而不是直接调用

'''
    导包区
'''
from datetime import datetime
from typing import List, Dict

import pandas as pd

from utils import Utils

'''
    1. STA Feature的定义   
'''
class staFeature:
    def __init__(self, repoName:str):
        self.dfPREvent = None
        self.dfIssueEvent = None
        self.dfPRFileAndCommit = None
        self.repoName = repoName
        self.utils = Utils(repoName)
        self.initialization()
        self.time_line = []
        self.state_line = []
        self.filePaths = []
        self.participants = []

    def initialization(self):
        issueEvent = self.utils.load_json(self.utils.issueEventPath)
        prEvent = self.utils.load_json(self.utils.prEventPath)
        self.dfIssueEvent = pd.DataFrame(issueEvent)
        self.dfIssueEvent.drop_duplicates(subset=["id"], inplace=True)
        self.dfIssueEvent.reset_index(drop=True, inplace=True)
        self.dfPREvent = pd.DataFrame(prEvent)
        self.dfPREvent.drop_duplicates(subset=["id"], inplace=True)
        self.dfPREvent.reset_index(drop=True, inplace=True)
        prFileAndCommit = self.utils.load_json(self.utils.prFileAndCommitPath)
        self.dfPRFileAndCommit = pd.DataFrame(prFileAndCommit).T
        self.dfPRFileAndCommit['eventId'] = self.dfPRFileAndCommit.index.astype(int)
        self.dfPRFileAndCommit.reset_index(drop=True, inplace=True)
        print("STA Feature Initialization is success!")


    def getSTAInfoByEventId(self, eventId:int, type:str) -> (pd.DataFrame, pd.DataFrame):
        # type表示该event的类型，是issue还是pullrequest
        eventInfo, fileInfo = None, None
        if type == "PR":
            eventInfo, fileInfo = self.getPRSTAInfoByEventId(eventId)
            return self.postProcessPR(eventInfo, fileInfo)
        else:
            eventInfo = self.getIssueSTAInfoByEventId(eventId)
            return self.postProcessIssue(eventInfo, fileInfo)


    def getPRSTAInfoByEventId(self, eventId):
        # 通过eventId获取pullrequest的STA信息
        rangeEventInfo = self.dfPREvent[self.dfPREvent["eventId"] == eventId]
        rangeFileAndCommit = self.dfPRFileAndCommit[self.dfPRFileAndCommit["eventId"] == eventId]
        return rangeEventInfo, rangeFileAndCommit

    def getIssueSTAInfoByEventId(self, eventId):
        # 通过eventId获取issue的STA信息
        rangeEventInfo = self.dfIssueEvent[self.dfIssueEvent["eventId"] == eventId]
        return rangeEventInfo

    def reformatTime(self, date_string):
        date_format = "%Y-%m-%dT%H:%M:%SZ"
        try:
            date = datetime.strptime(date_string, date_format)
        except Exception as e:
            #print(f"Error: {date_string} not regular format style {date_format} ")
            date_string = date_string.replace(" ", "T")
            date_string = date_string.split("+")[0] + "Z"
        return date_string

    def postProcessIssue(self,eventInfo, fileInfo):
        participants = eventInfo["actor"].unique().tolist()
        event_timeline = eventInfo[["event", "created_at"]]
        event_timeline['created_at'] = event_timeline['created_at'].map(self.reformatTime)
        assignees_ls = eventInfo["assginees"].dropna().tolist()
        assignees = [item for sublist in assignees_ls for item in sublist]

        return {"participants":participants, "event_timeline":event_timeline,
                    "assignees":assignees, "file_paths":None, "commit_shas":[]}

    def postProcessPR(self,eventInfo, fileInfo):
        participants = eventInfo["actor"].unique().tolist()
        event_timeline = eventInfo[["event", "created_at"]]
        event_timeline['created_at'] = event_timeline['created_at'].map(self.reformatTime)
        assignees_ls = eventInfo["assginees"].dropna().tolist()
        assignees = [item for sublist in assignees_ls for item in sublist]
        # fileInfo
        file_paths = fileInfo["files"].tolist()
        file_paths = set(item for sublist in file_paths for item in sublist)
        commit_shas = fileInfo["commits"].tolist()
        commit_shas = set(item for sublist in commit_shas for item in sublist)
        return {"participants":participants, "event_timeline":event_timeline,
                    "assignees":assignees, "file_paths":file_paths, "commit_shas":commit_shas}


    def getStateByindex(self, index:int):
        return self.state_line[index]

    def getParticipantsUntilindex(self, index:int):
        return self.participants[:index]

    def getTimeLineByindex(self, index:int):
        return self.time_line[index]

    def getNeareststateline(self, time:str):
        # 从时间线中获取距离给定时间 最近的时间下标以及对应的状态
        pass

    def getState(self):
        return self.state_line

    # 一种普遍的使用方式，就是给定artifactId, 返回其指定的时间点
    def gettimeline(self):
        return self.time_line

    def getParticipants(self):
        return self.participants


