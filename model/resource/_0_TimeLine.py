# 本节的主要目的是生成所有制品节点的时间线，主要包括每个节点的开始时间
'''
    data = {
        'Event': ['Event 1', 'Event 2', 'Event 3', 'Event 4'],
        'Date': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-10', '2023-04-20'])
    }
    可以将时间线制备成这种格式，这样既方便支持时间线的可视化，也支持时间线的数据分析
'''
from collections import defaultdict

from matplotlib import pyplot as plt

'''
    导包区
'''
import pandas as pd
from typing import List
from model.resource._0_Artifact import Artifact
from utils import Utils


class TimeLine_Artifact:

    def __init__(self, repoName: str,  arts: List[Artifact], split_num: int):
        self.ArtifactTimeLine = defaultdict(list)
        self.utils = Utils(repoName)
        self.arts = arts
        self.dfTL = None
        self.setTimeLine(split_num)

    def generateTimeLine(self, split_num: int):
        for k, art in enumerate(self.arts):
            self.ArtifactTimeLine["artifactID"].append(art.getArtifactId())
            self.ArtifactTimeLine["artifactType"].append(art.getArtifactType())
            self.ArtifactTimeLine["createdAt"].append(art.getArtifactFeature("createdAt"))
            self.ArtifactTimeLine["filepath"].append(art.getArtifactFeature("filepath"))
            last_event = art.getArtifactFeature("event_timeline")
            if len(last_event) == 0:
                self.ArtifactTimeLine["last_event_time"].append(art.getArtifactFeature("createdAt"))
            else:
                self.ArtifactTimeLine["last_event_time"].append(last_event[-1])
            self.ArtifactTimeLine["art"].append(art)
            if k < split_num:
                self.ArtifactTimeLine["type"].append("train")
            else:
                self.ArtifactTimeLine["type"].append("test")
        return self.ArtifactTimeLine

    def setTimeLine(self, split_num: int):
        self.dfTL = pd.DataFrame(self.generateTimeLine(split_num))
        # 将 createdAt 和 last_event_time 列转换为 datetime 类型
        self.dfTL["createdAt"] = pd.to_datetime(self.dfTL["createdAt"])
        self.dfTL["last_event_time"] = pd.to_datetime(self.dfTL["last_event_time"])
        self.dfTL.sort_values('createdAt', inplace=True)
        self.dfTL.reset_index(drop=True, inplace=True)

    def getTimeline(self):
        return self.dfTL

    def visualizeTimeLine(self):
        # 创建一个数据框，包含事件和日期
        df = self.dfTL
        # 按照时间排序
        df = df.sort_values('createdAt')
        # 仅选择前100行数据
        df = df.head(50)

        # 绘制时间线
        plt.figure(figsize=(20, 10))
        plt.yticks([])  # 隐藏y轴
        plt.title('Timeline of Artifact')
        plt.xlabel('createdAt')

        # 根据artifactType选择颜色
        for i, row in df.iterrows():
            color = 'red' if row['artifactType'] == 'Issue' else 'blue' if row['artifactType'] == 'PR' else 'black'
            plt.plot(row['createdAt'], 1, 'o', color=color)  # 绘制事件点
            # 添加事件标签，设置为斜45度
            plt.text(row['createdAt'], 1.01, row['artifactID'], ha='center', rotation=45)

        plt.yticks([])  # 隐藏y轴
        plt.title('Timeline of Artifact')
        plt.xlabel('createdAt')

        # 设置x轴标签为斜45度
        plt.xticks(rotation=45)

        plt.show()
