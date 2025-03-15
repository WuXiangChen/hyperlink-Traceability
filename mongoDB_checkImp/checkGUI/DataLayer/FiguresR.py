# 本节主要目的是直接从fileFolder中重建所有的figures，然后直接进行记载实际图像，并进行判断
## 跳过file这个层级
import os
import queue

import simplejson as json
import numpy as np
from .utils import readJsonFile, symmetric_matrix_addition, random_select


class FiguresR:

    def __init__(self, folderPath,dml):
        self.folderPath = folderPath
        self.figures = queue.Queue()
        self.dml = dml
        self.getAllotedFigure(self.allotmentFigures())

    def allotmentFigures(self):
        dirName = os.path.basename(self.folderPath)
        fileLS = os.listdir(self.folderPath)
        if dirName.endswith("BigGraph"):
            # 从文件夹中随机选择一定数量的文件
            fileSelected = random_select(fileLS, 1)
        elif dirName.startswith("keyAndSmallGraph"):
            # 从文件夹中随机选择一定数量的文件
            fileSelected = random_select(fileLS, 0.3)
        elif dirName.startswith("NoKeyAndSmallGraph"):
            # 从文件夹中随机选择一定数量的文件
            fileSelected = random_select(fileLS, 0.5)
        else:
            fileSelected = None
        return fileSelected

    # 循环这个文件夹 读取其中所有的成对图像 并将其放置在一个队列中
    def getAllotedFigure(self,fileSelected):
        for filePath in fileSelected:
            filePath_ = os.path.join(self.folderPath,filePath)
            comZeroIndex = self.parseFigureDict(filePath_)
            for pair in comZeroIndex:
                self.figures.put(pair)

    def getNotZeroIndex(self,subFigure):
        np_array = np.array(subFigure)
        # 这里的array需要进行变化，
        sym_array = symmetric_matrix_addition(np_array)
        upper_triangle = np.triu(sym_array)
        non_zero_indices = np.where(upper_triangle != 0)
        row_index = non_zero_indices[0]
        col_index = non_zero_indices[1]
        non_zero_index = list(zip(row_index, col_index))
        return non_zero_index

    def parseFigureDict(self,filePath):
        # 获取基本信息
        comZeroIndex = readJsonFile(filePath)
        return comZeroIndex

    def getFigure(self):
        # 不断checkExist 然后返回
        while not self.figures.empty():
            pair = self.figures.get()
            if not self.dml.checkExist(pair[0],pair[1]):
                return pair

        return None