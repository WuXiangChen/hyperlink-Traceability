# 本节的目的是对于给定数据，提供必要的处理格式
from .DataPersistenceLayer import DataPersistenceLayer
import simplejson as json
from .utils import readJsonFile, writeJsonFile

class DataManipulateLayer:
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.dps = DataPersistenceLayer(owner,repo)
        self.filePath = self.dps.filePath

    # 这里得到给定的节点对应的值
    def getValue(self,leftNode,rightNode):
        self.checkvalidity(leftNode, rightNode)
        data = readJsonFile(self.filePath)
        return data[leftNode + "-" + rightNode]

    def recordLinking(self,leftNode,rightNode,verified):
        self.checkvalidity(leftNode,rightNode)
        data = readJsonFile(self.filePath)
        data[leftNode + "-" + rightNode] = verified
        writeJsonFile(self.filePath, data)

    # 检查有效性
    def checkvalidity(self,leftNode,rightNode):
        if leftNode is None or rightNode is None :
            raise ValueError("The input _originalData_Archive is not valid")

    # 判断这里的键值对是否已经存在
    def checkExist(self,leftNode,rightNode):
        self.checkvalidity(leftNode, rightNode)
        data = readJsonFile(self.filePath)
        keyLS = list(data.keys())
        key1 = leftNode + "-" + rightNode
        key2 = rightNode + "-" + leftNode
        if key1 in keyLS or key2 in keyLS:
            return True
        else:
            return False
