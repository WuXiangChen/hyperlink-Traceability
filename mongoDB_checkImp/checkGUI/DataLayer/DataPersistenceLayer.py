# 本节目的是进行定义数据持久化层的类，用于对数据进行持久化操作
# 注意这里的这里的数据将以json格式进行存储

import os
import simplejson as json
from typing import Union

# 这里数据持久化针对的是Related Linking
class DataPersistenceLayer:
    def __init__(self,owner,repo):
        self.owner = owner
        self.repo = repo
        self.filePath = f"./DataLayer/DataPersistenceVerified/{owner}_{repo}_.json"

        # 判断文件路径是否存在，不存在则创建
        if not os.path.exists(self.filePath):
            self.initalJson(self.filePath)


    def initalJson(self,filepath):
        # 其中Num-Num 代表着两个不同的访问节点，而0则代表着人工判断它们是否相关的结果
        verifiedData = {"Num-Num":0}
        with open(filepath, 'w') as f:
            json.dump(verifiedData, f)
