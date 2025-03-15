from itertools import combinations
import os
import torch
from typing import Dict, List
import numpy as np
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class generateFeatureDynamically:
    def __init__(self):
        pass

    def generateFeatureGivenHyperlink(self, hyperlinks: List[List[Dict]],  arts: Dict[int, Dict]):
        sta_fea_forHyperlinks = []
        for hyperlink in hyperlinks:
            art_index = hyperlink[-1].item()
            artifact_dict = arts[art_index].artifact_dict
            hyperlink = hyperlink[:-1]
            edge = []
            for node_id in hyperlink:
                if node_id == -1:
                    continue
                edge.append(node_id.item())
            hyperlink = edge
            features = {
                'Average Lag': self.calculate_average_lag(hyperlink, artifact_dict),
                'Average Branch Size': self.calculate_average_branch_size(hyperlink, artifact_dict),
                'Average Files Touched': self.calculate_average_files_touched(hyperlink, artifact_dict),
                'FileName Overlapped': self.calculate_filename_overlapped(hyperlink, artifact_dict),
                'userNumber': self.calculate_user_number(hyperlink, artifact_dict),
                'userOverlapped': self.calculate_user_overlapped(hyperlink, artifact_dict),
                'Average Cs(title, report)': self.calculate_average_cs_title_report(hyperlink, artifact_dict)
            }
            sta_fea_forHyperlinks.append(features)
        # 将列表转换为 Pandas DataFrame
        df = pd.DataFrame(sta_fea_forHyperlinks)
        return df

    def calculate_average_lag(self, hyperlink, artifact_dict):
        total_lag = 0
        num_nodes = len(hyperlink)
        for node in hyperlink: # hyperlink中存储的就是节点
            artId = node
            if artId in artifact_dict:
                staCon = artifact_dict[artId]
                if staCon['open_time']==None:
                    continue
                created_at = staCon['open_time']
                closed_at = staCon.get('close_time')
                if closed_at:
                    closed_at = closed_at
                    lag = (closed_at - created_at).total_seconds() / 3600  # 转换为小时
                else:
                    lag = 1000  # 如果没有 closed_at，赋予最大值
                total_lag += lag
        
        return total_lag / num_nodes if num_nodes > 0 else 0

    def calculate_average_branch_size(self, hyperlink, artifact_dict):
        total_branch_size = 0
        total_pr_num = 0
        for node in hyperlink:
            artId = node
            if artId in artifact_dict.keys():
                staCon = artifact_dict[artId]
                if "commits" not in staCon or staCon['commits']==None:
                    continue
                if 'commits' in staCon:
                    total_pr_num += 1
                    total_branch_size += len(staCon.get('commits', []))

        return total_branch_size / total_pr_num if total_pr_num > 0 else 0

    def calculate_average_files_touched(self, hyperlink, artifact_dict):
        total_files_touched = 0
        total_pr_num = 0
        for node in hyperlink:
            artId = node
            if artId in artifact_dict.keys():
                staCon = artifact_dict[artId]
                if "files" not in staCon or staCon['files']==None:
                    continue
                if 'files' in staCon:
                    total_pr_num += 1
                    total_files_touched += len(staCon.get('files', []))
        return total_files_touched / len(hyperlink) if total_pr_num > 0 else 0

    def calculate_filename_overlapped(self, hyperlink, artifact_dict):
        # 初始化交集和并集
        union_set = set()
        file_count = {}
        for node in hyperlink:
            artId = node
            if artId in artifact_dict.keys():
                staCon = artifact_dict[artId]
                if "files" not in staCon or staCon['files']==None:
                        continue
                files_touched = set(staCon.get('files', []))
                # 更新并集并统计文件出现次数
                for file in files_touched:
                    union_set.add(file)
                    if file in file_count:
                        file_count[file] += 1
                    else:
                        file_count[file] = 1
        # 计算独立文件和重复文件的数量
        unique_files = len(file_count)  # 独立文件的数量
        duplicate_files = sum(count for count in file_count.values() if count > 1)  # 重复文件的数量

        # 计算比例
        if unique_files > 0:
            total_file_overlap = duplicate_files / unique_files
        else:
            total_file_overlap = 0  # 防止除以零

        return total_file_overlap

    def calculate_user_number(self, hyperlink, artifact_dict):
        all_users = set()
        for node in hyperlink:
            artId = node
            if artId in artifact_dict.keys():
                staCon = artifact_dict[artId]
                if "reviewers" not in staCon or "author" not in staCon or staCon['reviewers']==None or staCon['author']==None:
                    continue
                users = set(staCon.get('reviewers', []))
                author = staCon.get('author')
                if author:
                    users.add(author)
            all_users.update(users)
        return len(all_users)

    def calculate_user_overlapped(self, hyperlink, artifact_dict):
        union_set = []
        for node in hyperlink:
            artId = node
            if artId in artifact_dict.keys():
                staCon = artifact_dict[artId]
                # 获取 reviewers 和 author
                if staCon['reviewers']==None or staCon['author']==None:
                    continue
                users = set(staCon.get('reviewers', []))
                author = staCon.get('author', [])
                if author:
                    users.add(author)
                union_set.extend(users)
        name_count = {}
        for name in union_set:
            if name in name_count:
                name_count[name] += 1
            else:
                name_count[name] = 1

        # 计算独立名字和重复名字的数量
        unique_names = len(name_count)  # 独立名字的数量
        duplicate_names = sum(count for count in name_count.values() if count > 1)  # 重复名字的数量

        # 计算比例
        if unique_names > 0:
            ratio = duplicate_names / unique_names
        else:
            ratio = 0  # 防止除以零
        return ratio

    def calculate_average_cs_title_report(self, hyperlink, artifact_dict):
        total_cs = 0
        num_pairs = 0
        for node1, node2 in combinations(hyperlink, 2):
            artId1 = node1
            artId2 = node2
            if artId1 in artifact_dict.keys() and artId2 in artifact_dict.keys():
                staCon1 = artifact_dict[artId1]
                staCon2 = artifact_dict[artId2]
                title1 = staCon1.get('title', '')
                report1 = staCon1.get('desc', '')
                title2 = staCon2.get('title', '')
                report2 = staCon2.get('desc', '')
                cs_value = self.cos_sim(title1 + report1, title2 + report2)
                total_cs += cs_value
                num_pairs += 1
        return total_cs / num_pairs if num_pairs > 0 else 0

    def cos_sim(self, text_a, text_b):
        vector_a = self.encode_text_with_glove(text_a)
        vector_b = self.encode_text_with_glove(text_b)
        num = np.dot(vector_a, vector_b)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        return num / denom if denom != 0 else 0

    def encode_text_with_glove(self, text, glove_model_name="../dataset/glove.6B/"):
        # 加载预训练的 GloVe 词向量模型
        wordList = np.load(glove_model_name+'/wordsList.npy')
        wordVectors = np.load(glove_model_name+'/wordVectors.npy')
        # 将文本分词
        words = text.lower().split()
        encoded_text = np.zeros((len(words), 50))
        for i, word in enumerate(words):
            if word in wordList:
                index = np.where(wordList == word)[0][0]
                encoded_text[i] = wordVectors[index]
        # 按列求和，取平均，并取其前50维度的信息
        mean_encoded_50 = np.mean(encoded_text, axis=0)
        return mean_encoded_50.tolist()



# gFD = generateFeatureDynamically("angular")
# gFD.generateFeatureGivenPair(['10003', '10010'])





