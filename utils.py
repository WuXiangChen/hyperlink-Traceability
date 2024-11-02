# 本节的主要目的是 为了辅助主函数过程中的诸多操作，比如数据的加载，模型的保存等等

'''
    导包区
'''
import glob
import json
import math
import os.path
import pickle
import random
import joblib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import torch
from model._0_dataClean import DataCleaner
seed = 43
random.seed(seed)
np.random.seed(seed)

class Utils:
    def __init__(self, repoName):
        self.repoName = repoName
        # issueIndexPath = f'dataset/{self.repoName}/Index/issue_index.txt'
        # prIndexPath = f'dataset/{self.repoName}/Index/pr_index.txt'
        # self.issueIndex = self.readTxtFile(issueIndexPath)
        # self.prIndex = self.readTxtFile(prIndexPath)
        # self.staDataRootPath = f'dataset/{self.repoName}/staFeature'
        # self.issueEventPath = self.staDataRootPath+f'/issueEventInfo.json'
        # self.prEventPath = self.staDataRootPath+f'/prEventInfo.json'
        # self.prFileAndCommitPath = self.staDataRootPath+f'/prFileAndCommit.json'
        self.connect_components = None
        self.repoG = None
        # dataset/angular/originalDataSupport/linkingPairs.json
        linkingPairPath = f'dataset/{self.repoName}/originalDataSupport/linkingPairs.json'
        self.linkingPairs = self.getLinkingPairs(linkingPairPath)
        self.posHyperLink = self.generatePosGraph()

    def generatePosGraph(self):
        G = nx.Graph()
        for data in self.linkingPairs:
            sou = data[0]
            tar = data[1]
            G.add_edge(int(sou), int(tar))
        self.repoG = G
        self.connect_components = [list(map(int, component)) for component in nx.connected_components(G)]
        # self.get_adjacency_matrix(G)
        #self.split_edges_and_save(G)
        # 取所有的边，取其中0.9的边作为训练集，0.1的边作为测试集，并将结果存储为npz文件
        return self.connect_components

    def getNodes(self):
        return list(self.repoG.nodes())

    def get_adjacency_matrix(self):
        ########## 保存adjacency_matrix ##########
        adjacency_matrix = nx.adjacency_matrix(self.repoG).toarray()
        return adjacency_matrix

    def get_negtive_adjacency_matrix(self, negative_adjacents):
        ########## 保存adjacency_matrix ##########
        node_list = list(self.repoG.nodes())
        neg_adjacency_matrix = np.zeros((len(node_list), len(node_list)))
        for neg_hyperlink in negative_adjacents:
            for node_pair in neg_hyperlink:
                source, target = node_pair
                source_index = node_list.index(source)
                target_index = node_list.index(target)
                neg_adjacency_matrix[source_index, target_index] = 1
                neg_adjacency_matrix[target_index, source_index] = 1
        return neg_adjacency_matrix


    def split_edges_and_save(self, G, train_ratio=0.9):
        # 获取所有边
        all_edges = list(G.edges())
        # 随机打乱边的顺序
        np.random.shuffle(all_edges)
        # 计算训练集和测试集的边数
        train_size = int(len(all_edges) * train_ratio)
        # 划分训练集和测试集
        train_edges = all_edges[:train_size]
        test_edges = all_edges[train_size:]
        # 保存结果为npz文件
        np.savez_compressed(f"dataset/{self.repoName}/processed/train_data.npz", train_data=train_edges)
        np.savez_compressed(f"dataset/{self.repoName}/processed/test_data.npz", test_data=test_edges)
        print(f"Train edges: {len(train_edges)}")

    def visualizeGraph(self, components_index):
        longest_component = max(self.connect_components, key=len)
        #components = self.connect_components[longest_component]
        subG = self.repoG.subgraph(longest_component)
        # 将这里的边关系导出来
        edges = list(subG.edges())
        # 将边关系写入 TXT 文件
        with open("longestEdge.txt", 'w') as f:
            for edge in edges:
                f.write(f"{edge[0]} {edge[1]}\n")
        print()
        #self.draw_graph(subG)

    @staticmethod
    def create_hyperedge_index(incidence_matrix):
        # 转置关联矩阵，并找到非零元素的索引
        row, col = torch.where(incidence_matrix.T)
        # 将列索引和行索引连接起来形成超边索引
        # 结果是一个二维张量，其中第一行包含列索引（原始行索引），
        # 第二行包含行索引（原始列索引）
        hyperedge_index = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        # 返回创建的超边索引
        return hyperedge_index

    def getLinkingPairs(self, linkingPairPath):
        pairs = self.load_json(linkingPairPath)
        linkingPairs = set()
        for _ in pairs:
            pair = list(_.items())[0]
            linkingPairs.add(pair)
        return linkingPairs

    def judgePRORIssue(self, artifactId: str) -> str:
        if artifactId in self.prIndex:
            return "PR"
        elif artifactId in self.issueIndex:
            return "Issue"
        else:
            #raise Exception(f"Artifact {artifactId} is not in the dataset.")
            return "None"

    def getNodeIds(self, dataset) -> list:
        temp_nodes = set()
        for item in dataset:
            temp_nodes.add(item[0])
            temp_nodes.add(item[1])
        return list(temp_nodes)

    # 加载txt文件
    def readTxtFile(self, filePath) -> list:
        re = []
        with open(filePath, 'r') as f:
            for line in f.readlines():
                re.append(line.split(" ")[1].strip())
        return re

    # 加载json文件
    def load_json(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def loadJson(path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    def save_json(self,  data, path: str):
        with open(path, 'w') as f:
            return json.dump(data, f)

    @staticmethod
    def saveJson(data, path: str):
        with open(path, 'w') as f:
            return json.dump(data, f)

    # 一个快速打包工具，在提供地址的情况下，快速将给定对象打包成可存储的形式
    def save_pickle(self, object_, filename):
        with open(filename, 'wb') as f:
            pickle.dump(object_, f)

    def save_artifact_pickle(self, artifact, filename):
        existed_artifact = []
        if os.path.exists(filename):
            existed_artifact = self.load_pickle(filename)
        existed_artifact.extend(artifact)
        with open(filename, 'wb') as f:
            joblib.dump(existed_artifact, f)

    # 一个快速读取工具，在提供地址的情况下，快速将给定对象读取出来
    def load_pickle(self, filename):
        with open(filename,"rb") as f:
            return pickle.load(f)

    @staticmethod
    def transNxGraphToTorch(G: nx.Graph):
        """将 NetworkX 图转换为 PyTorch 张量"""
        # 转换为 PyTorch 张量
        feature_dim = len(list(G.nodes(data=True))[0][1]['ft'])
        x = torch.empty((len(G.nodes), feature_dim), dtype=torch.float)
        for index, node in enumerate(G.nodes(data=True)):
            x[index] = node[1]['ft'] # 将边转换为元组列表
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        # 创建 PyTorch Geometric 数据对象
        data = Data(x=x, edge_index=edge_index)
        return data

    @staticmethod
    def draw_graph(G):
        """绘制给定的 NetworkX 图"""
        plt.figure(figsize=(8, 6))  # 设置图形大小
        nx.draw(G, with_labels=True, node_color='lightblue', node_size=1000, font_size=8, font_color='black')
        plt.title("Graph Visualization")  # 设置标题
        plt.show()

    @staticmethod
    def get_cuda_device(cuda_N):
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if 0 <= cuda_N < device_count:
                device = torch.device(f'cuda:{cuda_N}')
                print(f"Using CUDA device: {device}")
            else:
                device = torch.device('cpu')
                print(f"CUDA device index {cuda_N} is not invalid. Using CPU.")
        else:
            device = torch.device('cpu')
            print("CUDA is not available. Using CPU.")

        return device

    @staticmethod
    def print_evaluation_results(accuracy, precision, recall, f1):
        results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

    @staticmethod
    def remove_stopwords(input_text, stopwords_path="model/resource/stopwords.txt"):
        # 读取停用词文件并去掉换行符
        with open(stopwords_path, 'r', encoding='utf-8') as file:
            stopwords = file.read().splitlines()
        # 去掉停用词中的空白字符
        stopwords = [word.strip() for word in stopwords]
        # 将输入字符串按空格分割成单词
        words = input_text.split(" ")
        # 去除停用词
        filtered_words = [word for word in words if word not in stopwords]
        # 返回去除停用词后的字符串
        return " ".join(filtered_words)

    @staticmethod
    def is_linked(source, target, linkingpairs_set):
        # 使用集合的无序特性来检查链接
        return frozenset([str(source), str(target)]) in linkingpairs_set
    # 示例用法
    # result = remove_stopwords("这是一个示例输入", "path/to/stopwords.txt")

    @staticmethod
    def getArtifactDict(artifacts: list):
        artDict = {}
        for art in artifacts:
            artId = art.getArtifactId()
            artDict[artId] = art
        # 按照artId进行排序
        sorted_artDict = dict(sorted(artDict.items()))
        return sorted_artDict

    @staticmethod
    def sumBeginWithPlus(text: str):
        # 统计以+开头的行数
        text_n = text.split("\n")
        return len([line for line in text_n if line.startswith("+")])

    @staticmethod
    def getnearsetFile(reponame: str, artifactFilePaths: str, style: str):
        files = glob.glob(artifactFilePaths+f"/*.{style}")
        for file in files:
            if reponame in file:
                return Utils.loadJson(file)
        return None

    @staticmethod
    def getAnotherNegSet(test_pos_set):
        non_zero_indices = [np.nonzero(test_pos_set[:, col])[0].tolist() for col in range(test_pos_set.shape[1])]
        test_index_set = set(id_ for ids in non_zero_indices for id_ in ids)
        all_node_len = len(test_pos_set)
        # 在这些节点中进行随机组合
        from DataStructure._1_HyperlinkGenerate import generateHyperLinkDataset
        gh = generateHyperLinkDataset(range(all_node_len), non_zero_indices)
        negative_set,negative_adjacents = gh.generate_specific_negative_samples(positive_set=non_zero_indices, num_samples=1, selected_nodes=test_index_set)

        num_hyperedges = len(negative_set)
        num_artifacts = all_node_len
        matrix = np.zeros((num_artifacts, num_hyperedges), dtype=int)
        for hyperedge_idx, hyperedge in enumerate(negative_set):
            for artifactId in hyperedge:
                matrix[artifactId, hyperedge_idx] = 1
        return matrix

    @staticmethod
    def getValidIndexFromList(data_incidence, artifact_dict):
        valid_index = {}
        artifact_dict_T = {value:key for key, value in artifact_dict.items()}
        data_incidence = data_incidence.T
        non_zero_indices = list(zip(*data_incidence.nonzero()))
        for row, col in non_zero_indices:
            if row not in valid_index.keys():
                valid_index[row] = []
            valid_index[row].append(artifact_dict_T[col])

        return valid_index


    @staticmethod
    def process_edges_data(artifacts, edges):
        dataClean = DataCleaner
        nodes_list = []
        node_sentence_list = []
        for edge in edges:
            art_index = edge[-1].item()
            edge = edge[:-1]
            arts = artifacts[art_index]
            edge_sentence = []
            for node_id in edge:
                node_id = node_id.item()
                if node_id == -1:
                    continue
                
                if node_id not in list(arts.artifact_dict.keys()):
                    raise ValueError(f"repo id {art_index} not completed!")

                node_sen = arts.artifact_dict[node_id]
                nodes_list.append(node_id)
                if 'desc' not in node_sen.keys() or node_sen['desc'] is None:
                    node_sen['desc'] = ''
                elif 'title' not in node_sen.keys() or node_sen['title'] is None:
                    node_sen['title'] = ''
                # node_word = node_sen["title"] + "\n" + node_sen["desc"]
                node_word = node_sen["title"]
                node_cleaned_word = dataClean(node_word).clean_data()
                edge_sentence.append(node_cleaned_word)
            node_sentence_list.append(edge_sentence)
        return node_sentence_list, nodes_list

    @staticmethod
    def get_fully_network(edges):
        connected_graph = []
        for edge in edges:
            points = [node.item() for node in edge if node!=-1]
            graph = nx.complete_graph(points)
            connected_graph.append(graph)
        return connected_graph


    @staticmethod
    def pad_list(train_data, fill_value=-1, index_changes=[], train=True):
        if train:
            repo_index = index_changes["train_index"]
        else:
            repo_index = index_changes["test_index"]
        max_length = max(len(inner) for inner in train_data)  # 找到最长的子列表
        data = []
        j = 0
        for k, inner in enumerate(train_data):
            if k>repo_index[j]:
                j+=1
            padded_data = inner + [fill_value] * (max_length - len(inner)) + [j]  # 填补
            data.append(padded_data)
        return data