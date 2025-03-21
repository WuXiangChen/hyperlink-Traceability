# 本节的主要目的是 为了辅助主函数过程中的诸多操作，比如数据的加载，模型的保存等等

'''
    导包区
'''
import glob
from itertools import chain
import itertools
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
from GraphLinker._0_dataClean import DataCleaner
from torch_geometric.data import Data,Batch
from torch_geometric.utils import from_networkx
import numpy as np
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
        row, col = torch.where(incidence_matrix.T)
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
        stopwords = [word.strip() for word in stopwords]
        words = input_text.split(" ")
        filtered_words = [word for word in words if word not in stopwords]
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
            filename = file.split("/")[-1]
            if filename.startswith(reponame):
                return Utils.loadJson(file)
            if filename.startswith("graph_"+reponame):
                return Utils.loadJson(file)
        return None
    
    @staticmethod
    def getnearsetFileDf(reponame: str, artifactFilePaths: str, style: str):
        files = glob.glob(artifactFilePaths+f"/*.{style}")
        for file in files:
            if reponame in file:
                return pd.read_json(file)
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
                    print(ValueError(f"node_id {node_id} not in repo id {arts.pName} not completed!"))
                    continue
                node_sen = arts.artifact_dict[node_id]
                nodes_list.append(node_id)
                if 'desc' not in node_sen.keys() or node_sen['desc'] is None:
                    node_sen['desc'] = ''
                if 'title' not in node_sen.keys() or node_sen['title'] is None:
                    node_sen['title'] = ''
                #node_word = node_sen["title"] + "\n" + node_sen["desc"]
                node_word = node_sen["title"]
                node_cleaned_word = dataClean(node_word).clean_data()
                edge_sentence.append(node_cleaned_word)
            node_sentence_list.append(edge_sentence)
        return node_sentence_list, nodes_list
    
    @staticmethod
    def get_structure_fea(artifacts, edges, stru_max_len):
        torch_cgs = []
        for k, edge in enumerate(edges):
            # 这里直接建图， 下面就是对图节点赋值的过程
            art_index = edge[-1].item()
            edge = edge[:-1]
            art = artifacts[art_index]
            ## 先获取所有的有效节点
            node_varified = []
            for node_id in edge:
                if node_id == -1:
                    continue
                node_varified.append(node_id.item())
            cg = Utils.get_fully_network([node_varified])[0]
            for node_id in node_varified:
                if node_id not in list(art.indexToArtId.keys()):
                    print(ValueError(f"node_id {node_id} not in repo id {art.pName} not completed!"))
                    continue
                index = art.indexToArtId[node_id]
                stru_fea = art.struFea[index] # 这里要做padding，将所有的节点的结构特征补充到最长的状态
                stru_fea_padded = np.pad(stru_fea, (0, stru_max_len - len(stru_fea)), mode='constant', constant_values=0)
                if node_id in cg.nodes:  # 检查节点是否在 cg 中
                    cg.nodes[node_id]["stru_fea"] = stru_fea_padded
                    cg.nodes[node_id]["batch"] = k # 将 stru_fea 赋值给对应节点
            cg = from_networkx(cg, group_node_attrs=["stru_fea", "batch"])
            torch_cgs.append(cg)
        data = Batch.from_data_list(torch_cgs)
        return data

    @staticmethod
    def get_fully_network(edges, tensor=False):
        connected_graph = []
        for edge in edges:
            if not tensor:
                points = [node for node in edge if node!=-1]
            else:
                points = [node.item() for node in edge[:-1] if node!=-1]
            graph = nx.complete_graph(points)
            connected_graph.append(graph)
        return connected_graph


    @staticmethod
    def pad_list(train_data, test_data, fill_value=-1):
        if len(train_data[0])!=0:
            max_length_1 = max(len(inner) for inner in chain(*train_data))  # 找到最长的子列表
        else:
            max_length_1 = 0
        max_length_2 = max(len(inner) for inner in chain(*test_data))  # 找到最长的子列表
        max_length = max(max_length_1, max_length_2)
        train_data_ = []
        test_data_ = []
        for j, data in enumerate(train_data):
            for k, inner in enumerate(list(data)):
                padded_data = inner + [fill_value] * (max_length - len(inner)) + [j]  # 填补
                train_data_.append(padded_data)

        for j, data in enumerate(test_data):
            for k, inner in enumerate(list(data)):
                padded_data = inner + [fill_value] * (max_length - len(inner)) + [j]  # 填补
                test_data_.append(padded_data)
        return train_data_, test_data_, max_length

    @staticmethod
    def P2P_Expanded(selected_data, edges, addTwoComORNot=False, testFlag=False):
        P2PDatasets = []
        P2PLabels = []
        all_index = len(selected_data)
        if all_index==0:
            return None,None
        for hpLink in selected_data:
            if len(hpLink)!=2 and (-1 in hpLink or hpLink[-1]==0):
                # 如果包含 -1，则去除 -1 和最后一个元素
                hpLink = hpLink[hpLink != -1][:-1]                
            if len(hpLink)<2:
                continue
            combinations = list(itertools.combinations(hpLink, 2))
            for com in combinations:
                inverse_com = com[::-1]
                if (com in edges) or (inverse_com in edges):
                    P2PDatasets.append(list(com))
                    P2PLabels.append(1)
                else:
                    P2PDatasets.append(list(com))
                    P2PLabels.append(0)
                all_index+=1
        # 加个方法对正负样本进行平衡
        if not testFlag:
            P2PDatasets, P2PLabels = Utils.balance_samples(P2PDatasets, P2PLabels)
        return P2PDatasets, P2PLabels
    
    @staticmethod
    def balance_samples(P2PDatasets, P2PLabels):
        """
        平衡正负样本数量，使其数量相同。
        """
        positive_samples = [i for i, label in enumerate(P2PLabels) if label == 1]
        negative_samples = [i for i, label in enumerate(P2PLabels) if label == 0]

        # 取较少的样本数量来平衡
        if len(positive_samples) > len(negative_samples):
            return P2PDatasets, P2PLabels
        else:
            num_samples = len(positive_samples)
            # 随机选择样本，使得正负样本数量相同
            balanced_pos = random.choices(positive_samples, k=num_samples)
            balanced_neg = random.sample(negative_samples, num_samples)

            # 更新 P2PDatasets 和 P2PLabels
            balanced_indices = balanced_pos + balanced_neg
            random.shuffle(balanced_indices)

            # 返回平衡后的数据和标签
            P2PDatasets = [P2PDatasets[i] for i in balanced_indices]
            P2PLabels = [P2PLabels[i] for i in balanced_indices]
            return P2PDatasets, P2PLabels

    @staticmethod
    def calculate_metrics(conf_matrix):
        # 提取混淆矩阵的元素
        TN = conf_matrix[0, 0]  # 真负例
        FP = conf_matrix[0, 1]  # 假正例
        FN = conf_matrix[1, 0]  # 假负例
        TP = conf_matrix[1, 1]  # 真正例

        # 计算 Accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        # 计算 Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # 计算 Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # 计算 F1-Score
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 返回结果字典
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def print_args(args):
        """打印参数的方法"""
        print("===============================")
        print("Parameters:")
        print(f"Repo Name: {args.saveName}")
        print(f"CUDA Number: {args.cudaN}")
        print(f"Number of Folds: {args.num_folds}")
        # print(f"Test Ratio: {args.test_ratio}")
        # print(f"Running Type: {args.running_type}")
        # print(f"Freeze: {args.freeze}")
        # print(f"With Knowledge: {args.with_knowledge}")
        # print(f"Cat: {args.cat}")
        print(f"Training Type: {args.training_type}")
        print(f"Training_with_gnn: {args.training_with_gnn}")
        print("===============================")

def getCheckPointFilePath(finetuneFolder, cur_repoName, k):
    if k==0:
        root_checkpoint = f'../CHESHIRE/{cur_repoName}/'
    else:
        root_checkpoint = finetuneFolder + f"/{k}/"
    files = os.listdir(root_checkpoint)
    file_paths = [os.path.join(root_checkpoint, file) for file in files]
    # 按照文件的创建时间排序
    sorted_files = sorted(file_paths, key=os.path.getctime)
    checkpointPath = sorted_files[-1]
    save_checkpoint = checkpointPath
    return save_checkpoint