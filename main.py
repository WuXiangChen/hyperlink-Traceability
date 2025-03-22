# 本节的主要目的是为了统筹整个项目的流程，将各个模块串联起来，实现一个完整的预测过程

'''
    导包区
'''
import argparse
import glob
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
#from torch_geometric.nn.nlp import SentenceTransformer, classes
from transformers import BertForSequenceClassification, AutoModel
from GraphLinker.models.P2PModule.utils.utils import transferArtIdIntoIndex
from utils import Utils, getCheckPointFilePath
from DataStructure._1_HyperlinkGenerate import generateHyperLinkDataset
from processerHook.processer_BAAI import processer_
# from processerHook.autoProcesser import processer_
from GraphLinker._0_Artifact import Artifacts
from GraphLinker._0_GroundTruthGraph import GroundTruthGraph
import shutil
import torch
import random
from utils import Utils
from safetensors.torch import load_file
#seed = 43
seed = 1314
torch.random.initial_seed()  
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.	
# torch.set_deterministic(True)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from torch.utils.tensorboard import SummaryWriter
import os 
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_SHM_DISABLE "] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# 这部分代码片段用于将train+test Artifacts转化成all Artifacts

def main_prepareData(root_Repo:str, repoName:str):
    # 读取数据
    data = np.load(root_Repo, allow_pickle=True)
    posHyperlink, negHyperlink = data['positive'], data['negative']
    artifact_dict = data['artifact_dict'].tolist()
    full_posHyperlink = data['full_posHyperlink']

    indices = np.random.permutation(posHyperlink.shape[1])
    posHyperlink = posHyperlink[:, indices]
    indices = np.random.permutation(negHyperlink.shape[1])
    negHyperlink = negHyperlink[:, indices]

    repos_artifact_info_path = "../dataset/other_repos_artifact_info"
    repo_artifact_info = Utils.getnearsetFile(repoName, repos_artifact_info_path, "json")
    # 生成超链接数据集

    STA_FILE_AND_COMMIT_PATH = "../dataset/STA_FeaTure/PRCommitAndFileName"
    STA_ARTIFACTINFO_PATH = "../dataset/STA_FeaTure/ArtifactInfo"
    REPO_CONNECT_INFO_PATH = "../dataset/other_repo_connect_info"
    repo_connect_info = Utils.getnearsetFile(repoName, REPO_CONNECT_INFO_PATH, "json")

    repo_sta_commitAndFiles = Utils.getnearsetFileDf(repoName, STA_FILE_AND_COMMIT_PATH, "json").to_dict()
    repo_sta_artifactInfo = Utils.getnearsetFileDf(repoName, STA_ARTIFACTINFO_PATH, "json").set_index('ArtId').T.to_dict()
    if repo_sta_commitAndFiles is None or repo_sta_artifactInfo is None:
        raise Exception("repo_sta_commitAndFileName or repo_sta_artifact_Name cannot be None")
    artifacts = Artifacts(fine_tune_model_path, repo_sta_commitAndFiles, repo_sta_artifactInfo, artifact_dict)
    artifacts.registerArtifacts(repo_artifact_info, repoName)
    # ==============Training and Testing================
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    repo_dict = {}
    all_ratio_test_list = []
    all_ratio_train_list = []
    full_index = np.arange(posHyperlink.shape[1])
    pre_index = np.empty(0).astype(np.int8)

    #1. 加载Pair2Pair的GroundTruth， groundTruthGraph缩写为gTG
    if repoName in ["angular", "ansible", "elastic", "symfony", "dotnet", "saltstack"]:
        trusted_project=True
    else:
        trusted_project=False

    gTG = GroundTruthGraph(repo_connect_info, trusted_project=trusted_project)

    for train_index, test_index in kf.split(posHyperlink.T):  # Transpose to get samples as rows
        pre_index = np.union1d(pre_index, test_index)
        if pre_index.shape[0] == len(full_index):
            break
        all_ratio_test_list.append(pre_index)
        train_index = np.setdiff1d(full_index, pre_index)  # 使用 setdiff1d 来计算差集
        all_ratio_train_list.append(train_index)
    all_ratio_test_list.append(full_index)
    all_ratio_train_list.append(np.empty(0))

    # ====================== 构建训练与测试数据集 ===================
    for k, test_index in enumerate(all_ratio_test_list):
        train_index = all_ratio_train_list[k]
        if train_index.size == 0:
            train_hyperlink = None 
            train_labels = None
            train_pos_set = None
        else:
            train_pos_set, train_neg_set = posHyperlink[:, train_index], negHyperlink[:, train_index]
            train_set = np.concatenate([train_pos_set, train_neg_set], axis=1)
            train_labels = np.concatenate([np.ones(train_pos_set.shape[1]), np.zeros(train_neg_set.shape[1])])
            # 随机打乱
            indices = np.random.permutation(train_set.shape[1])
            train_set = train_set[:, indices]
            train_labels = train_labels[indices]
            train_hyperlink = Utils.getValidIndexFromList(train_set, artifact_dict)

        test_pos_set, test_neg_set = posHyperlink[:, test_index], negHyperlink[:, test_index]

        test_pos_set = np.concatenate([test_pos_set, full_posHyperlink], axis=1)
        test_set = np.concatenate([test_pos_set, test_neg_set], axis=1)
        test_labels = np.concatenate([np.ones(test_pos_set.shape[1]), np.zeros(test_neg_set.shape[1])])

        test_hyperlink = Utils.getValidIndexFromList(test_set, artifact_dict)

        train_P2Pdatasets, train_P2PLabels = None, None

        if train_hyperlink is not None:
            train_P2Pdatasets, train_P2PLabels = gTG.generateTrainingP2PDataset(train_hyperlink)
            # test_P2Pdatasets, test_P2PLabels = gTG.generateTrainingP2PDataset(test_hyperlink, testFlag=True)
            
            #     basic_repo = f'../dataset/P2PDataSupport/{repoName}/'
            #     mapping = pd.read_json(basic_repo + 'mapping.json').T
            #     rootPath = f'AllMiddleP2PDatasets/{repoName}/{4-k}/'
            #     if not os.path.exists(rootPath):
            #         os.makedirs(rootPath)

            #     test_file_path = rootPath + f'dataset_Testing_True.pkl'
            #     test_dataset = transferArtIdIntoIndex(test_P2Pdatasets, test_P2PLabels, mapping)
            #     with open(test_file_path, 'wb') as f:
            #         pickle.dump(test_dataset, f)

            #     train_file_path = rootPath + f'dataset_Testing_False.pkl'
            #     train_dataset = transferArtIdIntoIndex(train_P2Pdatasets, train_P2PLabels, mapping)
            #     with open(train_file_path, 'wb') as f:
            #         pickle.dump(train_dataset, f)

        repo_dict[k] = [train_hyperlink, train_labels, test_hyperlink, test_labels, train_pos_set, train_P2Pdatasets, train_P2PLabels]

    return repo_dict, artifacts, gTG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The arguments set for GraphLinker')
    # 添加参数及其简写
    parser.add_argument('-r', '--saveName', type=str, default="casesandberg", help='The repository name in the dataset. This is a required parameter.')
    parser.add_argument('-c', '--cudaN', type=int, default=0, help='The GPU number to use. Default is 0.')
    parser.add_argument('-tt','--training_type', type=int, default=3, help='The Type set for Training Process. 1 for semantic, 2 for add statistic, 3 for add statistic and structure ')
    parser.add_argument('-l', '--LOPO', action='store_false', help='Disable the Leave One Project Out strategy for training the HP Module')
    parser.add_argument('-hp', '--HP_Only', action='store_false', help='Help To choose run the experiment by hp-prediction only.')
    
    args = parser.parse_args()
    Utils.print_args(args)
    training_with_gnn = args.training_with_gnn
    saveName = args.saveName
    lopo = args.LOPO
    hp_only = args.HP_Only
    print("LOPO:",str(lopo))
    device = Utils.get_cuda_device(0)
    num_folds = args.num_folds
    training_type = args.training_type
    root_path = "../dataset/hyperlink_npz/*.npz"
    files_path = glob.glob(root_path)

    LM_model_selected = "BAAI_bge-m3_small"
    fine_tune_model_path = f"model/runningModel/{saveName}/{LM_model_selected}/"
    if not os.path.exists(fine_tune_model_path):
        os.makedirs(fine_tune_model_path)
    writer_tb_log_dir='logsAndResults/logs/original_NSplited_node_LLM_Linear_Structure/'
    writer = SummaryWriter(log_dir=writer_tb_log_dir)

    results = []
    all_repo_dict = {}
    all_repo_artifacts = []
    files_path = files_path[:]

    temporal_file_path = ['../dataset/hyperlink_npz/'+saveName+".npz"]
    shutil.copytree(f"../text_LM_model/{LM_model_selected}/", fine_tune_model_path, dirs_exist_ok=True)
    for i, filePath in enumerate(temporal_file_path):
        cur_repoName = filePath.split("/")[-1].split(".")[0]
        print(cur_repoName)
        cur_repoDict, cur_artifacts, gTG = main_prepareData(filePath, cur_repoName)
        all_artifacts = [cur_artifacts]
        for k, cur in enumerate(list(cur_repoDict.keys())[::-1]):
            print("k:",k)
            all_train, all_TrainLabels, all_test, all_TestLabels = [], [], [], []
            cur_repo_dict = cur_repoDict[cur]
            model_save_path = f"../CHESHIRE/{cur_repoName}"

            if training_type!=3:
                if lopo==True:
                    finetuneFolder = model_save_path+ f"/finetune_{training_type}"
                else:
                    finetuneFolder = model_save_path+ f"/finetune_{training_type}_F"
            else:
                if lopo==True:
                    finetuneFolder = model_save_path+ f"/finetune"
                else:
                    finetuneFolder = model_save_path+ f"/finetune_F"


            if not os.path.exists(finetuneFolder):
                os.mkdir(finetuneFolder)

            embedding_model = AutoModel.from_pretrained(fine_tune_model_path,  attn_implementation="sdpa")
            if lopo:
                files = os.listdir(model_save_path)
                checkpoint = [f for f in files if f.startswith('checkpoint')][0]
                save_checkpoint = model_save_path+"/"+checkpoint + "/"
                embedding_model.load_state_dict(load_file(save_checkpoint+"model.safetensors"), strict=False)

            if k==0: 
                train_hyperlink, train_labels, test_hyperlink, test_labels, train_pos_set, P2Pdatasets, P2PLabels = cur_repo_dict
                all_train.append([])
                all_TrainLabels.extend([])
                train_pos_set = []
                all_test.append(test_hyperlink.values())
                all_TestLabels.extend(test_labels)
                setattr(cur_artifacts, 'pName', filePath)
                setattr(cur_artifacts, 'struFea', train_pos_set)
                epoch_size_set = 3
                hp_hiddenDList = None
            else:
                train_hyperlink, train_labels, test_hyperlink, test_labels, train_pos_set, P2Pdatasets, P2PLabels = cur_repo_dict
                all_train.append(train_hyperlink.values())
                all_TrainLabels.extend(train_labels)
                all_test.append(test_hyperlink.values())
                all_TestLabels.extend(test_labels)
                setattr(cur_artifacts, 'pName', filePath)
                setattr(cur_artifacts, 'struFea', train_pos_set)
                epoch_size_set = 250
                # epoch_size_set = 40
                hp_hiddenDList = cur_artifacts.hp_HiddenDList

            train_data_, test_data_, max_length = Utils.pad_list(all_train, all_test)
            all_test = np.array(test_data_)
            all_train = np.array(train_data_)

            all_TrainLabels = np.array(all_TrainLabels)
            all_TestLabels = np.array(all_TestLabels)
            
            pro = processer_(embedding_type=LM_model_selected, repoName=cur_repoName, artifacts=all_artifacts, 
                            tokenizer=cur_artifacts.tokenizer_NL, device=device, embedding_model=embedding_model,writer_tb_log_dir=writer_tb_log_dir, training_type=training_type,
                            training_with_gnn=training_with_gnn, max_length=max_length,epoch_size_set=epoch_size_set, test_k_index=k, hp_hiddenDList=hp_hiddenDList, lopo=lopo, hp=hp_only)
            
            if k!=0:
                result = pro.train(all_train, all_TrainLabels, writer=writer, P2Pdatasets=P2Pdatasets, P2PLabels=P2PLabels, k=k)
                save_checkpoint = getCheckPointFilePath(finetuneFolder, cur_repoName, k)
            else:
                save_checkpoint = getCheckPointFilePath(finetuneFolder, cur_repoName, k)

            test_repo_index = all_test
            test_repo_label = all_TestLabels
            if hp_only:
                result = pro.test_HP(test_repo_index, test_repo_label, test_flag=True, checkpointPath=save_checkpoint, gTG=gTG)
            else:
                result = pro.test_HP_P2P(test_repo_index, test_repo_label, test_flag=True, checkpointPath=save_checkpoint, gTG=gTG)
            print(result)
            results.append(result)

    output_dir = f"logsAndResults/saved_results/RQ3 AblationResult/HP-Based_TrainType3_NoLOPO/"
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.DataFrame(results)
    df.to_excel(f"{output_dir}/{saveName}_{str(training_type)}_GNN_{str(training_with_gnn)}.xlsx", index=False)
    print(df)
    print(f"The {cur_repoName} results have been saved successfully!")