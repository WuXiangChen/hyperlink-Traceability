# 本节的主要目的是为了统筹整个项目的流程，将各个模块串联起来，实现一个完整的预测过程

'''
    导包区
'''
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch_geometric.nn.nlp import SentenceTransformer, classes
from transformers import BertForSequenceClassification, AutoModel
from utils import Utils
from DataStructure._1_HyperlinkGenerate import generateHyperLinkDataset
from processerHook.processer_BAAI import processer_
from model._0_Artifact import Artifacts
from model._0_GroundTruthGraph import GroundTruthGraph
import shutil
import torch
import random
seed = 43
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
import os
import os 
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["NCCL_SHM_DISABLE "] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# 这部分代码片段用于将train+test Artifacts转化成all Artifacts

def main(root_Repo:str, device:int):
    # 读取数据
    data = np.load(root_Repo, allow_pickle=True)
    posHyperlink, negHyperlink = data['positive'], data['negative']
    artifact_dict = data['artifact_dict'].tolist()
    full_posHyperlink = data['full_posHyperlink']

    indices = np.random.permutation(posHyperlink.shape[1])
    posHyperlink = posHyperlink[:, indices]
    indices = np.random.permutation(negHyperlink.shape[1])
    negHyperlink = negHyperlink[:, indices]
    # 在这里需要注册Artifact和GroundTruth Graph的信息
    repos_connnect_info_path = "../dataset/other_repo_connect_info"
    repos_artifact_info_path = "../dataset/other_repos_artifact_info"
    repo_connect_info = Utils.getnearsetFile(repoName, repos_connnect_info_path, "json")
    repo_artifact_info = Utils.getnearsetFile(repoName, repos_artifact_info_path, "json")
    # 生成超链接数据集

    groundTruthGraph = GroundTruthGraph(repo_connect_info)
    gG = groundTruthGraph.build_groundtruth_graph() # 二阶段的匹配过程
    # groundTruthGraph.plot()
    LM_model_selected = "BAAI_bge-m3_small"

    fine_tune_model_path = f"model/{LM_model_selected}/"
    artifacts = Artifacts(fine_tune_model_path)
    artifacts.registerArtifacts(repo_artifact_info)
    # ==============Training and Testing================
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    results = []
    i = 0
    for train_index, test_index in kf.split(posHyperlink.T):  # Transpose to get samples as rows
        # if i!=0:
        #     continue
        i+=1
        # Generate train data and labels
        shutil.copytree(f"../text_LM_model/{LM_model_selected}/", fine_tune_model_path, dirs_exist_ok=True)
        # 在冻结LLM有效的情况下 这里就不用进行替换了
        # 删除CHESHIRE路径下的reponame文件
        saved_model_safetensor = f"CHESHIRE/{running_type}/{repoName}"
        # 检查文件夹是否存在
        if os.path.exists(saved_model_safetensor):
            # 删除文件夹及其内容
            shutil.rmtree(saved_model_safetensor)
            print(f"文件夹 '{saved_model_safetensor}' 已成功删除。")
        else:
            print(f"文件夹 '{saved_model_safetensor}' 不存在。")
        # 以半精度的方式加载 attn_implementation attention的计算方法
        embedding_model = AutoModel.from_pretrained(fine_tune_model_path,  attn_implementation="sdpa")
        # ====================== 构建训练与测试数据集 ===================
        train_pos_set, train_neg_set = posHyperlink[:, train_index], negHyperlink[:, train_index]
        train_set = np.concatenate([train_pos_set, train_neg_set], axis=1)
        train_labels = np.concatenate([np.ones(train_pos_set.shape[1]), np.zeros(train_neg_set.shape[1])])
        # 随机打乱
        indices = np.random.permutation(train_set.shape[1])
        train_set = train_set[:, indices]
        train_labels = train_labels[indices]

        test_pos_set, test_neg_set = posHyperlink[:, test_index], negHyperlink[:, test_index]
        another_neg_set_incidence_matrix = Utils.getAnotherNegSet(test_pos_set)
        test_neg_set = np.concatenate([test_neg_set, another_neg_set_incidence_matrix], axis=1)
        # 这里test_neg_set被重新组织了，所以需要重新计算一下
        test_pos_set = np.concatenate([test_pos_set, full_posHyperlink], axis=1)
        test_set = np.concatenate([test_pos_set, test_neg_set], axis=1)
        test_labels = np.concatenate([np.ones(test_pos_set.shape[1]), np.zeros(test_neg_set.shape[1])])

        train_pos_index = np.where(train_labels == 1)[0]
        writer_tb_log_dir='logsAndResults/logs/original_NSplited_node_LLM_Linear_Structure/'

        '''BAAI-bge'''
        pro = processer_(embedding_type=LM_model_selected, repoName=repoName, artifacts=artifacts,artifact_dict=artifact_dict, freeze=freeze, with_knowledge=with_knowledge, cat=cat,
                        tokenizer=artifacts.tokenizer_NL, device=device, embedding_model=embedding_model,writer_tb_log_dir=writer_tb_log_dir)
        # 这里转换一下数据，适配语义为基础的训练过程
        # 创建 TensorBoard 的 SummaryWriter
        train_hyperlink = Utils.getValidIndexFromList(train_set, artifact_dict)
        result = pro.train((train_hyperlink, train_labels))

        test_hyperlink = Utils.getValidIndexFromList(test_set, artifact_dict)
        result = pro.test((test_hyperlink, test_labels), checkpoint_path=None)
        print(result)
        results.append(result)

    # ==============Save the results================
    output_dir = f"logsAndResults/saved_results/{running_type}/"
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/{repoName}_results_{str(freeze)}_{str(with_knowledge)}_{str(cat)}.csv", index=False)
    print(df)
    print(f"The {repoName} results have been saved successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The arguments set for GraphLinker')
    # 添加参数及其简写
    parser.add_argument('-r', '--repoName', type=str, default="Activiti", help='The repository name in the dataset. This is a required parameter.')
    parser.add_argument('-c', '--cudaN', type=int, default=1, help='The GPU number to use. Default is 0.')
    parser.add_argument('-n', '--num_folds', type=int, default=5, help='The number of folds for cross-validation. Default is 5.')
    parser.add_argument('-t', '--test_ratio', type=float, default=0.2, help='The ratio of the test set. Default is 0.2.')
    parser.add_argument('-type', '--running_type', type=str, default="semantic", help='The running choice for the whole project. Default is structure.')
    # 增加三个对比实验参数，冻结-非冻结，自带知识-无自带知识，cat-非cat
    # 增加三个对比实验参数，初始值设置为True
    parser.add_argument('--freeze', type=str, default="false", help='Frozening The LLM when Training or not.')
    parser.add_argument('--with_knowledge', type=str, default="true", help='Take the prior-knowledge into training or not.')
    parser.add_argument('--cat', type=str, default="true", help='Using the cat module for the input information or not.')

    args = parser.parse_args()
    repoName = args.repoName
    device = Utils.get_cuda_device(0)
    root_path = "../dataset/hyperlink_npz"
    repopath = root_path + "/"+ repoName + ".npz"
    num_folds = args.num_folds
    test_ratio = args.test_ratio
    running_type = args.running_type

    freeze = args.freeze
    with_knowledge = args.with_knowledge
    cat = args.cat
    print(freeze,with_knowledge,cat)
    main(root_Repo=repopath, device=device)