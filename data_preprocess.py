# 本节的主要目的是为了统筹整个项目的流程，将各个模块串联起来，实现一个完整的预测过程

'''
    导包区
'''
import argparse
import os
import random
import numpy as np
from networkx.linalg.graphmatrix import adjacency_matrix
from scipy.io import savemat
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from utils import Utils
from DataStructure._1_HyperlinkGenerate import generateHyperLinkDataset
from processerHook import processer
from model.models.CMM.CMM_HyperLink import cmm

np.random.seed(43)
random.seed(43)

# 这部分代码片段用于将train+test Artifacts转化成all Artifacts
'''
    test_artifacts = getDataSet(root_Repo, 'test')
    all = train_artifacts + test_artifacts
    artifact_set = {}
    for artifact in all:
        artifactId = artifact.artifact_id
        artifact_set[artifactId] = artifact
    # 将这部分的内容保持到文件中
    utils.save_pickle(artifact_set, f"{root_Repo}/processed/all_artifacts.pkl")
'''
def main(root_Repo:str, device:int):
    # 读取数据
    all_artifacts = utils.getNodes()
    pos_hyperlink = utils.posHyperLink

    gH = generateHyperLinkDataset(all_artifacts, pos_hyperlink, utils=utils)
    posHyperlink, negHyperlink, negative_adjacents = gH.generateHyperLink(ratio=1)
    positive_adjacents = utils.get_adjacency_matrix()
    print((negative_adjacents==positive_adjacents).all())
    # 创建 Model 结构体
    Model = {
        'S': posHyperlink,
        'U': negHyperlink
    }
    savemat(f'dataset/mat/{repoName}.mat', {'Model': Model})
    np.savez(f"dataset/hyperlink_npz/{repoName}_hyperlink", positive=posHyperlink, negative=negHyperlink)
    np.savez(f"dataset/adjacent_npz/{repoName}", positive=positive_adjacents, negative=negative_adjacents)

    # 创建一个字典来存储数据
    # num_folds = 10
    # kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    # results = []
    # for train_index, test_index in kf.split(posHyperlink.T):  # Transpose to get samples as rows
    #     # Generate train data and labels
    #     train_pos_set, train_neg_set = posHyperlink[:, train_index], negHyperlink[:, train_index]
    #     train_set = np.concatenate([train_pos_set, train_neg_set], axis=1)
    #     train_labels = np.concatenate([np.ones(train_pos_set.shape[1]), np.zeros(train_neg_set.shape[1])])
    #
    #     # Generate test data and labels
    #     test_pos_set, test_neg_set = posHyperlink[:, test_index], negHyperlink[:, test_index]
    #     test_set = np.concatenate([test_pos_set, test_neg_set], axis=1)
    #     test_labels = np.concatenate([np.ones(test_pos_set.shape[1]), np.zeros(test_neg_set.shape[1])])
    #
    #     train_pos_index = np.where(train_labels == 1)[0]
    #     pos_set = train_set[:, train_pos_index]
    #
    #     # ==============Training and Testing================
    #     pos_set_sparse,test_data_sparse = csr_matrix(pos_set), csr_matrix(test_set)
    #     num_prediction = len(test_labels)//2
    #     Lambda, scores_auc = cmm(pos_set_sparse, test_data_sparse, num_prediction, "cv", test_labels)
    #     results.append(scores_auc)
    #     print(f"Train set: {len(train_set)}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='The arguments set for GraphLinker')
    args.add_argument('--repoName', type=str, default='ansible', help='The repoName in the dataset')
    args.add_argument('--cudaN', type=int, default=0, help='The num specified for specifying gpu.')
    args = args.parse_args()
    repoName = args.repoName
    device = Utils.get_cuda_device(args.cudaN)
    repoNameList = ['angular', 'ansible', 'elasticsearch','symfony']
    for repoName in repoNameList:
        rootRepo = f'dataset/{repoName}'
        utils = Utils(repoName)

        main(root_Repo=rootRepo, device=device)
