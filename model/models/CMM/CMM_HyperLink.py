import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import scipy
import tqdm
from numpy import hstack, vstack
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import KFold
from scipy.optimize import lsq_linear
from sklearn.metrics import roc_auc_score

def cmm(hltrain, hltest, num_prediction, k, test_labels):
    """
    主程序：协调矩阵最小化（CMM）算法，用于超链接预测。
    参数：
    hltrain: 观察到的超链接列（训练超链接）
    hltest: 候选超链接列（测试超链接）
    num_prediction: 要预测的超链接数量
    k: 用于非负矩阵分解的潜在因子数量
    test_labels: 测试超链接的真实标签
    返回：
    Lambda: 逻辑列指示向量（1 表示 hltest 中的第 i 列被选中）
    scores: 包含 hltest 中列的预测分数的列向量
    """
    # 如果 k 设置为 'cv'，则进行交叉验证以选择最佳 k
    if k == 'cv':
        if hltrain.shape[0] > 1000:  # 如果超链接网络较大，设置 k 为默认值 30
            k = 30
        else:
            K = [10, 20, 30]  # 潜在因子的候选值
            folds = 2  # 使用的折数
            kf = KFold(n_splits=folds)  # 创建 K 折交叉验证对象
            AUC = np.zeros((len(K), folds))  # 存储每个 k 的 AUC 值
            NMATCH = np.zeros((len(K), folds))  # 存储每个 k 的匹配数
            print('Cross validation begins...')  # 输出交叉验证开始的提示
            for i, (train_index, test_index) in enumerate(kf.split(hltrain.T)):
                hltrain1 = hltrain[:, train_index]  # 当前折的训练数据
                hltest1 = np.hstack((hltest, hltrain[:, test_index]))  # 当前折的测试数据
                test_labels1 = np.hstack((np.zeros(test_labels.shape), np.ones(test_index.shape[0])))  # 生成标签

                for j, k in enumerate(K):
                    nmatch, auc, _, _ = optimize(hltrain1, hltest1, num_prediction, k, test_labels1)  # 调用优化函数
                    AUC[j, i] = auc  # 存储 AUC 值
                    NMATCH[j, i] = nmatch  # 存储匹配数

            aAUC = np.mean(AUC, axis=1)  # 计算每个 k 的平均 AUC
            aNMATCH = np.mean(NMATCH, axis=1)  # 计算每个 k 的平均匹配数
            k = K[np.argmax(aNMATCH)]  # 使用匹配数最大值作为交叉验证标准

    # 调用优化函数进行超链接预测
    nmatch, auc, scores, Lambda = optimize(hltrain, hltest, num_prediction, k, test_labels)
    Lambda = Lambda.astype(bool)  # 将 Lambda 转换为布尔类型
    return Lambda, scores  # 返回结果

def process_column(i, hltest, pca):
    print(f'Processing column {i}...')
    u = hltest[:, i]  # 获取第 i 列
    u = u @ u.T  # 计算外积
    u = u.toarray()
    u = pca.fit_transform(u).flatten()
    return u

def process_columns_multithreaded(cc, hltest, max_workers=10):
    pca = PCA()  # 初始化 PCA
    U1 = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_column, i, hltest, pca) for i in range(cc)]
        for future in futures:
            U1.append(future.result())
    U1 = np.vstack(U1).T  # 转置 U1
    return U1

def symnmf_newton(A, k, params):
    # 这里需要实现或调用适当的 NMF 方法
    # 返回 W, H, res0
    pass  # 你需要根据具体情况实现这个函数


def optimize(hltrain, hltest, num_prediction, k, test_labels):
    """
    Alternate EM optimization between Lambda and W
    --Output--
    nmatch: number of true positive hyperlinks in the predictions
    """

    A = hltrain @ hltrain.T  # 观察到的邻接矩阵
    rr, cc = hltest.shape
    U1 = []  # vectorized uu^T 矩阵

    for i in range(cc):
        u = hltest[:, i]
        u = np.outer(u, u)  # 计算 uu^T
        U1.append(u.flatten())  # 将稀疏矩阵展平并添加到列表中

    U1 = np.array(U1).T  # 转换为列向量矩阵

    # 优化设置
    opts = {'max_iter': 100, 'disp': True}
    params = {'maxiter': 100, 'debug': 0}
    max_iter = 100  # 最大 EM 迭代次数
    res = 1e10  # 当前目标值

    scores = np.zeros(cc)  # 初始化分数
    Nmatch = []

    for iter in range(max_iter):  # 开始迭代，最多进行 max_iter 次
        old_res = res  # 保存当前的残差，以便后续比较

        # 更新矩阵 Ai，计算 A + UΛU^T
        Ai = A + (hltest @ np.diag(scores) @ hltest.T)

        # M 步骤：使用 symnmf_newton 函数进行 NMF 优化
        if iter == 0:  # 如果是第一次迭代
            W, _, res0 = symnmf_newton(Ai, k, params)  # 初始化 W
        else:  # 如果不是第一次迭代
            params['Hinit'] = W  # 将上一次迭代的 W 作为初始化
            W, _, res0 = symnmf_newton(Ai, k, params)  # 进行 NMF 优化

        WWT = W @ W.T  # 计算 W 的外积，得到 WWT

        # E 步骤：使用最小二乘法优化分数
        dA = WWT - A  # 计算目标差异
        res = lsq_linear(U1, dA.flatten(), bounds=(0, 1), **opts)  # 在 U1 上进行线性最小二乘法求解
        scores = res.x  # 更新分数

        # 当前迭代的结果
        print(f"Iteration: {iter + 1}, Residual: {res}")  # 显示当前迭代次数和残差
        Lambda = np.zeros(cc)  # 初始化 Lambda 向量
        I = np.argsort(scores)[::-1]  # 对分数进行降序排序
        Lambda[I[:num_prediction]] = 1  # 仅保留前 num_prediction 个高分的 hl
        nmatch = np.count_nonzero(Lambda * test_labels)  # 计算真实正例的匹配数量
        Nmatch.append(nmatch)  # 将当前匹配数量添加到 Nmatch 中

        # 提前停止条件
        if abs(res.cost - old_res) < 1e-4:  # 如果残差变化小于阈值
            break  # 提前结束迭代

    print(Nmatch)  # 输出所有迭代中的匹配数量
    auc = roc_auc_score(test_labels, scores)  # 计算 AUC（曲线下面积）指标
    return nmatch, auc, scores, Lambda
