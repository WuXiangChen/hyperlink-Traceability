import numpy as np
import torch
import torch.nn as nn

class OptimizedAllocLossCalculator:
    def __init__(self, V_no, lambda_reg=0.01, class_weights=None, epsilon=1e+2):
        self.V_no = V_no  # 节点数
        self.lambda_reg = lambda_reg  # 正则化强度
        self.epsilon = epsilon  # 平滑常数
        self.criterion = nn.BCEWithLogitsLoss()
    def compute_loss(self, score, gt_artifact):
        """计算优化后的损失 L_alloc"""
        # 初始化损失
        loss = self.criterion(score, gt_artifact)
        loss = loss.sum()
        # 添加正则化项
        loss += self.lambda_reg * torch.sum(score ** 2)
        return loss


    def compute_CrossEntoloss(self, score, gt_artifact):
        """计算优化后的损失 L_alloc"""
        batch_size = gt_artifact.size(0)  # 假设 gt_artifact 的行数是 batch_size

        # 平滑处理
        gt_artifact = (gt_artifact + self.epsilon) / (1 + self.epsilon * batch_size)
        # 计算类别权重
        weight = self.compute_class_weights(gt_artifact)  # 计算每个样本的类别权重
        # 将 one-hot 编码转换为类索引
        # target = torch.argmax(gt_artifact, dim=1)  # 获取每个样本的目标类索引
        # score = torch.argmax(score, dim=1)  # 获取每个样本的预测类索引
        # 创建交叉熵损失函数
        criterion = nn.CrossEntropyLoss(weight=weight[0,:])
        # 计算损失
        loss = criterion(score, gt_artifact)
        # 添加正则化项
        loss += self.lambda_reg * torch.sum(score ** 2)
        return loss


    def compute_class_weights(self, gt_artifact):
        """计算每个样本的类别权重"""
        batch_size = gt_artifact.size(0)
        num_classes = gt_artifact.size(1)
        # 初始化权重
        weight = torch.zeros_like(gt_artifact)
        for k in range(batch_size):
            positive_count = torch.sum(gt_artifact[k, :])  # 计算每个类别的正类样本数量
            negative_count = num_classes - positive_count  # 计算负类样本数量

            if positive_count > 0:
                positive_weight = negative_count / positive_count
            else:
                positive_weight = 1  # 如果没有正类样本，则权重为1
            negative_weight = positive_count / negative_count if negative_count > 0 else 1
            # 根据 gt_artifact 生成权重
            weight[k, :] = torch.where(gt_artifact[k, :] == 1, positive_weight, negative_weight)
        return weight


if __name__ == '__main__':
    # 测试
    V_no = [1, 2, 3]  # 示例节点
    class_weights = np.array([1.0, 2.0, 3.0])  # 示例类别权重
    calculator = OptimizedAllocLossCalculator(V_no, class_weights=class_weights)

    # 假设 score 和 gt_artifact 是 numpy 数组
    score = np.array([[0.1113, 0.1112, 0.1110, 0.1112, 0.1110, 0.1111, 0.1112, 0.1110, 0.1112]])
    gt_artifact = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    # 转换为 PyTorch 张量
    score_tensor = torch.tensor(score)
    gt_artifact_tensor = torch.tensor(gt_artifact)

    loss = calculator.compute_loss(score_tensor, gt_artifact_tensor)
    print("计算得到的优化损失:", loss)
