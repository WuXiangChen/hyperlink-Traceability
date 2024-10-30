# 生成数据集
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class ArtifactDataset(Dataset):
    def __init__(self, artifacts, ground_truth):
        self.artifacts = artifacts
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.artifacts)

    def __getitem__(self, idx):
        artifact = numpy.array(self.artifacts[idx])
        gt = numpy.array(self.ground_truth[idx])
        # 将列表转换为张量
        artifact_tensor = torch.tensor(artifact, dtype=torch.float)
        gt_tensor = torch.tensor(gt, dtype=torch.float)

        return artifact_tensor, gt_tensor