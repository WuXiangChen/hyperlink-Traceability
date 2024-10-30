# 本节的主要目的是控制训练和测试过程
import numpy as np
import torch
from xgboost.dask import train

from model.models.trainer_hook import ModelFineTuner
from model.models.CHESHIRE.CHESHIRE import CHESHIRE

class processer_:
    def __init__(self, pos_set, repoName, artifacts,artifact_dict, config, running_type, device, embedding_model):
        self.repoName = repoName
        self.artifact_dict = artifact_dict
        embedding_model = embedding_model.bert.embeddings
        inverse_dict = {v: k for k, v in artifact_dict.items()}
        ft_list = [artifacts.getArtifactFeature(inverse_dict[i]) for i in range(len(inverse_dict))]
        self.node_ft = torch.stack(ft_list, dim=0).squeeze()
        node_semantic_feature = []
        for slice_ in self.node_ft:
            slice_ = slice_.unsqueeze(0)
            node_semantic_feature.append(embedding_model(input_ids=slice_))
        self.node_semantic_feature = torch.stack(node_semantic_feature, dim=0).squeeze()
        self.node_semantic_feature = self.node_semantic_feature.permute(0, 2, 1).detach()

        self.model = CHESHIRE(pos_set=pos_set, emb_dim=config.emb_dim,
                              artifacts=artifacts,node_semantic_feature=self.node_semantic_feature,
                              conv_dim=config.conv_dim, k=config.k, p=config.p,
                              running_type=running_type)
        self.trainer = ModelFineTuner(self.model, device, num_labels=2)
        self.artifacts = artifacts


    def train(self, data):
        self.trainer.loadData(data, type="train")
        self.trainer.set_training_args(self.repoName)
        self.trainer.train()

    def test(self, data, checkpoint_path):
        self.trainer.loadData(data, type="test")
        self.trainer.set_training_args(self.repoName)
        return self.trainer.evaluate(checkpoint_path=checkpoint_path)