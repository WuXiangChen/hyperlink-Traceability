# 本节的主要目的是控制训练和测试过程
import numpy as np
import torch
from xgboost.dask import train
from model.models.trainer_hook import ModelFineTuner
from model.models.CHESHIRE.AIPL_Better import MAGNN_ctr_ntype_specific

class processer_:
    def __init__(self, repoName, artifacts, artifact_dict, device):
        self.repoName = repoName
        self.artifact_dict = artifact_dict
        self.model = MAGNN_ctr_ntype_specific(in_dim=50, etypes_list=[0], out_dim=50, num_heads=1, attn_vec_dim=128,
                                              rnn_type='RotatE0',attn_drop=0.5, deepSchema=True , artifacts=artifacts)
        self.trainer = ModelFineTuner(self.model, device, num_labels=1)
        self.artifacts = artifacts


    def train(self, data):
        self.trainer.loadData(data, type="train")
        self.trainer.set_training_args(self.repoName)
        self.trainer.train()

    def test(self, data, checkpoint_path):
        self.trainer.loadData(data, type="test")
        self.trainer.set_training_args(self.repoName)
        return self.trainer.evaluate(checkpoint_path=checkpoint_path)