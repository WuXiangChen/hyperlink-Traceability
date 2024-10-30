# 本节的主要目的是控制训练和测试过程
import numpy as np
import torch
from xgboost.dask import train

from model.models.CHESHIRE.BAAI_Model import BAAI_model
from model.models.trainer_hook import ModelFineTuner
from model.models.CHESHIRE.CHESHIRE import CHESHIRE

class processer:
    def __init__(self, pos_set, repoName, artifacts,artifact_dict, tokenizer, device, embedding_model):
        self.repoName = repoName
        self.artifacts_dict = artifact_dict
        self.model = BAAI_model(artifacts_dict=artifact_dict, artifacts=artifacts, model=embedding_model, tokenizer=tokenizer)
        self.trainer = ModelFineTuner(self.model, device, num_labels=2)
        self.artifacts = artifacts

    def train(self, data):
        self.trainer.loadData(data, type="train")
        self.trainer.set_training_args(self.repoName)
        self.model.train() # 设置为训练模式
        self.trainer.train()

    def test(self, data, checkpoint_path):
        self.trainer.loadData(data, type="test")
        self.trainer.set_training_args(self.repoName)
        self.model.eval() # 设置为测试模式
        return self.trainer.evaluate(checkpoint_path=checkpoint_path)