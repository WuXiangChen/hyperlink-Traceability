# 本节的主要目的是控制训练和测试过程
import numpy as np
import torch
from model.models.trainer_hook import ModelFineTuner
from model.models.CHESHIRE.BAAI_Model import BAAI_model
class processer_:
    def __init__(self, embedding_type, repoName, artifacts, tokenizer, device, embedding_model, writer_tb_log_dir, freeze, with_knowledge, cat):
        self.repoName = repoName
        if embedding_type=="BAAI_bge-m3_small":
            in_dim = 1024
        elif embedding_type=="all-MiniLM-L6-v2":
            in_dim = 384
        else:
            in_dim = 512
        
        self.model = BAAI_model(artifacts=artifacts, model=embedding_model, tokenizer=tokenizer,in_dim=in_dim, freeze=freeze, with_knowledge=with_knowledge)
        self.trainer = ModelFineTuner(self.model, device, num_labels=2, writer_tb_log_dir=writer_tb_log_dir)
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