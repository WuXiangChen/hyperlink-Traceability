# 本节的主要目的是 避开 transformer.trainer的执行流程
## 该框架虽然完整 但是太耗时间
import json
import os
import numpy as np
import torch
from model.models.trainer_hook import ModelFineTuner
from model.models.CHESHIRE.BAAI_Model_splitGraph import BAAI_model
from alive_progress import alive_bar

from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from alive_progress import alive_bar
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy,ConfusionMatrix
from processerHook.earlyStopping import EarlyStopping
from utils import Utils

class processer_:
    def __init__(self, repoName, artifacts, device, embedding_type, netModelPath, embedding_model, tokenizer, artifact_dict):

        config_path = "config_parameters.json"
        with open(config_path, 'r') as file:
            config = json.load(file)
        #model_name = config.get("model_name", "default_model_name")
        #evaluation_strategy = config.get("evaluation_strategy", "steps")
        #eval_steps = config.get("eval_steps", 500)
        #logging_dir = config.get("logging_dir", "/path/to/logs")
        #seed = config.get("seed", 42)
        self.lr = config.get("learning_rate", 5e-5)
        self.batch_size = config.get("batch_size", 16)
        self.num_train_epochs = config.get("num_train_epochs", 20)
        self.weight_decay = config.get("weight_decay", 9E-6)
        self.patience = config.get("patience", 8)

        self.repoName = repoName
        self.device = device
        self.artifacts = artifacts
        self.artifact_dict = artifact_dict

        if embedding_type=="BAAI_bge-m3_small":
            in_dim = 1024
        elif embedding_type=="all-MiniLM-L6-v2":
            in_dim = 384
        else:
            in_dim = 512

        self.early_stopping = EarlyStopping(patience=self.patience, verbose=True, save_path=netModelPath)
        self.model = BAAI_model(artifacts_dict=self.artifact_dict, artifacts=artifacts, model=embedding_model, tokenizer=tokenizer, in_dim=in_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.accuracy_metric = BinaryAccuracy()
        self.confmat = ConfusionMatrix(num_classes=2, task="binary")

    def train(self, train_hyperlink, train_labels, epochs=20, writer=None):
        padded_train_data = Utils.pad_list(list(train_hyperlink.values()))
        train_hyperlink, val_hyperlink, train_labels, val_labels = train_test_split(padded_train_data, train_labels, test_size=0.2, random_state=43)
        with alive_bar(epochs) as bar:
            for epoch in range(epochs):
                self.model.train()
                t_hyperlink = torch.tensor(train_hyperlink).to(self.device)
                t_labels = torch.tensor(train_labels).to(self.device)
                loss, _ = self.model(t_hyperlink, t_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                bar()
                if epoch % 2 == 0 and epoch!=0:  # 每100个小批量输出一次
                    ### 进行验证 ###
                    _, val_loss = self.test(val_hyperlink, val_labels)
                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        print('Early stopping!')
                        break
                    writer.add_scalar('training loss', loss )
                    writer.add_scalar('validating loss', val_loss)

    def test(self, test_hyperlink, test_labels, test_flag=False, checkpointPath=None):
        if test_flag:
           test_hyperlink = Utils.pad_list(list(test_hyperlink.values()))
        t_hyperlink = torch.tensor(test_hyperlink).to(self.device)
        t_labels = torch.tensor(test_labels).to(self.device)
        if checkpointPath!=None:
            # 加载这部分的内容，并将其作为当前的测试模型
            self.model = self.load_checkpoint(checkpointPath)
        self.model.eval()  # Set the model to evaluation mode
        #with torch.no_grad():  # Disable gradient calculation
        val_loss, output = self.model(t_hyperlink, t_labels)  # Forward pass
        re = self.compute_metrics((output.cpu(),t_labels.cpu()))
        re["loss"] = val_loss
        print("re:", re)
        return re, val_loss   # Return loss and accuracy

    def load_checkpoint(self, checkfolder):
            """Loads model from a checkpoint file."""
            if checkfolder is not None:
                if os.path.exists(checkfolder):
                    files = os.listdir(checkfolder)
                    checkpoint_path = os.path.join(checkfolder, files[0])
                    model = torch.load(checkpoint_path)
                    return model
                else:
                    print(f'Checkpoint file {checkfolder} does not exist.')
                    return None
            else:
                print('No checkpoint path provided.')
                return None

    def compute_metrics(self, p):
        preds, labels = p
        labels = torch.tensor(labels).int()
        
        # 将预测值转换为二进制
        preds_binary = (preds > 0.5).int().detach()
        
        # 计算 AUC
        auc = roc_auc_score(labels.numpy(), preds_binary.numpy())
        # 计算 Recall 和 Precision 仅针对 label 1
        precision = precision_score(labels.numpy(), preds_binary.numpy(), pos_label=1)
        recall = recall_score(labels.numpy(), preds_binary.numpy(), pos_label=1)
        
        acc = self.accuracy_metric(preds_binary, labels)
        cm = self.confmat(preds_binary, labels)
        
        return {
            "acc": acc,
            "confusion_matrix": cm.tolist(),
            "auc": auc,
            "precision": precision,
            "recall": recall
        }