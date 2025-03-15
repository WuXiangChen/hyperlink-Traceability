import os
import pandas as pd
import torch
from torch import nn
from utils import Utils
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from datasets import load_dataset, Features, Value, Sequence, Dataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from safetensors.torch import load_file


def compute_metrics(p):
    preds, labels = p
    # Calculate AUC
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = None
    # 将预测值转换为二进制
    preds_binary = np.where(preds > 0.5, 1, 0)
    # 计算 Precision 和 Recall，指定正样本标签为 1
    precision = precision_score(labels, preds_binary, pos_label=1, average="binary")
    recall = recall_score(labels, preds_binary, pos_label=1, average="binary")
    # 计算 F1 分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds_binary)
    acc = accuracy_score(labels, preds_binary)
    return {
        "acc": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }

class ModelFineTuner:
    def __init__(self, model,  device,  num_labels=2, writer_tb_log_dir=None,lopo=True):
        self.device = device  # 指定使用 GPU0
        self.model = model
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.writer_path = writer_tb_log_dir
        self.lopo=lopo
        #self.forzen_initialization()
        # 定义特征
        self.train_fea = Features({
            "labels":Value(dtype="int32"),  # 标签特征
            "feature": Sequence(Sequence(Value(dtype="int32"))),  # 保持为 int64
            "edges": Sequence(Sequence(Value(dtype="float32"))),  # 保持为 int64
        })

    def loadData(self, train_data_and_label, type="train"):
        train_data, labels = train_data_and_label
        dataset_ = Dataset.from_dict({"labels": labels,"edges": train_data})
        dataset_.set_format(type='torch', columns=['labels', 'edges'])
        if type == "train":
            split_datasets = dataset_.train_test_split(test_size=0.2, seed=42, shuffle=True)
            self.train_dataset = split_datasets['train']
            self.eval_dataset = split_datasets['test']
        elif type == "test":
            self.test_dataset = dataset_
        print(f"Data has been loaded successfully for {type}ing!")

    def initialize_classifier_weights(self, classifier):
        for name, param in classifier.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_normal_(param)  # 或者使用其他初始化方法
            elif "bias" in name:
                torch.nn.init.zeros_(param)

    def forzen_initialization(self):
        # 在这里添加您的初始化逻辑，例如设置权重、冻结层等
        for name, param in self.model.named_parameters():
            if "embeddings" in name:  # 假设我们要冻结 encoder 层
                param.requires_grad = False

    def set_training_args(self, repoName, learning_rate=2E-5, weight_decay=4E-5, epochs=80, batch_size=50, k=1, training_type=3):
        if training_type==3:
            if self.lopo==True:
                output_dir = f"../CHESHIRE/{repoName}/finetune/{str(k)}"
            else:
                output_dir = f"../CHESHIRE/{repoName}/finetune_F/{str(k)}"
        else:
            if self.lopo==True:
                output_dir = f"../CHESHIRE/{repoName}/finetune_{training_type}/{str(k)}"
            else:
                output_dir = f"../CHESHIRE/{repoName}/finetune_{training_type}_F/{str(k)}"
            
        if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",  # 在每个 epoch 结束时评估
            save_strategy="epoch",  # 在每个 epoch 结束时保存模型
            eval_steps=None,  # 不需要按步数评估
            save_steps=None,  # 不需要按步数保存
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay = weight_decay,
            load_best_model_at_end=True,  # 在训练结束时加载最佳模型
            metric_for_best_model="eval_f1",  # 或使用 "f1" 等其他指标
            greater_is_better=True,  # 如果使用 "f1"，则设置为 True
            save_total_limit=1,  # 只保存最近的 3 个模型
            dataloader_num_workers=30,
            dataloader_pin_memory=True,
            warmup_steps=10,
            report_to="tensorboard",  # 向 TensorBoard 报告指标
            logging_dir=self.writer_path,  # 指定日志目录
        )

    def train(self):
        print("Start training!")
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=100)]  # 设置早停的耐心值
        )
        trainer.train()

    def evaluate(self, checkpoint_path=None):
        print("Start testing!")
        if checkpoint_path is not None:
            self.model.load_state_dict(load_file(checkpoint_path+"/model.safetensors"), strict=False)
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=self.test_dataset,
            compute_metrics=compute_metrics,
        )
        # metrics = trainer.evaluate()
        preds, label_ids, metrics = trainer.predict(self.test_dataset)
        # 从metrics中去掉 eval_model_preparation_time  eval_runtime  eval_samples_per_second  eval_steps_per_second等关键字
        keys_to_remove = [
            "eval_model_preparation_time",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second"
        ]
        # 使用字典推导式创建新字典
        filtered_metrics = {k: v for k, v in metrics.items() if k not in keys_to_remove}
        return filtered_metrics, preds, label_ids

    def save_model(self, save_path="./fine_tuned_model"):
        self.model.save_pretrained(save_path)

    def load_model(self, save_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(save_path)
