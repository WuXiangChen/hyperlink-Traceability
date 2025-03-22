# 本节的主要目的是控制训练和测试过程
from ast import Add
import numpy as np
import torch
from GraphLinker.models.trainer_hook import ModelFineTuner
from GraphLinker.models.P2PModule.P2PModule import run_process_for_p2pMoudle
from utils import Utils

class processer_:
    def __init__(self, embedding_type, repoName, artifacts, tokenizer, device, embedding_model, writer_tb_log_dir, training_type, training_with_gnn, max_length, epoch_size_set, test_k_index, hp_hiddenDList, lopo,hp=False):
        self.repoName = repoName
        if embedding_type=="BAAI_bge-m3_small":
            in_dim = 512
            latent_dim = 512
        elif embedding_type=="all-MiniLM-L6-v2":
            in_dim = 384
            latent_dim = 384
        else:
            in_dim = 512
            latent_dim = 512
        self.k = test_k_index
        self.training_type = training_type
        from GraphLinker.models.HPModule.BAAI_Model import BAAI_model
        self.model = BAAI_model(artifacts=artifacts, model=embedding_model, tokenizer=tokenizer,in_dim=in_dim, 
                                training_type=training_type, max_length=max_length, 
                                latent_dim=latent_dim, test_k_index=test_k_index,hp_hiddenDList=hp_hiddenDList, hp=hp)
        self.p2pModelProcess = run_process_for_p2pMoudle
        self.trainer = ModelFineTuner(self.model, device, num_labels=2, writer_tb_log_dir=writer_tb_log_dir,lopo=lopo)
        self.artifacts = artifacts
        self.epoch_size_set = epoch_size_set
        self.P2Part_adj = None

    def train(self, all_train, all_TrainLabels, writer=None, P2Pdatasets=None, P2PLabels=None, k=1):
        data = (all_train, all_TrainLabels)
        self.trainer.loadData(data, type="train")
        self.trainer.set_training_args(self.repoName, epochs=self.epoch_size_set, k=k, training_type=self.training_type)
        self.model.setP2PInfo(self.p2pModelProcess, P2Pdatasets, P2PLabels, reponame=self.repoName, k=self.k)
        self.model.train() # 设置为训练模式
        self.trainer.train()
        self.P2Part_adj = self.model.getArt_adj()

    def prepare_and_evaluate(self, data, checkpoint_path):
        """
        Helper function to prepare data, configure training arguments, and evaluate the model.

        Parameters:
            data: Tuple containing datasets and labels.
            checkpoint_path: Path to the checkpoint file for evaluation.

        Returns:
            result: Evaluation result after model inference.
        """
        self.trainer.loadData(data, type="test")
        self.trainer.set_training_args(self.repoName)
        self.model.eval()  # Set model to evaluation mode
        return self.trainer.evaluate(checkpoint_path=checkpoint_path)

    def test_HP_HP(self, test_repo_index, test_repo_label, test_flag=True, checkpointPath=None, gTG=None):
        """
        Test function that evaluates the model on the test dataset and optionally performs link prediction.
        
        Parameters:
            test_repo_index: Index data for the test repository.
            test_repo_label: Label data for the test repository.
            test_flag: Boolean indicating whether to run the initial test phase.
            checkpointPath: Path to the checkpoint file for evaluation.
            gTG: Graph structure object with edge information for link prediction.

        Returns:
            HPResult: Evaluation results for the primary test phase.
            LinkResult: Evaluation results for the link prediction phase (if applicable).
        """
        # Phase 1: Standard test phase
        test_data = (test_repo_index, test_repo_label)
        HPResult, preds, label_ids_ = self.prepare_and_evaluate(test_data, checkpointPath)

        if not test_flag or gTG is None:
            # Skip the link prediction phase if test_flag is False or gTG is None
            return HPResult, None
        
        pre_true_index = np.where(preds > 0.4)[0]
        missed_HP = np.setdiff1d(np.where(label_ids_ > 0.5)[0], pre_true_index)
        mP2PDatasets, _ = test_repo_index[missed_HP], test_repo_label[missed_HP]
        _, mP2PLabels = Utils.P2P_Expanded(mP2PDatasets, gTG.getEdges(), addTwoComORNot=True)
        mP2PLabels = np.array(mP2PLabels)
        mNum = len(mP2PLabels[mP2PLabels==1])
        print("Missed Num:", mNum)

        P2PDatasets, _ = test_repo_index[pre_true_index], test_repo_label[pre_true_index]
        P2PDatasets, P2PLabels = Utils.P2P_Expanded(P2PDatasets, gTG.getEdges(), addTwoComORNot=True, testFlag=True)
        if P2PDatasets==None:
            return HPResult, None
        P2PDatasets_array = np.array(P2PDatasets)
        P2PDatasets = np.hstack([P2PDatasets_array, np.zeros((P2PDatasets_array.shape[0], 1))]).astype(int)
        P2PLabels = np.array(P2PLabels, dtype=float)
        P2PData = (P2PDatasets, P2PLabels)
        internalLinkResult, _, _ = self.prepare_and_evaluate(P2PData, checkpointPath)

        internal_confMatrix = np.array(internalLinkResult["test_confusion_matrix"])
        internal_confMatrix[1,0]+=mNum

        externalLinkResult = Utils.calculate_metrics(internal_confMatrix)
        All_Re = {"HP_Pre":HPResult, "interiorLinkResult":internalLinkResult, "externalLinkResult":externalLinkResult}
        return All_Re

    def test_HP_P2P(self, test_repo_index, test_repo_label, test_flag=True, checkpointPath=None, gTG=None):
        """
        Test function that evaluates the model on the test dataset and optionally performs link prediction.
        
        Parameters:
            test_repo_index: Index data for the test repository.
            test_repo_label: Label data for the test repository.
            test_flag: Boolean indicating whether to run the initial test phase.
            checkpointPath: Path to the checkpoint file for evaluation.
            gTG: Graph structure object with edge information for link prediction.

        Returns:
            HPResult: Evaluation results for the primary test phase.
            LinkResult: Evaluation results for the link prediction phase (if applicable).
        """
        # Phase 1: Standard test phase
        test_data = (test_repo_index, test_repo_label)
        HPResult, preds, label_ids_ = self.prepare_and_evaluate(test_data, checkpointPath)

        if not test_flag or gTG is None:
            return HPResult, None
        
        # 这里是验证missing true有多少
        pre_true_index = np.where(preds > 0.5)
        missed_HP = np.setdiff1d(np.where(label_ids_ > 0.5)[0], pre_true_index)
        mP2PDatasets = test_repo_index[missed_HP]
        _, mP2PLabels = Utils.P2P_Expanded(mP2PDatasets, gTG.getEdges(), addTwoComORNot=True, testFlag=True)
        mP2PLabels = np.array(mP2PLabels)
        mNum = len(mP2PLabels[mP2PLabels==1])
        
        pre_false_index = np.where(preds < 0.9)[0]
        mP2PDatasets = test_repo_index[pre_false_index]
        _, mP2PLabels = Utils.P2P_Expanded(mP2PDatasets, gTG.getEdges(), addTwoComORNot=True, testFlag=True)
        mP2PLabels = np.array(mP2PLabels)
        cNum = len(mP2PLabels[mP2PLabels==0])
        print("Missing Links:", mNum)
        print("Correct Remove Links:", cNum)

        P2Pdatasets = test_repo_index[pre_true_index]
        P2Pdatasets, P2PLabels = Utils.P2P_Expanded(P2Pdatasets, gTG.getEdges(), addTwoComORNot=False, testFlag=True)
        if not isinstance(P2Pdatasets, list):
            P2Pdatasets = list(P2Pdatasets.values())
        if P2Pdatasets==None:
            return HPResult, None
        
        if self.P2Part_adj==None:
            raise "测试时， P2Part_adj为None"
        # 在这里释放model
        del self.model
        import gc
        gc.collect()
        
        internalLinkResult = self.p2pModelProcess(P2Pdatasets, P2PLabels, testFlag=True, reponame=self.repoName, art_adj=self.P2Part_adj,k=self.k)

        internal_confMatrix = np.array(internalLinkResult)
        internal_confMatrix[1,0]+=mNum
        internal_confMatrix[0,0]+=cNum

        externalLinkResult = Utils.calculate_metrics(internal_confMatrix)
        print("P2P Matrix:")
        print(internal_confMatrix)
        All_Re = {**HPResult, **externalLinkResult}
        return All_Re

    def test(self, test_repo_index, test_repo_label, test_flag=True, checkpointPath=None, gTG=None):
        """
        Test function that evaluates the model on the test dataset and optionally performs link prediction.
        
        Parameters:
            test_repo_index: Index data for the test repository.
            test_repo_label: Label data for the test repository.
            test_flag: Boolean indicating whether to run the initial test phase.
            checkpointPath: Path to the checkpoint file for evaluation.
            gTG: Graph structure object with edge information for link prediction.

        Returns:
            HPResult: Evaluation results for the primary test phase.
            LinkResult: Evaluation results for the link prediction phase (if applicable).
        """
        P2PDatasets, _ = test_repo_index, test_repo_label
        P2PDatasets, P2PLabels = Utils.P2P_Expanded(P2PDatasets, gTG.getEdges(), addTwoComORNot=True)
        P2PDatasets_array = np.array(list(P2PDatasets.values()))
        P2PDatasets = np.hstack([P2PDatasets_array, np.zeros((P2PDatasets_array.shape[0], 1))]).astype(int)
        P2PLabels = np.array(P2PLabels, dtype=float)
        P2PData = (P2PDatasets, P2PLabels)
        externalLinkResult, _, _ = self.prepare_and_evaluate(P2PData, checkpointPath)

        All_Re = {"externalLinkResult":externalLinkResult}
        return All_Re

    def test_HP(self, test_repo_index, test_repo_label, test_flag=True, checkpointPath=None, gTG=None):
        """
        Test function that evaluates the model on the test dataset and optionally performs link prediction.
        
        Parameters:
            test_repo_index: Index data for the test repository.
            test_repo_label: Label data for the test repository.
            test_flag: Boolean indicating whether to run the initial test phase.
            checkpointPath: Path to the checkpoint file for evaluation.
            gTG: Graph structure object with edge information for link prediction.

        Returns:
            HPResult: Evaluation results for the primary test phase.
            LinkResult: Evaluation results for the link prediction phase (if applicable).
        """
        # Phase 1: Standard test phase
        test_data = (test_repo_index, test_repo_label)
        HPResult, preds, label_ids_ = self.prepare_and_evaluate(test_data, checkpointPath)

        if not test_flag or gTG is None:
            return HPResult, None
        
        return HPResult