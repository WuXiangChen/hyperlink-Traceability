from datetime import datetime

import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0,  save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, val_f1, model):
        score = val_f1
        if score<0.01:
            self.early_stop = True
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score <= self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def remove_old_checkpoints(self):
        """Removes all old model checkpoint files in the save_path directory."""
        if os.path.exists(self.save_path):
            for filename in os.listdir(self.save_path):
                file_path = os.path.join(self.save_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.pt'):
                    os.remove(file_path)

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        self.remove_old_checkpoints()
        if self.verbose:
            print(f'Validation F1 increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        savePath = self.save_path + f"/checkpoint_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
        torch.save(model, savePath)
        self.val_loss_min = val_loss