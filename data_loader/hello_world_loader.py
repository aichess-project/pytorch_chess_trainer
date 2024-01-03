import logging
import torch
from torch.utils.data import Dataset
import os


class HW_Dataset(Dataset):

    def __init__(self):
        pass

    def init(self, dl_config, converter, step):
        if step == "train":
            size = 5000
        elif step == "valid":
            size = 500
        else:
            size = 50
        self.y = []
        self.x = []
        self.load_data(converter, size)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        logging.debug(f"GET ITEM y: {self.y[idx].shape} {self.y[idx].dtype} X: {self.x[idx].shape} {self.x[idx].dtype}")
        return self.x[idx], self.y[idx]

    def load_data(self, converter, size):
        X = 20 * torch.rand((size,4)) - 10
        error = (torch.rand((size)) - 0.5) / 10
        x1 = X[:,0]
        x2 = X[:,1]
        x3 = X[:,2]
        x4 = X[:,3]
        y = x1**4 + x2**3 + x3**2 + x4 + error
        self.y = y
        self.x = X
        logging.info(f"y: {self.y.shape} {self.y.dtype} X: {self.x.shape} {self.x.dtype}")