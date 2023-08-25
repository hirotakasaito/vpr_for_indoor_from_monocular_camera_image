import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch
import json
from glob import iglob
import os

class Dataset(pl.LightningDataModule):

    def __init__(self, dataset_dir):
        
        data_path = []
        for data in iglob(os.path.join(dataset_dir, "*", "*.pt")):
            data_path.append(data)

        self.data_path = data_path
       

    def __len__(self):

        return len(self.data_path)

    def __getitem__(self, index):
        data = torch.load(self.data_path[index])
        return data["anchor_imgs"]/255, data["positive_imgs"]/255, data["negative_imgs"]/255


 
