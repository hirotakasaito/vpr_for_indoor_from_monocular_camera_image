import argparse
from typing import Any, Optional, Type
import os
import json
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, dataset_dir, test_dataset_dir, seed,  batch_size = 8, num_worker=4):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataset_dir = dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self._seed = seed

    def setup(self, stage):
        dataset = self.dataset(self.dataset_dir)
        self.train_dataset, self.valid_dataset = random_split(dataset, [int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))], generator=torch.Generator().manual_seed(self._seed))

        if stage == "test":
            self.test_dataset = self.dataset(self.test_dataset_dir)

    def train_dataloader(self,):
        return DataLoader(
                self.train_dataset,
                batch_size = 8,#self.batch_size,
                shuffle = True,
                drop_last = True,
                num_workers = self.num_worker)

    def val_dataloader(self,):
        return DataLoader(
                self.valid_dataset,
                batch_size = 8,#self.batch_size,
                shuffle = False,
                drop_last = True,
                num_workers = self.num_worker)

    def test_dataloader(self,):
        return DataLoader(
                self.test_dataset,
                batch_size = 1,
                shuffle = False,
                drop_last = True,
                num_workers = self.num_worker)
 
