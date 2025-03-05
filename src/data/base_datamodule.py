from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional


class BaseDatamodule(LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.dataset
        self.val_dataset = self.dataset
        self.test_dataset = self.dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                            persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)