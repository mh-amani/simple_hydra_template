from lightning import LightningDataModule
import hydra
from torch.utils.data import DataLoader
from typing import Optional


class BaseDatamodule(LightningDataModule):
    def __init__(self, dataset_config, batch_size: int, num_workers: int):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = hydra.utils.instantiate(self.dataset_config)
        self.val_dataset = hydra.utils.instantiate(self.dataset_config)
        self.test_dataset = hydra.utils.instantiate(self.dataset_config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                            persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)