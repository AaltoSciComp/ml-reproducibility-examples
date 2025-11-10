import torch

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import lightning as L


class MNISTDataModule(L.LightningDataModule):

    def __init__(self, data_dir="./data", batch_size=32):
        # In init-function you can set arguments like data paths
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage):
        # setup-function is used to specify the datasets
        if stage == "fit":
            self.train_dataset = datasets.MNIST(
                self.data_dir, train=True, download=True, transform=ToTensor()
            )
        if stage == "test":
            self.test_dataset = datasets.MNIST(
                self.data_dir, train=False, transform=ToTensor()
            )

    def train_dataloader(self):
        # train_dataloader specifies how to set up a training dataloader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        # test_dataloader specifies how to set up a test dataloader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)