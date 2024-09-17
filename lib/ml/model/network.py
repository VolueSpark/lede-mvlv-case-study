from tqdm.notebook import trange, tqdm
from torch import nn
import polars as pl
import os
import torch
import torch.optim as optim





class NeuralNetwork(nn.Module):
    def __init__(self, work_dir: str, params: dict):
        super().__init__()
        self.device = device
        self.work_dir = work_dir
        self.params = params

    def set_data(self, train_loader: DataLoader, test_loader: DataLoader, val_loader: DataLoader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader


    def set_optimizer(self):
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

    def train(self):

        for i, data in enumerate(tqdm(self.train_dataloader)):
            x, x_exo, y = data.values()

            self.optimizer.zero_grad()

    def predict(self, x, x_exo=None):
        pass