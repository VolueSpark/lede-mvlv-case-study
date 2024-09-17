from tqdm.notebook import trange, tqdm
from torch import nn
import polars as pl
import os
import torch
import torch.optim as optim

from lib.ml.dataloader import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(
            self,
            work_dir: str,
            params: dict
    ):

        super().__init__()
        self.device = device
        self.work_dir = work_dir
        self.params = params

        #self.dtl = DataLoader(os.path.join(self.work_dir, 'data/gold'), params=params['dataloader'])

        self.train_loss_logger = []


    def set_optimizer(self):
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

    def train_model(self):

        if self.train_loader is None:
            ValueError("Dataset not defined!")

        self.train()
        for i, (x, y) in enumerate(tqdm(self.train_loader, leave=False, desc="Training")):
            pass

        x = self.forward(x.to(self.device))

        loss = torch.nn.HuberLoss(x, y.to(self.device))

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.train_loss_logger.append(loss.item())