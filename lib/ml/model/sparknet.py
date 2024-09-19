
from torch import nn
import polars as pl
import os, torch
from tqdm import tqdm

from typing import List, Tuple

from lib import logger
from lib.ml.dataloader import DataLoader
from lib.ml.layer.casualresidual import CasualResidual

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class SparkNet(nn.Module):
    def __init__(
            self,
            work_dir: str,
            train_loader: DataLoader,
            test_loader: DataLoader,
            val_loader: DataLoader,
            params: dict
    ):
        super().__init__()

        logger.info(f"Instantiate deep neural network model {self.__class__.__name__} with work directory {work_dir} using device {device}")

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        sample_data = next(iter(train_loader))

        self.device = device
        self.work_dir = work_dir
        self.params = params

        self.casual_residual_x = CasualResidual(input_shape=tuple(sample_data['x'].shape))
        self.casual_residual_x_exo = CasualResidual(input_shape=tuple(sample_data['x_exo'].shape))

        self.loss_function = nn.HuberLoss(reduction='max')

    def forward(self, x: torch.Tensor, x_exo: torch.Tensor)->torch.Tensor:
        x = self.casual_residual_x(x)
        # TODO add a conv1d layer to reduce channels from 40 to 10 and maxpooling to reduce sequence from 48 to 24. Then contcat output to x_exo and linear layer, reshape 10 x 24 and then loss
        x_exo = self.casual_residual_x_exo(x_exo)
        return x

    def train_model(self):
        self.train()
        for epoch_i in tqdm(range(self.params['num_epochs']), desc='epoch interator'):
            for batch_i, data in enumerate(tqdm(self.train_loader, desc='batch iterator')):
                (x, x_exo, y) = data.values()
                fx = self.forward(x=x, x_exo=x_exo)








