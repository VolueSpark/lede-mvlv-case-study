from typing import Tuple
from torch import nn
import torch

from lib import logger
from lib.ml.layer.casualresidual import CasualResidual
from lib.ml.layer.casualdownsample import CasualDownSample


class SparkNet(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            target_shape: Tuple[int, int, int]
    ):
        super().__init__()

        self.name = self.__class__.__name__

        logger.info(f"Instantiate deep neural network model {self.__class__.__name__}: [inputs_shape={input_shape}, targets_shape={target_shape}]")

        self.conv1dres_1 = CasualResidual(
            inputs_shape=input_shape
        )

        self.conv1dsample_1 = CasualDownSample(
            in_channels=input_shape[1],
            out_channels=target_shape[1],
            in_sequence=input_shape[2],
            out_sequence=target_shape[2],
        )

    def forward(
            self,
            input: torch.Tensor
    )->torch.Tensor:

        out_1 = self.conv1dres_1(input)
        out_2 = self.conv1dsample_1(out_1)

        return out_2















