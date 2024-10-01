from typing import Tuple
from torch import nn
import torch

from lib import logger
from lib.ml.layer.mlp import MultiLayerPerceptron
from lib.ml.layer.casualresidual import CasualResidual
from lib.ml.layer.casualdownsample import CasualDownSample

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
            inputs_shape: Tuple[int, int, int],
            inputs_exo_shape: Tuple[int, int, int],
            targets_shape: Tuple[int, int, int]
    ):
        super().__init__()

        self.name = self.__class__.__name__

        logger.info(f"Instantiate deep neural network model {self.__class__.__name__} with work directory {work_dir} using device {device}")

        self.device = device
        self.work_dir = work_dir

        self.conv1dres_1 = CasualResidual(
            inputs_shape=inputs_shape
        )

        self.conv1dsample_1 = CasualDownSample(
            in_channels=inputs_shape[1],
            out_channels=targets_shape[1],
            in_sequence=inputs_shape[2],
            out_sequence=targets_shape[2],
        )

        self.conv1dres_2 = CasualResidual(
            inputs_shape=inputs_exo_shape
        )

        self.conv1dsample_2 = CasualDownSample(
            in_channels=inputs_shape[1],
            out_channels=targets_shape[1],
            in_sequence=targets_shape[2],
            out_sequence=targets_shape[2],
        )

        self.mlp = MultiLayerPerceptron(
            hidden_layers=[(targets_shape[1]*targets_shape[2], 100), (100, targets_shape[1]*targets_shape[2])],
            output_shape=(targets_shape[1], targets_shape[2]),
        )


    def forward(
            self,
            inputs: torch.Tensor,
            inputs_exo: torch.Tensor
    )->torch.Tensor:

        out_1 = self.conv1dres_1(inputs)
        out_2 = self.conv1dsample_1(out_1)
        out_3 = self.conv1dres_2(inputs_exo)
        out_4 = torch.cat((out_2, out_3), dim=1)
        out_5 = self.conv1dsample_2(out_4)
        out_6 =self.mlp(out_5)

        return out_6















