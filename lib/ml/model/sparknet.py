from typing import Tuple
from torch import nn
import torch

from lib import logger
from lib.ml.layer.casualresidual import CasualResidual
from lib.ml.layer.casualdownsample import CasualDownSample


class SparkNet(nn.Module):
    def __init__(
            self,
            inputs_shape: Tuple[int, int, int],
            inputs_exo_shape: Tuple[int, int, int],
            targets_shape: Tuple[int, int, int]
    ):
        super().__init__()

        self.name = self.__class__.__name__

        logger.info(f"Instantiate deep neural network model {self.__class__.__name__}: [inputs_shape={inputs_shape}, inputs_exo_shape={inputs_exo_shape}, targets_shape={targets_shape}]")


        self.lstm_1 = nn.LSTM(
            input_size=inputs_shape[1],
            hidden_size=inputs_shape[1],
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,    
            bidirectional=False
        )

        self.sigmoid_1 = nn.Sigmoid()


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




    def forward(
            self,
            inputs: torch.Tensor,
            inputs_exo: torch.Tensor
    )->torch.Tensor:

        inputs = inputs.permute(0,2,1)
        out_1, (_, _) = self.lstm_1(inputs)
        out_1 = out_1.permute(0,2,1)

        out_1 = self.sigmoid_1(out_1)

        #out_1 = self.conv1dres_1(inputs)
        out_2 = self.conv1dsample_1(out_1)
        out_3 = self.conv1dres_2(inputs_exo)
        out_4 = torch.cat((out_2, out_3), dim=1)
        out_5 = self.conv1dsample_2(out_4)

        return out_5















