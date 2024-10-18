from typing import Tuple
from torch import nn
import torch

from lib import logger
from lib.ml.layer.lstmresidual import LSTMResidual
from lib.ml.layer.mlp import MultiLayerPerceptron


class LstmResNet(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            target_shape: Tuple[int, int, int]
    ):
        super().__init__()

        self.name = self.__class__.__name__

        logger.info(f"Instantiate deep neural network model {self.__class__.__name__}: [inputs_shape={input_shape}, targets_shape={target_shape}]")

        self.lstm_res_1 = LSTMResidual(
            input_shape=input_shape,
            target_shape=target_shape
        )

        self.lstm_res_2 = LSTMResidual(
            input_shape=input_shape,
            target_shape=target_shape
        )

        self.dense = MultiLayerPerceptron(
            hidden_layers=[(input_shape[1]*target_shape[2], 50), (50, target_shape[1]*target_shape[2])],
            output_shape=(target_shape[1], target_shape[2]),
        )



    def forward(
            self,
            input: torch.Tensor
    )->torch.Tensor:

        out_1 = self.lstm_res_1(input)
        out_2 = self.dense(out_1)

        return out_2















