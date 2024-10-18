from typing import Tuple
from torch import nn
import torch

from lib import logger
from lib.ml.layer.mlp import MultiLayerPerceptron
from lib.ml.layer.casualconv1dres import CasualConv1dResidual

class ConvResNet(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            target_shape: Tuple[int, int, int]
    ):
        super().__init__()

        self.name = self.__class__.__name__

        logger.info(f"Instantiate deep neural network model {self.__class__.__name__}: [inputs_shape={input_shape}, targets_shape={target_shape}]")

        self.casuconvres_1 = CasualConv1dResidual(
            in_channels=input_shape[1],
            out_channels=input_shape[1],
            in_sequence=input_shape[2],
            out_sequence=input_shape[2],
            kernel_size=3,
            bias=False,

        )

        self.casuconvres_2 = CasualConv1dResidual(
            in_channels=input_shape[1],
            out_channels=input_shape[1],
            in_sequence=input_shape[2],
            out_sequence=input_shape[2],
            dilation=2,
            kernel_size=3,
            bias=False,
        )

        self.dense = MultiLayerPerceptron(
            hidden_layers=[(input_shape[1]*input_shape[2], target_shape[1]*target_shape[2])],
            #hidden_layers=[(1*input_shape[2], target_shape[1]*target_shape[2])],
            output_shape=(target_shape[1], target_shape[2]),
            bias=True
        )


    def forward(
            self,
            input: torch.Tensor
    )->torch.Tensor:

        out_1 = self.casuconvres_1(input)
        out_2 = self.casuconvres_2(out_1)
        out_3 = self.dense(out_2)

        return out_3