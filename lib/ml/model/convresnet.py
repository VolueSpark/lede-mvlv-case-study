from typing import Tuple
from torch import nn
import torch

from lib import logger
from lib.ml.layer.resnet import ResNet
from lib.ml.layer.casualconv1dres import CasualConv1dResidual
from lib.ml.layer.casualdownsample import CasualDownSample

class ConvResNet(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            target_shape: Tuple[int, int, int]
    ):
        super().__init__()

        self.name = self.__class__.__name__

        logger.info(f"Instantiate deep neural network model {self.__class__.__name__}: [inputs_shape={input_shape}, targets_shape={target_shape}]")

        self.casual_conv1d_res_1 = CasualConv1dResidual(
            in_channels=input_shape[1],
            out_channels=input_shape[1],
            kernel_size=3,
            bias=True,

        )

        self.casual_conv1d_res_2 = CasualConv1dResidual(
            in_channels=input_shape[1],
            out_channels=input_shape[1],
            dilation=2,
            kernel_size=3,
            bias=True,
        )


        self.dense_1 = ResNet(
            input_shape=input_shape,
            target_shape=target_shape,
            units=100,
            bias=True
        )



    def forward(
            self,
            input: torch.Tensor
    )->torch.Tensor:

        out_1 = self.casual_conv1d_res_1(input)
        out_2 = self.casual_conv1d_res_2(out_1)
        out_3 = self.dense_1(out_2)

        return out_3