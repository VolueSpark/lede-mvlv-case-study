from typing import Tuple
from torch import nn
import torch

from lib.ml.layer.casualconv1d import CasualConv1d


class CasualResidual(nn.Module):
    def __init__(self, inputs_shape: Tuple[int, int, int], kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()

        self.casual_conv1d_1 = CasualConv1d(
            in_channels=inputs_shape[1],
            out_channels=inputs_shape[1],
            kernel_size=kernel_size
        )

        self.relu_1 = nn.ReLU(inplace=True)

        self.casual_conv1d_2 = CasualConv1d(
            in_channels=inputs_shape[1],
            out_channels=inputs_shape[1],
            kernel_size=kernel_size,
            dilation=4
        )

        self.relu_2 = nn.ReLU(inplace=True)

        self.casual_skip = CasualConv1d(
            in_channels=inputs_shape[1],
            out_channels=inputs_shape[1],
            kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.casual_conv1d_1(x)
        x = self.relu_1(x)
        x = self.casual_conv1d_2(x)
        x += self.casual_skip(identity)
        x = self.relu_2(x)
        return x
