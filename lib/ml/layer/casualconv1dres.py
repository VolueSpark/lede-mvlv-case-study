from typing import Tuple
from torch import nn
import torch

from lib.ml.layer.casualconv1d import CasualConv1d


class CasualConv1dResidual(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            in_sequence: int,
            out_sequence: int,
            kernel_size: int,
            dilation: int = 1,
            bias: bool = False
    ):
        super().__init__()

        self.casual_conv1d_1 = CasualConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias
        )

        self.batch_norm_1 = nn.BatchNorm1d(
            num_features=in_channels
        )

        self.relu_1 = nn.ReLU(inplace=True)

        self.casual_conv1d_2 = CasualConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias
        )

        self.batch_norm_2 = nn.BatchNorm1d(
            num_features=out_channels
        )

        self.relu_2 = nn.ReLU(inplace=True)

        self.skip_conv1d_3 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self.maxpool1d = nn.MaxPool1d(
            kernel_size=in_sequence-out_sequence+1,
            stride=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.casual_conv1d_1(x)
        #x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.casual_conv1d_2(x)
        #x = self.batch_norm_2(x)
        x = x + self.skip_conv1d_3(identity)
        x = self.relu_2(x)
        x = self.maxpool1d(x)
        return x
