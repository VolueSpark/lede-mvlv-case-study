from torch import nn
from typing import Tuple, List
import numpy as np
import torch

from lib.ml.layer.reshape import Reshape


class ResNet(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            target_shape: Tuple[int, int, int],
            units: int,
            bias: bool = False,
    ):
        super(ResNet, self).__init__()

        self.flatten_1 = nn.Flatten(
            start_dim=1,
            end_dim=-1
        )

        self.linear_1 = nn.Linear(
            in_features=input_shape[1]*input_shape[2],
            out_features=units,
            bias=bias
        )

        self.relu_1 = nn.ReLU(inplace=True)

        self.linear_2 = nn.Linear(
            in_features=units,
            out_features=target_shape[1]*target_shape[2],
            bias=bias
        )

        self.skip_out = nn.Linear(
            in_features=input_shape[1]*input_shape[2],
            out_features=target_shape[1]*target_shape[2],
            bias=False,
        )

        self.relu_2 = nn.ReLU(inplace=True)

        self.reshape_1 = Reshape(
            output_shape=(target_shape[1], target_shape[2]),
        )

    def forward(self, input):
        identity = self.flatten_1(input)
        x = self.linear_1(identity)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x += self.skip_out(identity)
        x = self.relu_2(x)
        x = self.reshape_1(x)
        return x