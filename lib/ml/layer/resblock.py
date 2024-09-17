from typing import Tuple
from torch import nn
import torch


class ResBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], kernel_size: int=3, stride: int=1) -> None:
        super().__init__()
        padding = (kernel_size-1)//2

        dilation = 1<<0
        padding = dilation * (kernel_size - 1)//2

        self.conv1d_1 = nn.Conv1d(in_channels=input_shape[1],
                                  out_channels=input_shape[1],
                                  kernel_size=kernel_size,
                                  dilation=dilation,
                                  padding=padding)
        self.relu_1 = nn.ReLU(inplace=True)

        dilation = 1<<1
        padding = dilation * (kernel_size - 1)//2

        self.conv1d_2 = nn.Conv1d(in_channels=input_shape[1],
                                  out_channels=input_shape[1],
                                  kernel_size=kernel_size,
                                  dilation=dilation,
                                  padding=padding)
        self.relu_2 = nn.ReLU(inplace=True)

        dilation = 1<<0
        padding = dilation * (kernel_size - 1)//2

        self.skip_connection = nn.Conv1d(in_channels=input_shape[1],
                                         out_channels=input_shape[1],
                                         kernel_size=kernel_size,
                                         dilation=dilation,
                                         padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1d_1(x)
        x = self.relu_1(x)
        x = self.conv1d_2(x)

        x += self.skip_connection(identity)
        x = self.relu_2(x)

        return x




