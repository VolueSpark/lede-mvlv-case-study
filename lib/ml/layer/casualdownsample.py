from typing import Tuple
from torch import nn
import torch

class CasualDownSample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            in_sequence: int,
            out_sequence: int
    ):
        super().__init__()
        kernel_size = in_sequence - out_sequence + 1
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.relu(x)
        return x