from torch import nn
import numpy as np
import torch

def initialize_weights(module):
    if isinstance(module, nn.Conv1d):
        # Apply Glorot (Xavier) normal initialization to weights
        nn.init(module.weight)
        if module.bias is not None and len(module.bias.shape) >= 2:
            # Apply Glorot (Xavier) normal initialization to biases
            nn.init.xavier_normal_(module.bias)


class CasualConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            bias: bool = False
    ):
        super(CasualConv1d, self).__init__()

        self.casual_padding = (kernel_size-1)*dilation

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.casual_padding,
            dilation=dilation,
            bias=bias
        )

        initialize_weights(self)

    def forward(self, input):
        x = self.conv1d(input)
        return x[:, :, :-self.casual_padding]
