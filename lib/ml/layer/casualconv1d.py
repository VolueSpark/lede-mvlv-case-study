from torch import nn
import numpy as np
import torch


class CasualConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            groups: int = 1,
            bias: int = True,
            weights: np.ndarray = None
    ):
        super(CasualConv1d, self).__init__()

        self.casual_padding = (kernel_size-1)*dilation

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.casual_padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        if weights is not None:
            with torch.no_grad():
                self.conv1d.weight.copy_(torch.tensor(weights, dtype=torch.float32).view(self.conv1d.weight.data.shape))
            self.conv1d.weight.requires_grad_(True)

    def forward(self, input):
        x = self.conv1d(input)
        return x[:, :, :-self.casual_padding]