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
            weights: np.ndarray = None,
            retain_sequence_length: bool = False,
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


    def forward(self, input):
        x = self.conv1d(input)
        return x[:, :, :-self.casual_padding]



net = CasualConv1d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    dilation=1,
    groups=1,
    bias=False,
    weights=np.array([-1,2,-1])
)

x_in =  torch.tensor([0,1,4,4,1,0], dtype=torch.float32).view(1, 1, 6)
y_out = net(x_in)

print(x_in)
print(y_out)