from typing import Tuple
from torch import nn
import torch


def initialize_weights(module):
    if isinstance(module, nn.Conv1d):
        # Apply Glorot (Xavier) normal initialization to weights
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None and len(module.bias.shape) >= 2:
            # Apply Glorot (Xavier) normal initialization to biases
            nn.init.xavier_normal_(module.bias)

class ConvResidual(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            in_sequence: int,
            out_sequence: int,
            dilation: int=1,
            stride: int=1, # TODO hard constraints need to dynamically asign P so K is in integer

    ) -> None:
        super().__init__()

        def resolve_kp(Lin: int, Lout:int, S: int, D: int, name: str):
            padding_down_sample = 1
            while True:
                num = Lin - S*Lout + 2*padding_down_sample
                if (num>0) and (num % D) == 0:
                    kernel_down_sample = (Lin - S*Lout + 2*padding_down_sample)//D+1
                    print(f'Resolved Lin={Lin}, Lout={(Lin + 2*padding_down_sample -D*(kernel_down_sample-1)-1)//S+1} with (K,P)=({kernel_down_sample},{padding_down_sample}) for {name}')
                    return kernel_down_sample, padding_down_sample
                padding_down_sample += 1

        kernel_down_sample, padding_down_sample = resolve_kp(Lin=in_sequence, Lout=out_sequence, S=stride, D=dilation, name='conv1d_1')

        # input (N=#batch,Cin=in_channels,Lin=in_sequence);
        # output (N=#batch, Cout=out_channels, Lout=out_sequence)
        self.conv1d_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            stride=stride,
            kernel_size=kernel_down_sample,
            padding=padding_down_sample,
        )

        self.dropout_1 = nn.Dropout(0)

        self.relu_1 = nn.ReLU()

        kernel_down_sample, padding_down_sample = resolve_kp(Lin=out_sequence, Lout=out_sequence, S=stride, D=dilation, name='conv1d_2')

        self.conv1d_2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            dilation=dilation,
            stride=stride,
            kernel_size=kernel_down_sample,
            padding=padding_down_sample,
        )

        self.dropout_2 = nn.Dropout(0)

        self.relu_2 = nn.ReLU()

        kernel_down_sample, padding_down_sample = resolve_kp(Lin=in_sequence, Lout=out_sequence, S=stride, D=dilation, name='conv1d_3')

        self.conv1d_3 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            stride=stride,
            kernel_size=kernel_down_sample,
            padding=padding_down_sample,
        )

        self.relu_3 = nn.ReLU()

        self.apply(initialize_weights)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1d_1(input)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.conv1d_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x)
        skip_out_x = self.conv1d_3(input)
        x = x+ skip_out_x
        x = self.relu_3(x)
        return x