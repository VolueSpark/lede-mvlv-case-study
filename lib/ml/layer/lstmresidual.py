from typing import Tuple
from torch import nn
import torch

class LSTMResidual(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            target_shape: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.lstm_1 = nn.LSTM(
            input_size=input_shape[1],
            hidden_size=input_shape[1],
            num_layers=6,
            batch_first=True
        )

        self.relu_1 = nn.ReLU()

        self.lstm_2 = nn.LSTM(
            input_size=input_shape[1],
            hidden_size=input_shape[1],
            num_layers=6,
            batch_first=True
        )

        self.relu_2 = nn.ReLU()

        self.lstm_skip = nn.LSTM(
            input_size=input_shape[1],
            hidden_size=input_shape[1],
            num_layers=1,
            batch_first=True
        )

    # input/out (#batch, #features, #sequence)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute((0,2,1))
        skip = x
        x, (_,_) = self.lstm_1(x)
        x = self.relu_1(x)
        x, (_,_)  = self.lstm_2(x)
        x_skip, (_,_) = self.lstm_skip(skip)
        x = x+ x_skip
        x = self.relu_2(x)
        x=x.permute((0,2,1))
        return x