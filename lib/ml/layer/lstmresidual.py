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
            input_size=input_shape[2],
            hidden_size=target_shape[2],
            num_layers=1,
            batch_first=True
        )

        self.relu_1 = nn.ReLU()

        self.lstm_2 = nn.LSTM(
            input_size=target_shape[2],
            hidden_size=target_shape[2],
            num_layers=1,
            batch_first=True
        )

        self.relu_2 = nn.ReLU()

        self.lstm_skip = nn.LSTM(
            input_size=input_shape[2],
            hidden_size=target_shape[2],
            num_layers=1,
            batch_first=True
        )

    # input/out (#batch, #features, #sequence)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        skip = x
        x, (_,_) = self.lstm_1(x)
        x = self.relu_1(x)
        x, (_,_)  = self.lstm_2(x)
        x_skip, (_,_) = self.lstm_skip(skip)
        x = x+ x_skip
        x = self.relu_2(x)
        return x