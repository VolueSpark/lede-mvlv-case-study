from torch import nn
from typing import Tuple, List
import numpy as np
import torch

from lib.ml.layer.reshape import Reshape


class MultiLayerPerceptron(nn.Module):
    def __init__(
            self,
            hidden_layers: List[Tuple[int, int]],
            output_shape: Tuple[int, int],
            bias: bool = False,
    ):
        super(MultiLayerPerceptron, self).__init__()
        assert len(hidden_layers) > 0

        self.layers = nn.ModuleDict()

        for layer_i, (in_features, out_features) in enumerate(hidden_layers):
            if layer_i == 0:
                self.layers['mlp_flatten'] = nn.Flatten(
                    start_dim=1,
                    end_dim=-1
                )
            self.layers[f'mlp_layer_{layer_i}'] = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias
            )
            self.layers[f'mlp_activation_{layer_i}'] = nn.ReLU(
                inplace=True
            )
            if layer_i == len(hidden_layers) - 1:
                self.layers[f'mlp_reshape_layer_{layer_i}'] = Reshape(
                    output_shape=output_shape
                )


    def forward(self, input):
        x= input
        for name, layer in self.layers.items():
            x = layer(x)
        return x