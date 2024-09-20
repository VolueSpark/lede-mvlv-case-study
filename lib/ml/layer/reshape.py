from torch import nn
import torch

class Reshape(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, x):
        return x.view(x.size(0), *self.output_shape)