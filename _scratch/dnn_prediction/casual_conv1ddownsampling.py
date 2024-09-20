from torch import nn
import numpy as np
import torch

x_in = torch.tensor([[[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]]], dtype=torch.float32)

weights = torch.from_numpy(np.ones((1,3,3)))
in_channels=3
out_channels=1
in_sequence=5
out_sequence=3
kernel_size = in_sequence - out_sequence + 1
conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False)


with torch.no_grad():
    conv1d.weight.copy_(weights)
conv1d.weight.requires_grad_(True)

y_out = conv1d(x_in)

print(x_in)
print(y_out)


