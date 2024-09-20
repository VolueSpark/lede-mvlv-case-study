from typing import Tuple
from torch import nn
import torch

from lib import logger
from lib.ml.layer.casualresidual import CasualResidual

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class SparkNet(nn.Module):
    def __init__(
            self,
            work_dir: str,
            inputs_shape: Tuple[int, int, int],
            inputs_exo_shape: Tuple[int, int, int],
            targets_shape: Tuple[int, int, int]
    ):
        super().__init__()

        logger.info(f"Instantiate deep neural network model {self.__class__.__name__} with work directory {work_dir} using device {device}")

        self.device = device
        self.work_dir = work_dir

        self.casual_residual_x = CasualResidual(inputs_shape=inputs_shape)
        self.casual_residual_x_exo = CasualResidual(inputs_shape=inputs_exo_shape)


    def forward(
            self,
            inputs: torch.Tensor,
            inputs_exo: torch.Tensor
    )->torch.Tensor:

        outputs = self.casual_residual_x(inputs)
        # TODO add a conv1d layer to reduce channels from 40 to 10 and maxpooling to reduce sequence from 48 to 24. Then contcat output to x_exo and linear layer, reshape 10 x 24 and then loss
        outputs_exo = self.casual_residual_x_exo(inputs_exo)
        return 0














