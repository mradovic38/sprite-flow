from abc import abstractmethod, ABC

import torch
from torch import nn


class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parametrization of the learned vector field u_t^theta(x)
    """
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the conditional vector field.
        :param x: shape (bs, c, h, w)
        :param t: shape (bs, 1, 1, 1)
        :return: u_t^theta(x|y)
        """
        pass