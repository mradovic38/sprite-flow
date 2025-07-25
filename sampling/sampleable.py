from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


class Sampleable(ABC):
    """
    Distribution to sample from.
    """
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        :param num_samples
        :return: samples: shape (batch_size, ...)
        """
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        :param shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called

    def sample(self, num_samples) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)
