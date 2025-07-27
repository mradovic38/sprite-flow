from abc import ABC, abstractmethod

import torch
from torch import vmap
from torch.func import jacrev


class Alpha(ABC):
    def __init__(self):
        # Make sure alpha_0 = 0
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        # Make sure alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.ones(1,1,1,1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        :param t: time, shape (num_samples, 1, 1, 1)
        :return: alpha_t, shape (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        :param t: time, shape (num_samples, 1, 1, 1)
        :return: d/dt alpha_t, shape (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)


class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates beta_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        :param t: time, shape (num_samples, 1, 1, 1)
        :return: beta_t, shape (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        :param t: time, shape (num_samples, 1, 1, 1)
        :return: d/dt beta_t, shape (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)


class LinearAlpha(Alpha):
    """
    Implements alpha_t = t.
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class LinearBeta(Beta):
    """
    Implements beta_t = 1-t.
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1-t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return - torch.ones_like(t)
