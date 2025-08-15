from abc import ABC, abstractmethod

import torch
from torch import vmap
from torch.func import jacrev
import math


class Alpha(ABC):
    def __init__(self, inverted: bool = False, atol: float = 1e-3, **kwargs):

        if not inverted:
            # Make sure alpha_0 = 0
            assert torch.allclose(
                self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1), atol=atol
            )
            # Make sure alpha_1 = 1
            assert torch.allclose(
                self(torch.ones(1,1,1,1)), torch.ones(1,1,1,1), atol=atol
            )
        else:
            # Make sure alpha_0 = 1
            assert torch.allclose(
                self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1), atol=atol
            )
            # Make sure alpha_1 = 0
            assert torch.allclose(
                self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1), atol=atol
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
    def __init__(self, inverted: bool = False, atol: float = 1e-3, **kwargs):
        if not inverted:
            # Check beta_0 = 1
            assert torch.allclose(
                self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1), atol=atol
            )
            # Check beta_1 = 0
            assert torch.allclose(
                self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1), atol=atol
            )
        else:
            # Check beta_0 = 0
            assert torch.allclose(
                self(torch.zeros(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1), atol=atol
            )
            # Check beta_1 = 1
            assert torch.allclose(
                self(torch.ones(1, 1, 1, 1)), torch.ones(1, 1, 1, 1), atol=atol
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
        return -torch.ones_like(t)


class CosineAlpha(Alpha):
    """
    Cosine Alpha noise schedule.
    """
    def __init__(self,  s: float = 0.008):
        """
        :param s: small value used for stability
        """
        self.s = s
        self._f_0 = self._calculate_f(torch.tensor(0.0))
        super().__init__(inverted=True)

    def _calculate_f(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate f(t) = cos^2((t + s)/(1 + s) * pi/2)
        :param t: current timestep
        :return: f(t)
        """
        angle = (t  + self.s) / (1 + self.s) * (math.pi / 2)
        return torch.cos(angle) ** 2

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha-bar(t) = f(t) / f(0) where f(t) = cos^2((t + s)/(1 + s) * pi/2)
        """
        f_t = self._calculate_f(t)
        return f_t / self._f_0

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        angle = (t + self.s) / (1 + self.s) * math.pi / 2
        d_angle_dt = math.pi / (2 * (1 + self.s))
        return -2 * torch.cos(angle) * torch.sin(angle) * d_angle_dt / self._f_0


class CosineBeta(Beta):
    """
    Cosine Beta noise schedule.
    """
    def __init__(self, alpha: CosineAlpha, dt: float = 1e-3):
        """
        :param alpha: cosine alpha noise schedule.
        :param dt: small timestep to approximate beta_t via finite differences
        """
        self.alpha = alpha
        self.dt = dt
        super().__init__(inverted=True)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        beta_t = 1 - alpha-bar(t) / alpha-bar(t - dt)
        """
        t_prev = torch.clamp(t - self.dt, min=0.0)
        alpha_t = self.alpha(t)
        alpha_prev = self.alpha(t_prev)
        return 1.0 - alpha_t / alpha_prev

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t_prev = torch.clamp(t - self.dt, min=0.0)
        t_next = torch.clamp(t + self.dt, max=1.0)
        beta_prev = self.__call__(t_prev)
        beta_next = self.__call__(t_next)
        return (beta_next - beta_prev) / (2 * self.dt)