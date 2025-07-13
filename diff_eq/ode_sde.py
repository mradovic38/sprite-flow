from abc import ABC, abstractmethod
import torch

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """

        :param xt:
        :param t:
        :param kwargs:
        :return:
        """
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """

        :param xt:
        :param t:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """

        :param xt:simsi
        :param t:
        :param kwargs:
        :return:
        """
        pass