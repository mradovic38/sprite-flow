from abc import ABC, abstractmethod
import torch


class Sampleable(ABC):
    """
    Distribution to sample from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """

        :param num_samples:
        :return: samples: shape (batch_size, ...)
        """
        pass
