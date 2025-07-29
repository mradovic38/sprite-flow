from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn

from dataset.image_only_dataset import ImageOnlyDataset


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

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)


class PixelArtSampler(nn.Module, Sampleable):
    """
    Sampleable for pixel art character dataset.
    """
    def __init__(self, root_dir: str = '/dataset/images'):
        super().__init__()
        self.dataset = ImageOnlyDataset(root_dir)
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called

    def sample(self, num_samples: int) -> torch.Tensor:
        if num_samples > len(self.dataset):
            raise ValueError(f"Number of samples ({num_samples}) exceeds images size ({len(self.dataset)}).")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples = [self.dataset[i] for i in indices]
        samples = torch.stack(samples).to(self.dummy)

        return samples


