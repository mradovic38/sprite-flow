from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn
from torch.utils.data import Subset
import random

from dataset.image_only_dataset import ImageOnlyDataset


class Sampleable(ABC):
    """
    Distribution to sample from.
    """
    @abstractmethod
    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
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

    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)


class PixelArtSampler(nn.Module, Sampleable):
    """
    Sampleable for pixel art character dataset.
    """
    def __init__(
            self,
            root_dir: str = '/dataset/images',
            random_seed: int = 42,
            train_factor: float = 0.7,
            val_factor:float = 0.15
    ) -> None:

        super().__init__()
        self.full_dataset = ImageOnlyDataset(root_dir)

        # Get all indices
        total_size = len(self.full_dataset)
        indices = list(range(total_size))

        random.seed(random_seed)
        random.shuffle(indices)

        # Split indices -> train_factor : val_factor : train_factor - val_factor
        train_size = int(train_factor * total_size)
        val_size = int(val_factor * total_size)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        # Create subsets
        self.data_train = Subset(self.full_dataset, train_idx)
        self.data_val = Subset(self.full_dataset, val_idx)
        self.data_test = Subset(self.full_dataset, test_idx)

        self.dummy = nn.Buffer(torch.zeros(1))  # for device handling# Will automatically be moved when self.to(...) is called

    def sample(self, num_samples: int, mode: str = 'train', **kwargs) -> torch.Tensor:
        if mode == 'train':
            dataset = self.data_train
        elif mode == 'val':
            dataset = self.data_val
        elif mode == 'test':
            dataset = self.data_test
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of 'train', 'val', 'test'.")

        if num_samples > len(dataset):
            raise ValueError(f"Number of samples ({num_samples}) exceeds images size ({len(dataset)}).")

        indices = torch.randperm(len(dataset))[:num_samples]
        samples = [dataset[i] for i in indices]
        samples = torch.stack(samples).to(self.dummy)

        return samples


