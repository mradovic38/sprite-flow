from abc import ABC, abstractmethod
from typing import List, Iterator

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


class IterableSampleable(ABC, Sampleable):
    """
    Sampleable for finite datasets.
    """
    @abstractmethod
    def iterate_dataset(self, batch_size: int, mode: str = 'val') -> Iterator[torch.Tensor]:
        """
        Iterates over the entire dataset (val/test) in mini-batches of size `batch_size`.
        :param batch_size: number of images per batch
        :param mode: 'train', 'val', or 'test'
        :return: yields batches as torch tensors
        """
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: List[int], std: float = 1.0, binary_alpha: bool = True):
        """
        :param shape: shape of sampled data, e.g. [4, 128, 128]
        :param std: standard deviation for RGB sampling
        :param binary_alpha: whether to sample alpha as binary
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.binary_alpha = binary_alpha

        self.dummy = nn.Buffer(torch.zeros(1))  # Will automatically be moved when self.to(...) is called

        if self.binary_alpha and self.shape[0] < 4:
            raise ValueError("binary_alpha=True requires at least 4 channels (RGBA).")

    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        device = self.dummy.device
        C, H, W = self.shape

        print(self.binary_alpha)
        print(C)
        if self.binary_alpha and C >= 4:
            rgb = self.std * torch.randn(num_samples, C - 1, H, W, device=device)
            alpha = torch.randint(0, 2, (num_samples, 1, H, W), device=device).float()
            return torch.cat([rgb, alpha], dim=1)
        else:
            return self.std * torch.randn(num_samples, C, H, W, device=device)


class PixelArtSampler(nn.Module, IterableSampleable):
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

        self.dummy = nn.Buffer(torch.zeros(1))  # Will automatically be moved when self.to(...) is called

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
        samples = torch.stack(samples).to(self.dummy.device)

        return samples

    def iterate_dataset(self, batch_size: int, mode: str = 'val') -> Iterator[torch.Tensor]:
        if mode == 'train':
            dataset = self.data_train
        elif mode == 'val':
            dataset = self.data_val
        elif mode == 'test':
            dataset = self.data_test
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of 'train', 'val', 'test'.")

        device = self.dummy.device
        for start_idx in range(0, len(dataset), batch_size):
            batch = [dataset[i] for i in range(start_idx, min(start_idx + batch_size, len(dataset)))]
            yield torch.stack(batch).to(device)