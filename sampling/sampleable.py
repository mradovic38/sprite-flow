from abc import ABC, abstractmethod
from typing import List, Iterator

import os
import torch
from torch import nn
import random

from dataset.pixel_art_dataset import PixelArtDataset


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


class IterableSampleable(Sampleable, ABC):
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
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        :param shape: shape of sampled data, e.g. [4, 128, 128]
        :param std: standard deviation for RGB sampling
        """
        super().__init__()
        self.shape = shape
        self.std = std

        self.dummy = nn.Buffer(torch.zeros(1))  # Will automatically be moved when self.to(...) is called

    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        device = self.dummy.device
        C, H, W = self.shape
        return self.std * torch.randn(num_samples, C, H, W, device=device)


class PixelArtSampler(nn.Module, IterableSampleable):
    def __init__(self, root_dir: str = "/dataset/images",
                 random_seed: int = 42,
                 train_factor: float = 0.70,
                 val_factor: float = 0.15) -> None:
        super().__init__()

        # Collect image paths
        all_images = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]

        total_size = len(all_images)
        indices = list(range(total_size))
        random.seed(random_seed)
        random.shuffle(indices)

        # Split indices
        train_size = int(train_factor * total_size)
        val_size = int(val_factor * total_size)

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        # Assign datasets
        self.data_train = PixelArtDataset([all_images[i] for i in train_idx], augment=True)
        self.data_val   = PixelArtDataset([all_images[i] for i in val_idx], augment=False)
        self.data_test  = PixelArtDataset([all_images[i] for i in test_idx], augment=False)

        # Dummy buffer for device placement
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples: int, mode: str = "train") -> torch.Tensor:
        if mode == "train":
            dataset = self.data_train
        elif mode == "val":
            dataset = self.data_val
        elif mode == "test":
            dataset = self.data_test
        else:
            raise ValueError(f"Invalid mode '{mode}'.")

        if num_samples > len(dataset):
            raise ValueError(f"num_samples ({num_samples}) > dataset size ({len(dataset)})")

        indices = torch.randperm(len(dataset))[:num_samples]
        samples = [dataset[i] for i in indices]
        return torch.stack(samples).to(self.dummy.device)

    def iterate_dataset(self, batch_size: int, mode: str = "val") -> Iterator[torch.Tensor]:
        if mode == "train":
            dataset = self.data_train
        elif mode == "val":
            dataset = self.data_val
        elif mode == "test":
            dataset = self.data_test
        else:
            raise ValueError(f"Invalid mode '{mode}'.")

        for start_idx in range(0, len(dataset), batch_size):
            batch = [dataset[i] for i in range(start_idx, min(start_idx + batch_size, len(dataset)))]
            yield torch.stack(batch).to(self.dummy.device)
