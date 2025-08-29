import os
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt

from dataset.augmentation import RandomChannelDropout, RGBAColorJitter


class PixelArtDataset(Dataset):
    """
    Dataset class for pixel art characters.
    """
    def __init__(self, image_paths: List[str], augment: bool = False) -> None:
        """
        :param image_paths
        :param augment: Augment if True, preprocess only if False
        """
        self.image_paths = image_paths
        self.augment = augment

        # Preprocessing only
        self.base_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5, 0.5)),
        ])

        # Augmentation + Preprocessing
        self.augment_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            RGBAColorJitter(0.5, 0.5, 0.6, 0.5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            RandomChannelDropout(0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGBA")

        if self.augment:
            return self.augment_transform(img)
        else:
            return self.base_transform(img)

    def visualize_augmentations(self, num_samples=4, save_path="augmentation_comparison.png") -> None:
        """
        Visualize original vs augmentations
        """
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))

        # Column headers
        col_titles = ['Original', 'Augmented']
        for i, title in enumerate(col_titles):
            axes[0, i].set_title(title, fontsize=12, fontweight='bold')

        for i in range(num_samples):
            # Get a random image
            idx = random.randint(0, len(self.image_paths) - 1)
            img_path = self.image_paths[idx]
            original_image = Image.open(img_path).convert('RGBA')

            # Original
            orig_resized = transforms.Resize((128, 128))(original_image)
            axes[i, 0].imshow(np.array(orig_resized))
            axes[i, 0].axis('off')

            # Heavy augmentation
            heavy_aug = self.augment_transform(original_image)
            # Denormalize for visualization
            heavy_aug = (heavy_aug * 0.5) + 0.5
            heavy_aug = torch.clamp(heavy_aug, 0, 1)
            axes[i, 1].imshow(heavy_aug.permute(1, 2, 0).numpy())
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Augmentation comparison saved to {save_path}")


if __name__ == "__main__":
    some_images = [
        os.path.join("images", f)
        for f in os.listdir("images")
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ][:100]
    dataset = PixelArtDataset(some_images, augment=True)

    dataset.visualize_augmentations(
        save_path="../assets/random/augmentation_comparison.png",
        num_samples=6
    )