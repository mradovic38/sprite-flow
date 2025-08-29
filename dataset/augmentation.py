from abc import ABC, abstractmethod

import torchvision.transforms as T
from PIL import Image
import random


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Perform augmentation on an image
        :param img: PIL Image
        :return: Augmented PIL Image
        """
        pass


class RGBAColorJitter(Augmentation):
    """
    RGBA implementation of random color jitter transformation
    """
    def __init__(self, brightness: float = 0, contrast: float = 0, saturation: float = 0, hue: float = 0) -> None:
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img: Image.Image) -> Image.Image:
        # Split RGB and A
        rgb, a = img.convert("RGB"), img.getchannel("A")
        # Apply jitter only to RGB
        rgb = self.jitter(rgb)
        # Merge back
        return Image.merge("RGBA", (*rgb.split(), a))


class RandomChannelDropout(Augmentation):
    """
    Randomly zeros out entire color channels
    """
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            channels = list(img.split())
            channel_to_drop = random.randint(0, 2)
            channels[channel_to_drop] = Image.new("L", img.size, 0)
            img = Image.merge("RGBA", channels)
        return img