from abc import ABC, abstractmethod

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.helpers import rgba_to_rgb, resize_images


class EvaluationMetric(ABC):
    """
    Abstract class for evaluation metrics.
    """
    @abstractmethod
    def evaluate(self, real_data: torch.Tensor, generated_data: torch.Tensor,  device: torch.device) -> torch.Tensor:
        """
        Evaluates generated data over the metric.
        :param real_data:
        :param generated_data:
        :param device: device to perform computation on
        :return: evaluation score
        """
        pass


class FID(EvaluationMetric):
    def __init__(self, feature=2048, normalize=True, image_size=(299, 299)):
        super().__init__()
        self.metric = FrechetInceptionDistance(feature=feature, normalize=normalize)
        self.image_size = image_size

    def _process_images(self, images: torch.Tensor) -> torch.Tensor:
        rgb = rgba_to_rgb(images)
        rgb = resize_images(rgb, self.image_size)  # Ensure this returns float in [0, 1]
        return rgb

    def evaluate(self, real_data: torch.Tensor, generated_data: torch.Tensor, device: torch.device) -> torch.Tensor:
        real = self._process_images(real_data).to(device)
        fake = self._process_images(generated_data).to(device)

        self.metric = self.metric.to(device)
        self.metric.reset()
        self.metric.update(real, real=True)
        self.metric.update(fake, real=False)

        return self.metric.compute()