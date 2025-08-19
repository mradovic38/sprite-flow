from abc import ABC, abstractmethod

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.helpers import rgba_to_rgb, resize_images, normalize_to_unit


class EvaluationMetric(ABC):
    """
    Abstract class for evaluation metrics.
    """
    @abstractmethod
    def evaluate_batch(self, real_data: torch.Tensor, generated_data: torch.Tensor,  device: torch.device):
        """
        Evaluates a batch of generated data and updates the evaluation metric.
        :param real_data:
        :param generated_data:
        :param device: device to perform computation on
        """
        pass

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """
        Computes the total evaluation metric.
        :return: evaluation metric over all the generated data
        """
        pass

    @abstractmethod
    def prepare(self, device: torch.device) -> None:
        """
        Prepare metric for evaluation.
        :param device: Device to perform computation on
        """
        pass


class FID(EvaluationMetric):
    def __init__(self, feature=2048, normalize=True, image_size=(299, 299)):
        super().__init__()
        self.metric = FrechetInceptionDistance(feature=feature, normalize=normalize)
        self.image_size = image_size

    def _process_images(self, images: torch.Tensor) -> torch.Tensor:
        images = normalize_to_unit(images)
        rgb = rgba_to_rgb(images)  # convert RGBA -> RGB
        rgb = resize_images(rgb, self.image_size)  # return float32 [0,1]
        return rgb

    def evaluate_batch(self, real_data: torch.Tensor, generated_data: torch.Tensor, device: torch.device) -> torch.Tensor:
        real = self._process_images(real_data).to(device)
        fake = self._process_images(generated_data).to(device)

        self.metric.update(real, real=True)
        self.metric.update(fake, real=False)

    def compute(self) -> torch.Tensor:
        score = self.metric.compute()
        self.metric.reset()
        return score

    def prepare(self, device) -> None:
        self.metric.to(device)
        self.metric.reset()


