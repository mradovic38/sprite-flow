from abc import ABC, abstractmethod

import torch
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn as nn
import numpy as np

from helpers import rgba_to_rgb, resize_images


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


class InceptionFeatureExtraction(nn.Module):
    def normalize(self, images) -> torch.Tensor:
        """
        Normalize the input images using InceptionV3's mean and std. Assumes input is in [0, 1] range.
        :param images: RGB images to normalize, shape (num_images, C, H, W)
        :return: normalized images, shape (num_images, C, H, W)
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, -1, 1, 1)
        return (images - mean) / std


class InceptionV3FeatureExtractor(InceptionFeatureExtraction):
    def __init__(self) -> None:
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.features = nn.Sequential(*list(inception.children())[:-1])  # up to avgpool

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class FID(EvaluationMetric):
    def __init__(self, inception_extractor: InceptionFeatureExtraction, extractor_input_img_size=(299, 299)) -> None:
        """
        :param inception_extractor: inception feature extractor
        :param extractor_input_img_size: input image size of the inception feature extractor
        """
        super().__init__()
        self.inception_extractor = inception_extractor
        self.extractor_input_img_size = extractor_input_img_size

    def _calculate_activation_statistics(self, features: np.ndarray):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def _calculate_fid(self, mu1, sigma1, mu2, sigma2) -> torch.Tensor:
        """
        Calculates FID using the arguments provided.
        :param mu1:
        :param sigma1:
        :param mu2:
        :param sigma2:
        :return: calculated FID score
        """
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return torch.tensor(fid, dtype=torch.float32)

    def _process_images(self, images) -> torch.Tensor:
        rgb_images = rgba_to_rgb(images)
        rgb_images = resize_images(rgb_images, self.extractor_input_img_size)
        rgb_images = self.inception_extractor.normalize(rgb_images)
        return rgb_images

    def evaluate(self, real_data: torch.Tensor, generated_data: torch.Tensor, device: torch.device) -> torch.Tensor:
        # Convert RGBA to RGB and rescale [0,1]
        gen_images = self._process_images(generated_data).to(device)
        real_images = self._process_images(real_data).to(device)

        # Extract features for real and generated images (do the same for real images)
        self.inception_extractor.to(device)
        self.inception_extractor.eval()
        with torch.no_grad():
            features_gen = self.inception_extractor(gen_images).cpu().numpy()
            features_real = self.inception_extractor(real_images).cpu().numpy()

        # Calculate statistics
        mu_gen, sigma_gen = self._calculate_activation_statistics(features_gen)
        mu_real, sigma_real = self._calculate_activation_statistics(features_real)

        # Calculate and return FID
        return self._calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)