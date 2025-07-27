from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn

from sampling.sampleable import Sampleable, IsotropicGaussian
from sampling.noise_scheduling import Alpha, Beta


class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract class for conditional probability paths.
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        """
        :param p_simple: A simple probability distribution,
        :param p_data: Probability distribution of the data
        """
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z).
        :param t: time, shape (num_samples, 1, 1, 1)
        :return: samples from p_t(x), shape (num_samples, c, h, w)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples) # (num_samples, c, h, w)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (num_samples, c, h, w)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z.
        :param num_samples: number of samples
        :return: z, shape (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z).
        :param z: conditioning variable, shape (num_samples, c, h, w)
        :param t: time, shape (num_samples, 1, 1, 1)
        :return: x: samples from p_t(x|z) ,shape (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z).
        :param x: position variable, shape (num_samples, c, h, w)
        :param z: conditioning variable, shape (num_samples, c, h, w)
        :param t: time, shape (num_samples, 1, 1, 1)
        :return: conditional vector field, shape (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z).
        :param x: position variable, shape (num_samples, c, h, w)
        :param z: conditioning variable, shape (num_samples, c, h, w)
        :param t: time, shape (num_samples, 1, 1, 1)
        :return: conditional score, shape (num_samples, c, h, w)
        """
        pass


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t) # (num_samples, 1, 1, 1)
        beta_t = self.beta(t) # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1, 1, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2